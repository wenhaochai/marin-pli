# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-bandwidth transformer (arXiv:2608.08888) on top of the Qwen3 dense model, plus readout variants.

Paper form: at each position the previous position's top-layer state (the final-norm output that feeds the LM head)
is fused with the current token embedding through a gated linear unit, ``W_u h_{t-1} * sigmoid(W_g rmsnorm(e_t))``,
and re-enters the stack as the input. Training keeps parallel teacher forcing with a multi-pass objective: pass 1 is
the ordinary forward pass, pass k shifts pass k-1's states one position right, fuses them with the embeddings and
re-runs the full stack; the loss is ``ntp(pass 1) + mean_k ntp(pass k)``. Gradients flow through the earlier
passes (no stop-gradient). Following the paper's Listing 3, the fed-back state gets uniform jitter noise, the
fused input goes through a shared RMSNorm, and a random plain prefix per sequence (positions ``<= p``, ``p``
uniform over the sequence) keeps plain token embeddings so training covers the plain-prefill-then-fused regime
of inference. Position 0 always sees the plain embedding.

Every batch runs the same fixed number of passes (``feedback_passes``, 2 here); this deliberately drops the
paper's mixed 1/2/3-pass schedule. Evaluation (``key=None``) reports the fused final pass without noise or a
random prefix, which is the model's decoding-time behaviour; the logged train loss is the two-pass sum. Depth
scaling from the paper is not applied: Qwen3's hybrid norm and final RMSNorm already keep the fed-back state
O(1) per coordinate.

Readout variants, all built so that a plain Qwen3 checkpoint continues without a loss jump (every new path is
multiplied by a learnable scalar ``alpha`` that starts at ``feedback_alpha_init``, 0 by default, and the plain
path is untouched):

* ``feedback_residual``: input-level term ``e_t + alpha * (W_u h_{t-1} * sigmoid(W_g rmsnorm(e_t)))`` with no
  RMSNorm on the sum (the paper's GLU-only form has no identity point).
* ``feedback_input_layers = k > 1``: the input-level term reads a softmax-weighted mix of the last ``k`` layers'
  pass-1 residual states (passed through the final norm) instead of the top layer alone.
* ``feedback_layerwise``: in every layer ``l`` of a feedback pass, the residual stream at position ``t`` receives
  ``alpha_l * (P_l h^{(l)}_{t-1} * sigmoid(G_l rmsnorm_l(x^{(l)}_t)))`` where ``h^{(l)}`` is pass 1's output of the
  same layer, so each layer continues from where the same layer got to one position earlier (L x D numbers read
  out per position instead of D). ``feedback_input=False`` drops the input-level term to isolate this path.

The transformer's own parameters keep their tree paths, so a plain checkpoint loads into every variant with
``allow_partial_checkpoint`` and the new leaves keep their initial values.
"""

from dataclasses import dataclass
from typing import Optional, cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel


@LmConfig.register_subclass("qwen3_fbt")
@dataclass(frozen=True)
class FullBandwidthQwen3Config(Qwen3Config):
    feedback_passes: int = 2
    feedback_noise: float = 0.02  # jitter on the fed-back states during training, Uniform[-noise, noise]
    feedback_residual: bool = False  # input-level residual form with a scalar gate (identity at alpha=0)
    feedback_alpha_init: float = 0.0
    feedback_input: bool = True  # keep the input-level term (paper form or residual form)
    feedback_input_layers: int = 1  # >1: the input-level term reads a softmax mix of the last k layers (residual form only)
    feedback_layerwise: bool = False  # per-layer aligned injection with per-layer scalar gates (identity at alpha=0)

    def __post_init__(self):
        if self.feedback_input_layers > 1 and not self.feedback_residual:
            raise ValueError("feedback_input_layers > 1 needs feedback_residual=True")
        if self.feedback_layerwise and not self.feedback_residual:
            raise ValueError("feedback_layerwise needs feedback_residual=True")
        if not self.feedback_input and not self.feedback_layerwise:
            raise ValueError("feedback_input=False needs feedback_layerwise=True (otherwise there is no feedback)")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return FullBandwidthQwen3LMHeadModel

    def flops_per_token(self, vocab_size: int, context_length: int):
        # Every pass runs the full stack and the LM head, so the reported MFU counts them all (the small adapter
        # projections are not counted).
        return self.feedback_passes * super().flops_per_token(vocab_size, context_length)


class FeedbackAdapter(eqx.Module):
    """Per-layer readout of the previous position's same-layer pass-1 state: ``alpha * (P fb) * sigmoid(G rmsnorm(x))``."""

    proj: hnn.Linear
    gate: hnn.Linear
    norm: hnn.RmsNorm
    alpha: NamedArray

    @staticmethod
    def init(config: FullBandwidthQwen3Config, *, key) -> "FeedbackAdapter":
        k_p, k_g = jrandom.split(key, 2)
        Fused = config.Embed.alias("fused_embed")
        proj = hnn.Linear.init(In=config.Embed, Out=Fused, key=k_p, use_bias=False)
        gate = hnn.Linear.init(In=config.Embed, Out=Fused, key=k_g, use_bias=False)
        alpha = hax.named(jnp.asarray(config.feedback_alpha_init, dtype=jnp.float32), ())
        return FeedbackAdapter(proj, gate, config.mk_LayerNorm(config.Embed), alpha)

    def branch(self, x: NamedArray, fb: NamedArray) -> NamedArray:
        g = hnn.sigmoid(self.gate(self.norm(x)).rename({"fused_embed": "embed"}))
        return self.alpha.astype(x.dtype) * (self.proj(fb).rename({"fused_embed": "embed"}) * g)


class FullBandwidthQwen3LMHeadModel(Qwen3LMHeadModel):
    w_u: hnn.Linear
    w_g: hnn.Linear
    input_norm: hnn.RmsNorm
    alpha: Optional[NamedArray]  # residual-form input gate scalar; None in the paper's GLU-only form
    fb_adapters: Optional[Stacked[FeedbackAdapter]]  # feedback_layerwise
    mix_logits: Optional[NamedArray]  # feedback_input_layers > 1: softmax weights over the last k layers

    @classmethod
    def init(cls, Vocab, config: FullBandwidthQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)  # same base initialisation as the baseline for this key
        k_u, k_g, k_l = jrandom.split(jrandom.fold_in(key, 1), 3)
        Fused = config.Embed.alias("fused_embed")
        w_u = hnn.Linear.init(In=config.Embed, Out=Fused, key=k_u, use_bias=False)
        w_g = hnn.Linear.init(In=config.Embed, Out=Fused, key=k_g, use_bias=False)
        alpha = hax.named(jnp.asarray(config.feedback_alpha_init, dtype=jnp.float32), ()) if config.feedback_residual else None
        fb_adapters = None
        if config.feedback_layerwise:
            fb_adapters = Stacked.init(config.Layers, FeedbackAdapter, gradient_checkpointing=config.gradient_checkpointing)(
                config, key=shaped_rng_split(k_l, config.num_layers)
            )
        mix_logits = None
        if config.feedback_input_layers > 1:
            # Start top-heavy so the mix is close to the paper's top-layer readout.
            k = config.feedback_input_layers
            mix_logits = hax.named(jnp.asarray([0.0] * (k - 1) + [3.0], dtype=jnp.float32), ("mix",))
        return cls(base.transformer, base.embeddings, base.lm_head, w_u, w_g, config.mk_LayerNorm(config.Embed), alpha, fb_adapters, mix_logits)

    def _stack(self, x, attn_mask, *, key, pos_ids, fb_layers, plain):
        """Run the decoder stack; returns (top state after the final norm, per-layer outputs stacked on Layers).

        ``fb_layers`` (Layers axis) and ``plain`` (bool, batch x position) switch on the per-layer injection.
        """
        layers = self.transformer.layers
        assert isinstance(layers, Stacked), "layerwise feedback expects a Stacked transformer"
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None

        if fb_layers is None:

            def step(block, x, mask, *, key, pos_ids):
                y = block(x, mask, key=key, pos_ids=pos_ids)
                return y, y

            out, per_layer = layers.scan_via(step)(x, attn_mask, key=keys, pos_ids=pos_ids)
        else:
            adapters = cast(Stacked[FeedbackAdapter], self.fb_adapters).stacked

            def step_fb(block, x, adapter, fb, plain, mask, *, key, pos_ids):
                x = x + hax.where(plain, 0.0, adapter.branch(x, fb))
                y = block(x, mask, key=key, pos_ids=pos_ids)
                return y, y

            out, per_layer = layers.scan_via(step_fb)(x, adapters, fb_layers, plain, attn_mask, key=keys, pos_ids=pos_ids)
        return self.transformer.norm(out), per_layer

    @named_call
    def _passes(self, input_ids: NamedArray, attn_mask, *, key, pos_ids, train: bool) -> list[NamedArray]:
        """Top-layer states of every pass; ``train`` adds the jitter noise and the random plain prefix."""
        cfg = cast(FullBandwidthQwen3Config, self.config)
        Pos = input_ids.resolve_axis("position")
        Layers = cfg.Layers
        k_model, k_noise, k_prefix = maybe_rng_split(key, 3) if key is not None else (None, None, None)
        e = self.embeddings.embed(input_ids)
        need_layers = cfg.feedback_layerwise or cfg.feedback_input_layers > 1
        if need_layers:
            h, per_layer = self._stack(e, attn_mask, key=k_model, pos_ids=pos_ids, fb_layers=None, plain=None)
        else:
            h, per_layer = self.transformer(e, attn_mask=attn_mask, key=k_model, pos_ids=pos_ids), None
        states = [h]
        gate = hnn.sigmoid(self.w_g(self.input_norm(e)).rename({"fused_embed": "embed"}))
        batch_axes = tuple(ax for ax in input_ids.axes if ax.name != Pos.name)
        position = hax.arange(Pos).broadcast_axis(batch_axes)

        def jitter(x, i, salt):
            if train and cfg.feedback_noise > 0:
                noise = hax.random.uniform(jrandom.fold_in(k_noise, i * 2 + salt), x.axes, minval=-cfg.feedback_noise, maxval=cfg.feedback_noise)
                x = x + noise.astype(x.dtype)
            return x

        for i in range(1, cfg.feedback_passes):
            plain_upto = hax.random.randint(jrandom.fold_in(k_prefix, i), batch_axes, 0, Pos.size) if train else 0
            plain = position <= plain_upto  # position 0 (which roll fills with the last position) is always plain
            # Input-level readout: the top state, or a softmax mix of the last k layers passed through the final norm.
            if cfg.feedback_input_layers > 1:
                k = cfg.feedback_input_layers
                last_k = per_layer.slice(Layers, start=Layers.size - k, length=k).rename({Layers.name: "mix"})
                weights = hnn.softmax(cast(NamedArray, self.mix_logits), axis="mix").astype(last_k.dtype)
                fb_top = self.transformer.norm(hax.dot(weights, last_k, axis="mix"))
            else:
                fb_top = h
            fused = self.w_u(hax.roll(jitter(fb_top, i, 0), 1, Pos)).rename({"fused_embed": "embed"}) * gate
            if cfg.feedback_residual:
                x = e
                if cfg.feedback_input:
                    x = x + cast(NamedArray, self.alpha).astype(e.dtype) * hax.where(plain, 0.0, fused)
                if cfg.feedback_layerwise:
                    fb_layers = hax.roll(jitter(cast(NamedArray, per_layer), i, 1), 1, Pos)
                    h, per_layer = self._stack(x, attn_mask, key=k_model, pos_ids=pos_ids, fb_layers=fb_layers, plain=plain)
                elif need_layers:
                    h, per_layer = self._stack(x, attn_mask, key=k_model, pos_ids=pos_ids, fb_layers=None, plain=None)
                else:
                    h = self.transformer(x, attn_mask=attn_mask, key=k_model, pos_ids=pos_ids)
            else:
                x = hax.where(plain, e, fused)
                h = self.transformer(self.input_norm(x), attn_mask=attn_mask, key=k_model, pos_ids=pos_ids)
            states.append(h)
        return states

    def activations(self, input_ids, attn_mask=None, *, key=None, pos_ids=None):  # type: ignore[override]
        return self._passes(input_ids, attn_mask, key=key, pos_ids=pos_ids, train=False)[-1]

    def compute_next_token_loss(  # type: ignore[override]
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = cast(Optional[hax.ReductionFunction], hax.mean),
        reduction_axis: Optional[hax.AxisSelection] = None,
        logsumexp_weight: Optional[float] = None,
        loss_dtype: Optional[jnp.dtype] = jnp.float32,
        logit_soft_cap: Optional[float] = None,
    ):
        train = key is not None  # the trainer passes a key; evaluation does not

        def ntp(h):
            return maybe_fused_next_token_loss(
                self.Pos,
                self.Embed,
                self.Vocab,
                h,
                self.get_lm_head(),
                example.tokens,
                loss_weight=example.loss_weight,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )

        states = self._passes(example.tokens, example.attn_mask, key=key, pos_ids=None, train=train)
        if not train:
            return ntp(states[-1])
        # The trainer differentiates through this function, so no host logging here (io callbacks reject JVP): the
        # logged train loss is the two-pass sum; eval reports the fused pass alone.
        losses = [ntp(h) for h in states]
        return losses[0] + sum(losses[1:]) / (len(losses) - 1)
