# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-bandwidth transformer (arXiv:2608.08888) on top of the Qwen3 dense model.

At each position the previous position's top-layer state (the final-norm output that feeds the LM head) is fused
with the current token embedding through a gated linear unit, ``W_u h_{t-1} * sigmoid(W_g rmsnorm(e_t))``, and
re-enters the stack as the input. Training keeps parallel teacher forcing with a multi-pass objective: pass 1 is
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
"""

from dataclasses import dataclass
from typing import Optional, cast

import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel


@LmConfig.register_subclass("qwen3_fbt")
@dataclass(frozen=True)
class FullBandwidthQwen3Config(Qwen3Config):
    feedback_passes: int = 2
    feedback_noise: float = 0.02  # jitter on the fed-back state during training, Uniform[-noise, noise]
    # Residual form for continued pretraining from a plain checkpoint: the feedback pass takes
    # ``e_t + alpha * (W_u h_{t-1} * sigmoid(W_g rmsnorm(e_t)))`` with no RMSNorm on the sum and a learnable scalar
    # ``alpha`` starting at ``feedback_alpha_init`` (0 makes pass 2 identical to pass 1 at the start, so the loss
    # begins at the checkpoint's level and the feedback path is switched on gradually). The paper's GLU-only form has
    # no identity point, so it cannot be initialised from a plain transformer without a loss jump.
    feedback_residual: bool = False
    feedback_alpha_init: float = 0.0

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return FullBandwidthQwen3LMHeadModel

    def flops_per_token(self, vocab_size: int, context_length: int):
        # Every pass runs the full stack and the LM head, so the reported MFU counts them all.
        return self.feedback_passes * super().flops_per_token(vocab_size, context_length)


class FullBandwidthQwen3LMHeadModel(Qwen3LMHeadModel):
    w_u: hnn.Linear
    w_g: hnn.Linear
    input_norm: hnn.RmsNorm
    alpha: Optional[NamedArray]  # residual-form gate scalar; None in the paper's GLU-only form

    @classmethod
    def init(cls, Vocab, config: FullBandwidthQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)  # same base initialisation as the baseline for this key
        k_u, k_g = jrandom.split(jrandom.fold_in(key, 1), 2)
        Fused = config.Embed.alias("fused_embed")
        w_u = hnn.Linear.init(In=config.Embed, Out=Fused, key=k_u, use_bias=False)
        w_g = hnn.Linear.init(In=config.Embed, Out=Fused, key=k_g, use_bias=False)
        alpha = hax.named(jnp.asarray(config.feedback_alpha_init, dtype=jnp.float32), ()) if config.feedback_residual else None
        return cls(base.transformer, base.embeddings, base.lm_head, w_u, w_g, config.mk_LayerNorm(config.Embed), alpha)

    @named_call
    def _passes(self, input_ids: NamedArray, attn_mask, *, key, pos_ids, train: bool) -> list[NamedArray]:
        """Top-layer states of every pass; ``train`` adds the jitter noise and the random plain prefix."""
        cfg = cast(FullBandwidthQwen3Config, self.config)
        Pos = input_ids.resolve_axis("position")
        k_model, k_noise, k_prefix = maybe_rng_split(key, 3) if key is not None else (None, None, None)
        e = self.embeddings.embed(input_ids)
        h = self.transformer(e, attn_mask=attn_mask, key=k_model, pos_ids=pos_ids)
        states = [h]
        gate = hnn.sigmoid(self.w_g(self.input_norm(e)).rename({"fused_embed": "embed"}))
        batch_axes = tuple(ax for ax in input_ids.axes if ax.name != Pos.name)
        position = hax.arange(Pos).broadcast_axis(batch_axes)
        for i in range(1, cfg.feedback_passes):
            fed_back = h
            if train and cfg.feedback_noise > 0:
                noise = hax.random.uniform(jrandom.fold_in(k_noise, i), h.axes, minval=-cfg.feedback_noise, maxval=cfg.feedback_noise)
                fed_back = fed_back + noise.astype(h.dtype)
            # Position t receives h_{t-1}; position 0 (which roll fills with h_{T-1}) is plain below.
            fused = self.w_u(hax.roll(fed_back, 1, Pos)).rename({"fused_embed": "embed"}) * gate
            plain_upto = hax.random.randint(jrandom.fold_in(k_prefix, i), batch_axes, 0, Pos.size) if train else 0
            if cfg.feedback_residual:
                # Plain prefix = branch masked to zero, so those positions see exactly the pass-1 input.
                x = e + self.alpha.astype(e.dtype) * hax.where(position <= plain_upto, 0.0, fused)
                h = self.transformer(x, attn_mask=attn_mask, key=k_model, pos_ids=pos_ids)
            else:
                x = hax.where(position <= plain_upto, e, fused)
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
