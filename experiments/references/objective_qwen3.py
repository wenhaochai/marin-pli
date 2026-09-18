# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-NTP auxiliary objectives on top of the Qwen3 dense model, for the objective hill-climb.

The forward model, data, batch, schedule and optimizer are the muonh_qwen3_scaling baseline's; only the training
objective changes, and evaluation (``key=None``) runs the plain forward next-token loss, so eval/paloma numbers are
directly comparable with the baseline. Two objectives, switchable independently:

* ``twin`` (Twin Networks, Serdyuk et al. ICLR 2018; roots in Schuster & Paliwal 1997 bidirectional RNNs): a second
  Qwen3 of the same shape reads every sequence backwards and is trained on its own reversed next-token loss. The
  forward top state ``h_t`` (which predicts ``x_{t+1}``) is pulled, through an affine map, towards the backward top
  state that has read ``x_{>= t+2}`` and therefore also predicts ``x_{t+1}`` -- neither state has seen the target
  token. The backward states are stop-gradiented, so the penalty shapes only the forward model; the backward model
  is scaffolding and plays no part at eval. Pairs that straddle a document boundary are masked out.

* ``sr`` (successor representation, Dayan 1993, learned by temporal differences, Sutton 1988): a linear head on
  ``h_t`` predicts the discounted sum of future token embeddings ``psi_t = sum_k gamma^k e(x_{t+k})``, fitted by the
  TD target ``e(x_{t+1}) + gamma * sg(psi_{t+1})`` (bootstrap dropped at the last token of a document). Embeddings
  are RMS-normalised per position before use as targets, otherwise their 0.02 scale makes the term vanish next to
  the cross-entropy. The target is stop-gradiented in full, including the embeddings.

* ``pi`` (predictive information: Becker & Hinton 1992 IMAX, Bialek-Nemenman-Tishby 1999; InfoNCE form of van den
  Oord et al. 2018): the forward state ``h_t``, through a linear map, must identify its own future state ``h_{t+k}``
  among ``pi_negatives`` states drawn at random from the whole batch. Both sides are L2-normalised and both receive
  gradient (IMAX's two modules co-adapting to maximise their mutual information). Anchors whose ``t+k`` lies in
  another document or past the end are masked out; a random negative can coincide with the positive or sit in the
  same document, which InfoNCE tolerates.

* ``eos`` (distance to document end; user proposal 2026-09-18): at each position ``h_t`` predicts how far the current
  document is from ending, ``d_t`` = index of the first EOS strictly after ``t`` minus ``t`` (>= 1), as a 13-way
  cross-entropy over log2-spaced bins {1}, {2}, {3-4}, {5-8}, ..., {2049-4096}. This is discourse-position
  information that NTP never asks for explicitly (``d_t = 1`` coincides with the NTP EOS logit; ``d_t >= 2`` is
  new). Masked: positions with no EOS later in the window (the truncated last document -- unknowable) and positions
  that are themselves EOS (the next EOS belongs to the following document, which cross-document masking hides).

Loss = forward NTP [+ backward NTP + twin_weight * twin] [+ sr_weight * sr] [+ pi_weight * pi] [+ eos_weight * eos].
The logged train loss is that sum; eval reports the forward NTP alone.

HEAD PARAMETRISATION (``free_heads``, default True). MuonH keeps every ``hnn.Linear`` weight at exactly its
initialisation Frobenius norm (levanter/optim/muonh.py: ``p_new = p_int / norm(p_int) * norm(p)``) and updates it
with orthogonalised steps, and its ``create_mask`` routes ANY ``hnn.Linear`` there. Auxiliary heads built as
``hnn.Linear`` therefore cannot change scale at all: in the 2026-09-18 130m runs ``sr_head.weight`` sat at 22.387
+/- 0.002 for the whole run, ``pi_proj`` and ``twin_proj`` likewise, the SR MSE plateaued at 0.88, and the trunk
compensated by growing the final-norm gain 5-11% over the baseline -- the lm_head's input scale -- which is a
plausible route to the NTP losses those runs showed. With ``free_heads`` the heads are plain NamedArray weights
(same init as ``hnn.Linear``), which the mask labels ``adam``: free norm, Adam updates at ``adam_lr``. The
optimizer itself is untouched; only parameters the baseline does not have are labelled differently. Run ids carry
``-fh`` so these runs never collide with or resume from the pinned-head ones.
"""

from dataclasses import dataclass
from typing import Any, Optional, cast

import equinox as eqx
import jax
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


@LmConfig.register_subclass("qwen3_objective")
@dataclass(frozen=True)
class ObjectiveQwen3Config(Qwen3Config):
    twin: bool = False
    twin_weight: float = 0.1
    # Forward h_t is matched to the backward state that has read x_{>= t+twin_offset}. 2 = both predict x_{t+1}.
    twin_offset: int = 2
    sr: bool = False
    sr_weight: float = 0.1
    sr_gamma: float = 0.9
    pi: bool = False
    pi_weight: float = 0.1
    pi_k: int = 4  # anchor h_t identifies h_{t+k}
    pi_tau: float = 0.1  # InfoNCE temperature on cosine logits
    pi_negatives: int = 512  # states sampled from the whole batch as shared negatives
    eos: bool = False
    eos_weight: float = 0.1
    eos_id: int = 128001  # marin tokenizer (same ids as llama3); checked against the tokenizer in the CPU smoke
    eos_bins: int = 13  # bin b holds d in (2^(b-1), 2^b]; 13 bins cover 1..4096
    free_heads: bool = True  # auxiliary heads as plain arrays in MuonH's adam group (norm free); False = hnn.Linear (norm pinned)

    def __post_init__(self):
        if self.twin_offset < 1:
            raise ValueError("twin_offset must be >= 1 (1 lets the backward state see the target token)")
        if not (0.0 <= self.sr_gamma < 1.0):
            raise ValueError("sr_gamma must be in [0, 1)")
        if self.pi_k < 1 or self.pi_tau <= 0 or self.pi_negatives < 1:
            raise ValueError("pi_k >= 1, pi_tau > 0 and pi_negatives >= 1 are required")
        if self.eos_bins < 2:
            raise ValueError("eos_bins must be >= 2")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return ObjectiveQwen3LMHeadModel

    def flops_per_token(self, vocab_size: int, context_length: int):
        # The backward model is a full second stack plus LM head; the two small heads are not counted.
        return (2 if self.twin else 1) * super().flops_per_token(vocab_size, context_length)


def _flip(x: NamedArray, axis) -> NamedArray:
    """Reverse ``x`` along one named axis (haliax has no flip)."""
    ax = x.resolve_axis(axis)
    return hax.named(jnp.flip(x.array, x.axes.index(ax)), x.axes)


def _flipped_mask(mask) -> AttentionMask:
    """The attention mask of the reversed sequence: causal again, with the segment ids reversed in position."""
    if not isinstance(mask, AttentionMask) or mask.explicit_mask is not None:
        raise NotImplementedError("objective_qwen3 expects a causal AttentionMask, not an explicit mask array")
    out = AttentionMask.causal(sliding_window=mask.sliding_window)
    if mask.segment_ids is None:
        return out
    q_ids, _ = mask.segment_ids  # (q, kv) NamedArrays; training packs identical ids for both
    return out.with_segment_ids(_flip(q_ids, "position"))


def _segment_ids(example: LmExample) -> Optional[NamedArray]:
    mask = example.attn_mask
    if not isinstance(mask, AttentionMask) or mask.segment_ids is None:
        return None
    return mask.segment_ids[0]


def _masked_mean(per_position: NamedArray, valid: NamedArray) -> NamedArray:
    return hax.sum(per_position * valid) / hax.maximum(hax.sum(valid), 1.0)


class FreeLinear(eqx.Module):
    """A linear map stored as plain NamedArrays so MuonH's mask puts it in the adam group (no norm pinning).

    Initialised exactly like ``hnn.Linear.init`` (the arrays are taken from one), so a pinned-head run and a
    free-head run start from identical heads and differ only in how the optimizer treats them.
    """

    weight: NamedArray
    bias: Optional[NamedArray]
    In: hax.Axis = eqx.field(static=True)
    Out: hax.Axis = eqx.field(static=True)

    @staticmethod
    def init(In: hax.Axis, Out: hax.Axis, *, key, use_bias: bool) -> "FreeLinear":
        lin = hnn.Linear.init(In=In, Out=Out, key=key, use_bias=use_bias)
        return FreeLinear(lin.weight, lin.bias, In, Out)

    def __call__(self, x: NamedArray) -> NamedArray:
        y = hax.dot(x, self.weight, axis=self.In.name)
        return y + self.bias if self.bias is not None else y


def _head(In: hax.Axis, Out: hax.Axis, *, key, use_bias: bool, free: bool):
    return FreeLinear.init(In, Out, key=key, use_bias=use_bias) if free else hnn.Linear.init(In=In, Out=Out, key=key, use_bias=use_bias)


class ObjectiveQwen3LMHeadModel(Qwen3LMHeadModel):
    backward: Optional[Qwen3LMHeadModel]  # twin: same shape, reads the sequence reversed
    twin_proj: Optional[eqx.Module]  # Embed -> Embed affine map from forward states to backward states
    sr_head: Optional[eqx.Module]  # Embed -> Embed successor-representation head
    pi_proj: Optional[eqx.Module]  # Embed -> Embed map from h_t to its prediction of h_{t+k}
    eos_head: Optional[eqx.Module]  # Embed -> eos_bins logits over log2 distance-to-EOS bins

    @classmethod
    def init(cls, Vocab, config: ObjectiveQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)  # same forward initialisation as the baseline for this key
        k_b, k_t, k_s, k_p, k_e = jrandom.split(jrandom.fold_in(key, 2), 5)
        fr = config.free_heads
        backward = Qwen3LMHeadModel.init(Vocab, config, key=k_b) if config.twin else None
        twin_proj = _head(config.Embed, config.Embed.alias("twin_embed"), key=k_t, use_bias=True, free=fr) if config.twin else None
        sr_head = _head(config.Embed, config.Embed.alias("sr_embed"), key=k_s, use_bias=False, free=fr) if config.sr else None
        pi_proj = _head(config.Embed, config.Embed.alias("pi_embed"), key=k_p, use_bias=False, free=fr) if config.pi else None
        eos_head = _head(config.Embed, hax.Axis("eos_bin", config.eos_bins), key=k_e, use_bias=True, free=fr) if config.eos else None
        return cls(base.transformer, base.embeddings, base.lm_head, backward, twin_proj, sr_head, pi_proj, eos_head)

    def _ntp(self, model: Qwen3LMHeadModel, h: NamedArray, tokens: NamedArray, loss_weight: NamedArray, **kw):
        return maybe_fused_next_token_loss(self.Pos, self.Embed, self.Vocab, h, model.get_lm_head(), tokens, loss_weight=loss_weight, **kw)

    @named_call
    def _twin_loss(self, h: NamedArray, example: LmExample, seg: Optional[NamedArray], *, key):
        cfg = cast(ObjectiveQwen3Config, self.config)
        Pos = example.tokens.resolve_axis("position")
        backward = cast(Qwen3LMHeadModel, self.backward)
        rev_tokens = _flip(example.tokens, Pos)
        # loss_weight[t] says whether position t has a real next token. In the reversed frame position i predicts
        # rev[i+1] = x_{T-2-i}, so the weight it needs is the forward weight of position T-2-i = flip(lw)[i+1]; a
        # plain flip would be off by one (masking the first reversed position, un-masking the last, whose target wraps).
        rev_weight = hax.roll(_flip(example.loss_weight, Pos), -1, Pos)
        hb_rev = backward.activations(rev_tokens, _flipped_mask(example.attn_mask), key=key)
        ntp_b = self._ntp(backward, hb_rev, rev_tokens, rev_weight)
        # hb[i] has read x_{>= i} and predicts x_{i-1}; the state that predicts x_{t+1} without seeing it is hb[t+2].
        hb = _flip(hb_rev, Pos)
        off = cfg.twin_offset
        target = jax.lax.stop_gradient(hax.roll(hb, -off, Pos)).astype(jnp.float32)
        pred = cast(Any, self.twin_proj)(h).rename({"twin_embed": "embed"}).astype(jnp.float32)
        batch_axes = tuple(ax for ax in example.tokens.axes if ax.name != Pos.name)
        position = hax.arange(Pos).broadcast_axis(batch_axes)
        valid = (position + off <= Pos.size - 1).astype(jnp.float32) * (example.loss_weight > 0).astype(jnp.float32)
        if seg is not None:
            valid = valid * (seg == hax.roll(seg, -off, Pos)).astype(jnp.float32)
        sq = hax.mean((pred - target) ** 2, axis="embed")
        return ntp_b, _masked_mean(sq, valid)

    @named_call
    def _sr_loss(self, h: NamedArray, example: LmExample, seg: Optional[NamedArray]):
        cfg = cast(ObjectiveQwen3Config, self.config)
        Pos = example.tokens.resolve_axis("position")
        e = self.embeddings.embed(example.tokens).astype(jnp.float32)
        e = e * (hax.mean(e * e, axis="embed") + 1e-6) ** -0.5  # per-position RMS normalisation, no parameters
        psi = cast(Any, self.sr_head)(h).rename({"sr_embed": "embed"}).astype(jnp.float32)
        batch_axes = tuple(ax for ax in example.tokens.axes if ax.name != Pos.name)
        position = hax.arange(Pos).broadcast_axis(batch_axes)
        in_range1 = (position + 1 <= Pos.size - 1).astype(jnp.float32)
        in_range2 = (position + 2 <= Pos.size - 1).astype(jnp.float32)
        same1 = (seg == hax.roll(seg, -1, Pos)).astype(jnp.float32) if seg is not None else 1.0
        same2 = (seg == hax.roll(seg, -2, Pos)).astype(jnp.float32) if seg is not None else 1.0
        valid = in_range1 * same1 * (example.loss_weight > 0).astype(jnp.float32)
        cont = in_range2 * same2  # bootstrap only when x_{t+2} is still in the same document
        target = hax.roll(e, -1, Pos) + cfg.sr_gamma * cont * hax.roll(psi, -1, Pos)
        target = jax.lax.stop_gradient(target)
        sq = hax.mean((psi - target) ** 2, axis="embed")
        return _masked_mean(sq, valid)

    @named_call
    def _pi_loss(self, h: NamedArray, example: LmExample, seg: Optional[NamedArray], *, key):
        cfg = cast(ObjectiveQwen3Config, self.config)
        Pos = example.tokens.resolve_axis("position")
        k = cfg.pi_k
        batch_axes = tuple(ax for ax in example.tokens.axes if ax.name != Pos.name)

        def unit(x):
            return x / (hax.sqrt(hax.sum(x * x, axis="embed")) + 1e-6)

        pred = unit(cast(Any, self.pi_proj)(h).rename({"pi_embed": "embed"}).astype(jnp.float32))
        tgt = unit(h.astype(jnp.float32))
        pos = hax.roll(tgt, -k, Pos)  # the positive for anchor t is the state at t+k
        # Shared negatives: pi_negatives states drawn without replacement from every (batch, position) of this step.
        flat = hax.flatten_axes(tgt, (*batch_axes, Pos), "cand")
        n = min(cfg.pi_negatives, flat.resolve_axis("cand").size)
        idx = jrandom.choice(key, flat.resolve_axis("cand").size, (n,), replace=False)
        neg = hax.take(flat, "cand", hax.named(idx, "neg"))
        pos_logit = hax.sum(pred * pos, axis="embed") / cfg.pi_tau
        neg_logits = hax.dot(pred, neg, axis="embed") / cfg.pi_tau
        m = hax.maximum(pos_logit, hax.max(neg_logits, axis="neg"))
        lse = m + hax.log(hax.exp(pos_logit - m) + hax.sum(hax.exp(neg_logits - m), axis="neg"))
        nce = lse - pos_logit
        position = hax.arange(Pos).broadcast_axis(batch_axes)
        valid = (position + k <= Pos.size - 1).astype(jnp.float32) * (example.loss_weight > 0).astype(jnp.float32)
        if seg is not None:
            valid = valid * (seg == hax.roll(seg, -k, Pos)).astype(jnp.float32)
        return _masked_mean(nce, valid)

    @staticmethod
    def eos_targets(tokens: NamedArray, loss_weight: NamedArray, eos_id: int, nbins: int):
        """Per-position (bin, valid) for the distance-to-EOS objective; exposed for the CPU smoke's exact checks.

        ``d_t`` = index of the first EOS strictly after ``t`` minus ``t``; bin = ceil(log2 d) clipped to nbins-1.
        valid = an EOS exists later in the window AND ``t`` is not itself EOS AND the position carries loss.
        """
        Pos = tokens.resolve_axis("position")
        ax = tokens.axes.index(Pos)
        big = 2 * Pos.size
        idx = hax.arange(Pos).broadcast_axis(tuple(a for a in tokens.axes if a.name != Pos.name))
        is_eos = tokens == eos_id
        eos_idx = hax.where(is_eos, idx, big)
        at_or_after = hax.named(jax.lax.cummin(eos_idx.array, axis=ax, reverse=True), eos_idx.axes)
        strictly_after = hax.where(idx == Pos.size - 1, big, hax.roll(at_or_after, -1, Pos))
        d = hax.maximum(strictly_after - idx, 1)
        bins = hax.clip(hax.ceil(hax.log2(d.astype(jnp.float32))).astype(jnp.int32), 0, nbins - 1)
        valid = (strictly_after < big) & (~is_eos) & (loss_weight > 0)
        return bins, valid.astype(jnp.float32)

    @named_call
    def _eos_loss(self, h: NamedArray, example: LmExample):
        cfg = cast(ObjectiveQwen3Config, self.config)
        bins, valid = self.eos_targets(example.tokens, example.loss_weight, cfg.eos_id, cfg.eos_bins)
        logits = cast(Any, self.eos_head)(h).astype(jnp.float32)
        EosBin = logits.resolve_axis("eos_bin")
        onehot = (bins.broadcast_axis(EosBin) == hax.arange(EosBin)).astype(jnp.float32)
        picked = hax.sum(logits * onehot, axis="eos_bin")
        ce = hnn.logsumexp(logits, axis="eos_bin") - picked
        return _masked_mean(ce, valid)

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
        cfg = cast(ObjectiveQwen3Config, self.config)
        train = key is not None  # the trainer passes a key; evaluation does not
        k_f, k_b, k_p = maybe_rng_split(key, 3) if key is not None else (None, None, None)
        kw = dict(reduction=reduction, reduction_axis=reduction_axis, logsumexp_weight=logsumexp_weight, dtype=loss_dtype, logit_soft_cap=logit_soft_cap)
        h = self.activations(example.tokens, example.attn_mask, key=k_f)
        loss = self._ntp(self, h, example.tokens, example.loss_weight, **kw)
        if not train:
            return loss
        # The trainer differentiates through this function, so no host logging here (io callbacks reject JVP).
        seg = _segment_ids(example)
        if cfg.twin:
            ntp_b, twin = self._twin_loss(h, example, seg, key=k_b)
            loss = loss + ntp_b + cfg.twin_weight * twin
        if cfg.sr:
            loss = loss + cfg.sr_weight * self._sr_loss(h, example, seg)
        if cfg.pi:
            loss = loss + cfg.pi_weight * self._pi_loss(h, example, seg, key=k_p)
        if cfg.eos:
            loss = loss + cfg.eos_weight * self._eos_loss(h, example)
        return loss
