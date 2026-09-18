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

Loss = forward NTP [+ backward NTP + twin_weight * twin] [+ sr_weight * sr] [+ pi_weight * pi]. The logged train
loss is that sum; eval reports the forward NTP alone.
"""

from dataclasses import dataclass
from typing import Optional, cast

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

    def __post_init__(self):
        if self.twin_offset < 1:
            raise ValueError("twin_offset must be >= 1 (1 lets the backward state see the target token)")
        if not (0.0 <= self.sr_gamma < 1.0):
            raise ValueError("sr_gamma must be in [0, 1)")
        if self.pi_k < 1 or self.pi_tau <= 0 or self.pi_negatives < 1:
            raise ValueError("pi_k >= 1, pi_tau > 0 and pi_negatives >= 1 are required")

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


class ObjectiveQwen3LMHeadModel(Qwen3LMHeadModel):
    backward: Optional[Qwen3LMHeadModel]  # twin: same shape, reads the sequence reversed
    twin_proj: Optional[hnn.Linear]  # Embed -> Embed affine map from forward states to backward states
    sr_head: Optional[hnn.Linear]  # Embed -> Embed successor-representation head
    pi_proj: Optional[hnn.Linear]  # Embed -> Embed map from h_t to its prediction of h_{t+k}

    @classmethod
    def init(cls, Vocab, config: ObjectiveQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)  # same forward initialisation as the baseline for this key
        k_b, k_t, k_s, k_p = jrandom.split(jrandom.fold_in(key, 2), 4)
        backward = Qwen3LMHeadModel.init(Vocab, config, key=k_b) if config.twin else None
        twin_proj = hnn.Linear.init(In=config.Embed, Out=config.Embed.alias("twin_embed"), key=k_t, use_bias=True) if config.twin else None
        sr_head = hnn.Linear.init(In=config.Embed, Out=config.Embed.alias("sr_embed"), key=k_s, use_bias=False) if config.sr else None
        pi_proj = hnn.Linear.init(In=config.Embed, Out=config.Embed.alias("pi_embed"), key=k_p, use_bias=False) if config.pi else None
        return cls(base.transformer, base.embeddings, base.lm_head, backward, twin_proj, sr_head, pi_proj)

    def _ntp(self, model: Qwen3LMHeadModel, h: NamedArray, tokens: NamedArray, loss_weight: NamedArray, **kw):
        return maybe_fused_next_token_loss(self.Pos, self.Embed, self.Vocab, h, model.get_lm_head(), tokens, loss_weight=loss_weight, **kw)

    @named_call
    def _twin_loss(self, h: NamedArray, example: LmExample, seg: Optional[NamedArray], *, key):
        cfg = cast(ObjectiveQwen3Config, self.config)
        Pos = example.tokens.resolve_axis("position")
        backward = cast(Qwen3LMHeadModel, self.backward)
        rev_tokens = _flip(example.tokens, Pos)
        rev_weight = _flip(example.loss_weight, Pos)
        hb_rev = backward.activations(rev_tokens, _flipped_mask(example.attn_mask), key=key)
        ntp_b = self._ntp(backward, hb_rev, rev_tokens, rev_weight)
        # hb[i] has read x_{>= i} and predicts x_{i-1}; the state that predicts x_{t+1} without seeing it is hb[t+2].
        hb = _flip(hb_rev, Pos)
        off = cfg.twin_offset
        target = jax.lax.stop_gradient(hax.roll(hb, -off, Pos)).astype(jnp.float32)
        pred = cast(hnn.Linear, self.twin_proj)(h).rename({"twin_embed": "embed"}).astype(jnp.float32)
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
        psi = cast(hnn.Linear, self.sr_head)(h).rename({"sr_embed": "embed"}).astype(jnp.float32)
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

        pred = unit(cast(hnn.Linear, self.pi_proj)(h).rename({"pi_embed": "embed"}).astype(jnp.float32))
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
        return loss
