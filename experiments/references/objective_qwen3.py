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

* ``swap`` (real text, wrong context; doc section 5.1, 2026-09-22): ``swap_spans`` contiguous spans per sequence, each of
  length ~U(swap_min, swap_max), are replaced in place by the SAME positions of the previous sequence in the batch (a
  batch roll), truncated at the current document's end; position 0 and EOS are never touched. A second trunk pass reads
  the spliced copy and a scalar head must score the clean prefix above the spliced one (binary NCE, exactly as ``ebm``) at
  every loss-carrying position at or after the first splice inside the same document. This is the complement of ebm's
  negatives: ebm's own samples are detectable in structured text and invisible in fluent prose, whereas a topic or
  coherence break in real text is most detectable in prose. Encoder-side relatives: NSP / sentence-order prediction,
  ELECTRA's replaced-token detection; here it is a gated early-phase auxiliary on a causal LM.

* ``adv`` (adversarial use of the ebm discriminator; doc section 5.3, 2026-09-22; requires ``ebm``): the LM is also the
  generator. At every replaced position of the corrupted copy the ebm head's "real" log-odds on the prefix ending in the
  sampled token is a reward (stop-gradient), its mean over replaced positions the baseline, and the score function is
  log p(x_hat_t | x_<t) from the CLEAN pass -- the distribution NTP trains. Minimised: mean_{replaced} sg(r - b) * CE(x_hat)
  (REINFORCE, Williams 1992; the discriminator-as-reward split of SeqGAN, Yu et al. 2017). The head receives no
  gradient from this term (NCE only), the LM receives NTP + this. The per-position CE comes from the fused kernel with
  reduction=None, which returns the unreduced loss times the not-last mask and applies no normaliser, so the signed
  weights are applied and normalised here (a weighted MEAN in the kernel would divide by the signed weight sum).

Loss = forward NTP [+ backward NTP + twin_weight * twin] [+ sr_weight * sr] [+ pi_weight * pi] [+ eos_weight * eos]
[+ ebm/denoise/mtp terms] [+ swap_weight * swap] [+ adv_weight * adv].
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
    # ebm: energy-based NCE against the model's own one-step denoiser samples (energy-based diffusion LM training signal,
    # made causal). A fraction rho ~ U(0, ebm_rho_max) of each sequence's tokens is replaced by samples from p(x_t | x_<t)
    # (Gumbel-max over the lm_head, computed in vocab blocks); a second trunk pass reads the corrupted sequence and a scalar
    # head on unit-RMS-normalised states must give the clean prefix a higher score than the corrupted one (binary NCE).
    ebm: bool = False
    ebm_weight: float = 0.1
    ebm_rho_max: float = 0.5
    ebm_temp: float = 1.0  # sampling temperature for the corruption samples
    ebm_blocks: int = 32  # vocab blocks for the Gumbel-max sampler (memory: batch*pos*(vocab/blocks) fp32 logits)
    # ebm_steps = k > 1: the negatives are k-step autoregressive rollouts of the model's own continuation instead of
    # independent one-step draws. Positions form aligned blocks of k; a block is chosen with probability rho (the same
    # per-sequence rho ~ U(0, ebm_rho_max), so the corrupted fraction is unchanged); offset j of a chosen block is drawn
    # given the sequence with offsets < j already replaced -- one extra stop-gradient trunk pass per offset. k = 1 is
    # the original one-step code path, bit for bit.
    ebm_steps: int = 1
    # denoise: standard NTP (clean next-token targets, same lm_head) computed on the SAME corrupted copy the ebm uses
    # (rho ~ U(0, ebm_rho_max) of the tokens replaced by the model's own samples). Input-side perturbation only; the
    # target and the loss are exactly NTP (denoising autoencoder, Vincent et al. 2008; training with noise as Tikhonov
    # regularisation, Bishop 1995; scheduled sampling, Bengio et al. 2015). Works with or without the ebm head.
    denoise: bool = False
    denoise_weight: float = 0.1
    # mtp: multi-token prediction as an auxiliary (Gloeckle et al. 2024; the classic 'predict several steps ahead'
    # objective): h_t is mapped by a D x D projection and decoded by the SHARED lm_head to predict x_{t+mtp_k}
    # (no new vocab-size head). Cross-entropy with the same fused kernel as NTP; positions whose target lies in
    # another document or past the window are masked. Eval unchanged.
    mtp: bool = False
    mtp_weight: float = 0.1
    mtp_k: int = 2
    # swap: real-text-wrong-context negatives (module docstring). Spans of the previous sequence in the batch spliced in
    # place, document-truncated; scalar head, binary NCE on a second trunk pass, document-scoped informative mask like ebm.
    swap: bool = False
    swap_weight: float = 0.1
    swap_spans: int = 1
    swap_min: int = 16
    swap_max: int = 128
    # adv: REINFORCE on the sampler with the ebm head's score as reward (module docstring). Needs ebm=True. No new parameters.
    adv: bool = False
    adv_weight: float = 0.03
    free_heads: bool = True  # auxiliary heads as plain arrays in MuonH's adam group (norm free); False = hnn.Linear (norm pinned)
    # Which state the auxiliary heads read. -1 = the final normed h that the lm_head reads (rung 1-5 behaviour). k in
    # 1..num_layers = the residual stream after layer k, RMS-normalised without parameters, so the auxiliary gradient
    # shapes only layers <= k and the top layers, final norm and lm_head operating point stay NTP's alone (Caruana 1993/97
    # multitask learning: share the hidden layers, keep the outputs task-specific). NTP always uses the final h.
    aux_layer: int = -1
    # Step-free annealing of the auxiliary weight, gated on the NTP loss LEVEL of the current batch (stop-gradient):
    # gate = clip((L_ntp - aux_gate_lo) / (aux_gate_hi - aux_gate_lo), 0, 1). aux_gate_hi = 0 disables the gate. With
    # hi/lo read off the baseline's own train-loss curve (130m: 4.0 ~ step 500, 3.6 ~ step 2400) the auxiliary is fully
    # on early, ramps down as NTP improves and is off for the second half -- 'early shaping without the late NTP tax',
    # implemented inside the objective because loss_function receives no step. twin's backward NTP is not gated.
    aux_gate_hi: float = 0.0
    aux_gate_lo: float = 0.0

    def __post_init__(self):
        if self.twin_offset < 1:
            raise ValueError("twin_offset must be >= 1 (1 lets the backward state see the target token)")
        if not (0.0 <= self.sr_gamma < 1.0):
            raise ValueError("sr_gamma must be in [0, 1)")
        if self.pi_k < 1 or self.pi_tau <= 0 or self.pi_negatives < 1:
            raise ValueError("pi_k >= 1, pi_tau > 0 and pi_negatives >= 1 are required")
        if self.eos_bins < 2:
            raise ValueError("eos_bins must be >= 2")
        if not (0.0 < self.ebm_rho_max <= 1.0) or self.ebm_temp <= 0 or self.ebm_blocks < 1:
            raise ValueError("ebm_rho_max in (0, 1], ebm_temp > 0 and ebm_blocks >= 1 are required")
        if self.ebm_steps < 1:
            raise ValueError("ebm_steps must be >= 1")
        if self.aux_layer == 0 or self.aux_layer < -1 or self.aux_layer > self.num_layers:
            raise ValueError(f"aux_layer must be -1 (final h) or in 1..num_layers={self.num_layers}, got {self.aux_layer}")
        if self.mtp_k < 2:
            raise ValueError("mtp_k must be >= 2 (k=1 is NTP itself)")
        if self.swap_spans < 1 or self.swap_min < 1 or self.swap_max < self.swap_min:
            raise ValueError("swap needs swap_spans >= 1 and 1 <= swap_min <= swap_max")
        if self.adv and not self.ebm:
            raise ValueError("adv rewards the sampler with the ebm head's score, so it needs ebm=True")
        if self.adv and self.ebm_steps > 1:
            raise ValueError("adv's score function is the clean pass's one-step log-prob, so it needs ebm_steps == 1")
        if self.aux_gate_hi > 0 and not (0.0 <= self.aux_gate_lo < self.aux_gate_hi):
            raise ValueError(f"aux_gate needs 0 <= lo < hi, got lo={self.aux_gate_lo} hi={self.aux_gate_hi}")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return ObjectiveQwen3LMHeadModel

    def flops_per_token(self, vocab_size: int, context_length: int):
        # twin: a full second stack plus LM head. ebm: a second trunk pass over the corrupted copy plus one lm_head-sized
        # matmul for the sampler; counted as 2x as well. The small heads are not counted. ebm_steps = k adds k - 1
        # forward-only rollout passes, ~1/3 of a forward+backward each (MFU reporting only).
        mult = 2 if (self.twin or self.ebm or self.denoise or self.swap) else 1
        if (self.ebm or self.denoise) and self.ebm_steps > 1:
            mult += (self.ebm_steps - 1) / 3
        return mult * super().flops_per_token(vocab_size, context_length)


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


def _softplus(x: NamedArray) -> NamedArray:
    return hax.maximum(x, 0.0) + hax.log1p(hax.exp(-hax.abs(x)))


def _doc_scoped_informative(replaced: NamedArray, is_start: NamedArray, loss_weight: NamedArray, Pos) -> NamedArray:
    """Loss-carrying positions with >= 1 replacement at or before them inside the same document (shared by ebm and swap)."""
    m = replaced.astype(jnp.int32)
    c = hax.cumsum(m, axis=Pos)
    ax = c.axes.index(Pos)
    base = hax.named(jax.lax.cummax(hax.where(is_start, c - m, 0).array, axis=ax), c.axes)  # count before this document
    return ((c - base >= 1) & (loss_weight > 0)).astype(jnp.float32)


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
    ebm_head: Optional[eqx.Module]  # Embed -> 1 "clean" log-odds (negative energy) of the prefix
    mtp_proj: Optional[eqx.Module]  # Embed -> Embed map whose output the shared lm_head decodes as x_{t+mtp_k}
    swap_head: Optional[eqx.Module]  # Embed -> 1 "clean" log-odds of the prefix against a real-text splice

    @classmethod
    def init(cls, Vocab, config: ObjectiveQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)  # same forward initialisation as the baseline for this key
        k_b, k_t, k_s, k_p, k_e, k_n, k_m = jrandom.split(jrandom.fold_in(key, 2), 7)
        fr = config.free_heads
        backward = Qwen3LMHeadModel.init(Vocab, config, key=k_b) if config.twin else None
        twin_proj = _head(config.Embed, config.Embed.alias("twin_embed"), key=k_t, use_bias=True, free=fr) if config.twin else None
        sr_head = _head(config.Embed, config.Embed.alias("sr_embed"), key=k_s, use_bias=False, free=fr) if config.sr else None
        pi_proj = _head(config.Embed, config.Embed.alias("pi_embed"), key=k_p, use_bias=False, free=fr) if config.pi else None
        eos_head = _head(config.Embed, hax.Axis("eos_bin", config.eos_bins), key=k_e, use_bias=True, free=fr) if config.eos else None
        ebm_head = _head(config.Embed, hax.Axis("ebm_out", 1), key=k_n, use_bias=True, free=fr) if config.ebm else None
        mtp_proj = _head(config.Embed, config.Embed.alias("mtp_embed"), key=k_m, use_bias=True, free=fr) if config.mtp else None
        # Own key stream: the 7-way split above must stay byte-identical for every existing variant's head initialisation.
        swap_head = _head(config.Embed, hax.Axis("swap_out", 1), key=jrandom.fold_in(key, 3), use_bias=True, free=fr) if config.swap else None
        return cls(base.transformer, base.embeddings, base.lm_head, backward, twin_proj, sr_head, pi_proj, eos_head, ebm_head, mtp_proj, swap_head)

    def _ntp(self, model: Qwen3LMHeadModel, h: NamedArray, tokens: NamedArray, loss_weight: NamedArray, **kw):
        return maybe_fused_next_token_loss(self.Pos, self.Embed, self.Vocab, h, model.get_lm_head(), tokens, loss_weight=loss_weight, **kw)

    def forward_with_aux(self, input_ids: NamedArray, attn_mask, *, key=None):
        """(h, h_aux): the final normed states the lm_head reads, and the state the auxiliary heads read.

        aux_layer = -1: h_aux is h (one forward, identical to ``activations``). aux_layer = k: one scan over the layers
        that also returns every layer's output; h_aux = residual stream after layer k, RMS-normalised per position with
        no parameters. Same attention mask, same (absent) dropout keys as the transformer's own call.
        """
        cfg = cast(ObjectiveQwen3Config, self.config)
        if cfg.aux_layer < 0:
            h = self.activations(input_ids, attn_mask, key=key)
            return h, h
        tr = self.transformer
        x = self.embeddings.embed(input_ids)
        keys = maybe_rng_split(key, cfg.num_layers) if key is not None else None

        def step(layer, carry, **kw):
            y = layer(carry, **kw)
            return y, y

        x, outs = tr.layers.scan_via(step)(x, mask=attn_mask, key=keys, pos_ids=None)
        h = tr.norm(x)
        xk = outs[tr.layers.Block.name, cfg.aux_layer - 1].astype(jnp.float32)
        h_aux = xk * (hax.mean(xk * xk, axis="embed") + 1e-6) ** -0.5
        return h, h_aux

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

    def _sample_next(self, h: NamedArray, *, key) -> NamedArray:
        """One Gumbel-max draw per position from softmax(lm_head(h_t) / temp): the model's own sample of x_{t+1}.

        The vocab is swept in ``ebm_blocks`` blocks with a running (max, argmax), so the full logits are never
        materialised; the result is an exact categorical sample (argmax of logit + Gumbel over all blocks).
        """
        cfg = cast(ObjectiveQwen3Config, self.config)
        W = self.get_lm_head().rearrange((self.Embed, self.Vocab)).array  # (D, V)
        D, V = W.shape
        nb = min(cfg.ebm_blocks, V)
        blk = -(-V // nb)
        pad = nb * blk - V
        if pad:
            W = jnp.pad(W, ((0, 0), (0, pad)))
        Wb = jnp.transpose(W.reshape(D, nb, blk), (1, 0, 2))  # (nb, D, blk)
        axes = tuple(ax for ax in h.axes if ax.name != self.Embed.name)
        hf = jax.lax.stop_gradient(h.rearrange((*axes, self.Embed)).array.reshape(-1, D))
        col = jnp.arange(blk)

        def body(carry, xs):
            best, arg = carry
            i, wb, kb = xs
            logits = jnp.dot(hf, wb, preferred_element_type=jnp.float32) / cfg.ebm_temp
            val = logits + jrandom.gumbel(kb, logits.shape, dtype=jnp.float32)
            val = jnp.where((i * blk + col)[None, :] < V, val, -jnp.inf)
            m = jnp.max(val, axis=1)
            a = jnp.argmax(val, axis=1).astype(jnp.int32) + i * blk
            upd = m > best
            return (jnp.where(upd, m, best), jnp.where(upd, a, arg)), None

        n = hf.shape[0]
        init = (jnp.full((n,), -jnp.inf, jnp.float32), jnp.zeros((n,), jnp.int32))
        (_, idx), _ = jax.lax.scan(body, init, (jnp.arange(nb, dtype=jnp.int32), Wb, jrandom.split(key, nb)))
        return hax.named(idx.reshape([ax.size for ax in axes]), axes)

    def _ebm_corrupt(self, h: NamedArray, example: LmExample, seg: Optional[NamedArray], *, key):
        """Diffusion-style corruption with the model as the noise kernel. Returns (x_noisy, replaced, informative).

        rho ~ U(0, ebm_rho_max) per sequence; each position is replaced with probability rho by a draw from p(x_t | x_<t)
        taken from the clean pass. Never touched: every document-start position (position 0, and any t with
        seg[t] != seg[t-1], whose sample would otherwise come from h_{t-1} in the PREVIOUS document), EOS tokens,
        draws that are EOS, and draws equal to the true token (no corruption happened). ``informative`` marks
        loss-carrying positions with >= 1 replacement at or before them inside the same document -- elsewhere the
        clean and corrupted prefixes coincide and the pair is a coin flip. ``ebm_steps`` > 1 hands the same draws to
        ``_ebm_rollout`` (k-step autoregressive negatives); ``ebm_steps`` = 1 never leaves this function.
        """
        cfg = cast(ObjectiveQwen3Config, self.config)
        tokens = example.tokens
        Pos = tokens.resolve_axis("position")
        batch_axes = tuple(ax for ax in tokens.axes if ax.name != Pos.name)
        k_s, k_r, k_u = jrandom.split(key, 3)
        samp = hax.roll(self._sample_next(h, key=k_s), 1, Pos)  # samp[t] ~ p(x_t | x_<t), drawn from h_{t-1}
        position = hax.arange(Pos).broadcast_axis(batch_axes)
        is_start = (position == 0) | (seg != hax.roll(seg, 1, Pos)) if seg is not None else position == 0
        rho = hax.random.uniform(k_r, batch_axes, maxval=cfg.ebm_rho_max) if batch_axes else cfg.ebm_rho_max
        u = hax.random.uniform(k_u, tokens.axes)
        if cfg.ebm_steps > 1:
            return self._ebm_rollout(example, samp, position, is_start, rho, u, key=k_s)
        protect = is_start | (tokens == cfg.eos_id) | (samp == cfg.eos_id) | (samp == tokens)
        replaced = (u < rho) & ~protect
        x_noisy = hax.where(replaced, samp, tokens)
        return x_noisy, replaced, _doc_scoped_informative(replaced, is_start, example.loss_weight, Pos)

    def _ebm_rollout(self, example: LmExample, samp0: NamedArray, position: NamedArray, is_start: NamedArray, rho, u: NamedArray, *, key):
        """k-step autoregressive own-sample corruption (``ebm_steps`` = k > 1). Returns (x_noisy, replaced, informative).

        Positions form aligned blocks of k; a block is chosen iff the uniform at its first position is < rho, so the
        expected corrupted fraction matches the one-step path. Offset 0 of a chosen block takes the clean pass's draw
        (``samp0``, the one-step sample); offset j >= 1 is drawn from p(x_t | current prefix), computed by one
        stop-gradient trunk pass over the sequence with offsets < j already replaced in every chosen block. Inside a
        block the negative is therefore a genuine k-step rollout of the model's own continuation, and the incoherent
        bigram that two independent one-step draws form when adjacent positions are both replaced cannot occur. Earlier
        blocks are seen in their partially replaced state (offsets >= j still clean) -- the same kind of approximation
        the one-step path makes by conditioning every draw on the clean prefix. Never touched, as in the one-step path:
        document starts, EOS tokens, and draws that are EOS (the true token is kept, so segment ids stay valid).
        """
        cfg = cast(ObjectiveQwen3Config, self.config)
        tokens = example.tokens
        Pos = tokens.resolve_axis("position")
        K = cfg.ebm_steps
        if Pos.size % K:
            raise ValueError(f"ebm_steps={K} must divide the sequence length {Pos.size}")
        pax = u.axes.index(Pos)
        u_blk = jnp.take(u.array, jnp.arange(0, Pos.size, K), axis=pax)  # the uniform at each block's first position
        chosen = hax.named(jnp.repeat(u_blk, K, axis=pax), u.axes) < rho
        offset = hax.named(position.array % K, position.axes)
        never = is_start | (tokens == cfg.eos_id)
        x = tokens
        for j in range(K):
            if j == 0:
                s_j = samp0
            else:
                h_j = jax.lax.stop_gradient(self.activations(x, example.attn_mask, key=None))
                s_j = hax.roll(self._sample_next(h_j, key=jrandom.fold_in(key, j)), 1, Pos)
            sel = chosen & (offset == j) & ~never & (s_j != cfg.eos_id)
            x = hax.where(sel, s_j, x)
        replaced = x != tokens
        return x, replaced, _doc_scoped_informative(replaced, is_start, example.loss_weight, Pos)

    def _swap_corrupt(self, example: LmExample, seg: Optional[NamedArray], *, key):
        """Real-text splice corruption. Returns (x_noisy, replaced, informative).

        For each of ``swap_spans`` spans: length L ~ U(swap_min, swap_max), start t0 ~ U(1, T - L); positions in
        [t0, t0 + L) whose document is the one at t0 are replaced by the same positions of the PREVIOUS sequence in the
        batch (real text from another document). Position 0, EOS tokens and positions where the source token equals the
        true token are left alone, so ``replaced`` marks exactly the changed tokens. No model forward is needed.
        """
        cfg = cast(ObjectiveQwen3Config, self.config)
        tokens = example.tokens
        Pos = tokens.resolve_axis("position")
        batch_axes = tuple(ax for ax in tokens.axes if ax.name != Pos.name)
        if not batch_axes:
            raise NotImplementedError("swap needs a batch axis to draw the splice from another sequence")
        position = hax.arange(Pos).broadcast_axis(batch_axes)
        is_start = (position == 0) | (seg != hax.roll(seg, 1, Pos)) if seg is not None else position == 0
        src = hax.roll(tokens, 1, batch_axes[0])  # the previous sequence in the batch
        bshape = tuple(ax.size for ax in batch_axes)
        in_span = None
        for k in jrandom.split(key, cfg.swap_spans):
            k_l, k_t = jrandom.split(k)
            L = jrandom.randint(k_l, bshape, cfg.swap_min, cfg.swap_max + 1)
            t0 = jrandom.randint(k_t, bshape, 1, jnp.maximum(Pos.size - L, 2))  # start in [1, T - L); span may be truncated at T
            L_n, t0_n = hax.named(L, batch_axes), hax.named(t0, batch_axes)
            span = (position >= t0_n) & (position < t0_n + L_n)
            if seg is not None:
                span = span & (seg == seg.take(Pos, t0_n))  # never cross the document that starts the span
            in_span = span if in_span is None else (in_span | span)
        protect = (position == 0) | (tokens == cfg.eos_id) | (src == tokens)
        replaced = in_span & ~protect
        x_noisy = hax.where(replaced, src, tokens)
        return x_noisy, replaced, _doc_scoped_informative(replaced, is_start, example.loss_weight, Pos)

    def _corrupted_pass(self, h: NamedArray, example: LmExample, seg: Optional[NamedArray], *, key):
        """One corrupted copy (samples from the lm_head on the FINAL h) and one trunk pass over it.

        Returns (informative, h_noisy_final, h_noisy_aux, x_noisy, replaced): the NCE mask, the final normed states (for
        denoising NTP), the auxiliary-readout states (for the energy head), the corrupted tokens and the replacement mask
        (for adv). Shared by ebm, denoise and adv so all cost one extra pass.
        """
        x_noisy, replaced, valid = self._ebm_corrupt(h, example, seg, key=key)
        h_noisy_final, h_noisy_aux = self.forward_with_aux(x_noisy, example.attn_mask, key=None)
        return valid, h_noisy_final, h_noisy_aux, x_noisy, replaced

    @named_call
    def _ebm_loss(self, h_aux: NamedArray, h_noisy_aux: NamedArray, valid: NamedArray):
        return self._nce_loss(cast(Any, self.ebm_head), h_aux, h_noisy_aux, valid)

    def _nce_scores(self, head, h_aux: NamedArray, h_noisy_aux: NamedArray):
        # The scalar head reads h_aux (final h, or the aux_layer state) for both the clean and the corrupted pass.
        def unit(x):
            x = x.astype(jnp.float32)
            return x * (hax.mean(x * x, axis="embed") + 1e-6) ** -0.5  # parameter-free: isolates the head from the final-norm gain

        out = head.Out.name
        s_clean = head(unit(h_aux))[out, 0].astype(jnp.float32)  # log-odds "this prefix is real" = negative energy
        s_noisy = head(unit(h_noisy_aux))[out, 0].astype(jnp.float32)
        return s_clean, s_noisy

    def _nce_loss(self, head, h_aux: NamedArray, h_noisy_aux: NamedArray, valid: NamedArray):
        s_clean, s_noisy = self._nce_scores(head, h_aux, h_noisy_aux)
        return _masked_mean(_softplus(-s_clean) + _softplus(s_noisy), valid)  # NCE: real -> 1, own samples -> 0 (Gutmann & Hyvarinen 2010; kexue.fm/5617)

    def _adv_loss(self, h: NamedArray, x_noisy: NamedArray, replaced: NamedArray, s_noisy: NamedArray, example: LmExample, **kw):
        """REINFORCE on the sampler: mean over replaced positions of sg(r - b) * CE(x_hat), r = the ebm head's log-odds
        on the corrupted prefix ending in x_hat (stop-gradient), b = its mean over replaced positions, CE from the CLEAN
        pass. ce[t] scores the target at t+1, so the reward and the mask are rolled by -1 to line up."""
        Pos = example.tokens.resolve_axis("position")
        ce = self._ntp(self, h, x_noisy, None, **dict(kw, reduction=None, reduction_axis=None))  # unreduced, not-last masked
        w = hax.roll(replaced, -1, Pos).astype(jnp.float32) * (example.loss_weight > 0).astype(jnp.float32)
        r = jax.lax.stop_gradient(hax.roll(s_noisy, -1, Pos).astype(jnp.float32))
        n = hax.maximum(hax.sum(w), 1.0)
        b = hax.sum(r * w) / n
        return hax.sum((r - b) * w * ce.astype(jnp.float32)) / n

    @named_call
    def _mtp_loss(self, h_aux: NamedArray, example: LmExample, seg: Optional[NamedArray], **kw):
        """CE of lm_head(mtp_proj(h_t)) against x_{t+k}. The fused NTP kernel shifts targets by one internally, so the
        tokens are pre-rolled by k-1; the weight is the baseline's loss weight rolled the same way, with the wrapped
        tail and cross-document pairs masked."""
        cfg = cast(ObjectiveQwen3Config, self.config)
        Pos = example.tokens.resolve_axis("position")
        k = cfg.mtp_k
        batch_axes = tuple(ax for ax in example.tokens.axes if ax.name != Pos.name)
        position = hax.arange(Pos).broadcast_axis(batch_axes)
        tokens_k = hax.roll(example.tokens, -(k - 1), Pos)  # tokens_k[t+1] = x_{t+k}
        weight = hax.roll(example.loss_weight, -(k - 1), Pos) * (position + k <= Pos.size - 1).astype(jnp.float32)
        if seg is not None:
            weight = weight * (seg == hax.roll(seg, -k, Pos)).astype(jnp.float32)
        pred = cast(Any, self.mtp_proj)(h_aux).rename({"mtp_embed": "embed"}).astype(h_aux.dtype)
        return self._ntp(self, pred, tokens_k, weight, **kw)

    @named_call
    def _denoise_loss(self, h_noisy_final: NamedArray, example: LmExample, **kw):
        # Plain NTP on the corrupted prefix with the CLEAN next tokens as targets and the baseline's loss weights.
        return self._ntp(self, h_noisy_final, example.tokens, example.loss_weight, **kw)

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
        k_f, k_b, k_p, k_e = maybe_rng_split(key, 4) if key is not None else (None, None, None, None)
        k_w = jrandom.fold_in(key, 7) if key is not None else None  # swap's own stream; the 4-way split above is unchanged
        kw = dict(reduction=reduction, reduction_axis=reduction_axis, logsumexp_weight=logsumexp_weight, dtype=loss_dtype, logit_soft_cap=logit_soft_cap)
        if not train:
            h = self.activations(example.tokens, example.attn_mask, key=None)  # plain forward, identical to the baseline
            return self._ntp(self, h, example.tokens, example.loss_weight, **kw)
        h, h_aux = self.forward_with_aux(example.tokens, example.attn_mask, key=k_f)
        loss = self._ntp(self, h, example.tokens, example.loss_weight, **kw)
        # The trainer differentiates through this function, so no host logging here (io callbacks reject JVP).
        seg = _segment_ids(example)
        gate: Any = 1.0
        if cfg.aux_gate_hi > 0:
            lvl = jax.lax.stop_gradient(loss)
            lvl = hax.mean(lvl) if isinstance(lvl, NamedArray) else jnp.mean(lvl)
            gate = hax.clip((lvl.astype(jnp.float32) - cfg.aux_gate_lo) / (cfg.aux_gate_hi - cfg.aux_gate_lo), 0.0, 1.0)
        if cfg.twin:
            ntp_b, twin = self._twin_loss(h_aux, example, seg, key=k_b)
            loss = loss + ntp_b + gate * cfg.twin_weight * twin
        if cfg.sr:
            loss = loss + gate * cfg.sr_weight * self._sr_loss(h_aux, example, seg)
        if cfg.pi:
            loss = loss + gate * cfg.pi_weight * self._pi_loss(h_aux, example, seg, key=k_p)
        if cfg.eos:
            loss = loss + gate * cfg.eos_weight * self._eos_loss(h_aux, example)
        if cfg.mtp:
            loss = loss + gate * cfg.mtp_weight * self._mtp_loss(h_aux, example, seg, **kw)
        if cfg.ebm or cfg.denoise:
            valid, h_noisy_final, h_noisy_aux, x_noisy, replaced = self._corrupted_pass(h, example, seg, key=k_e)
            if cfg.ebm:
                s_clean, s_noisy = self._nce_scores(cast(Any, self.ebm_head), h_aux, h_noisy_aux)
                loss = loss + gate * cfg.ebm_weight * _masked_mean(_softplus(-s_clean) + _softplus(s_noisy), valid)
                if cfg.adv:
                    loss = loss + gate * cfg.adv_weight * self._adv_loss(h, x_noisy, replaced, s_noisy, example, **kw)
            if cfg.denoise:
                loss = loss + gate * cfg.denoise_weight * self._denoise_loss(h_noisy_final, example, **kw)
        if cfg.swap:
            x_sw, _, valid_sw = self._swap_corrupt(example, seg, key=k_w)
            _, h_sw_aux = self.forward_with_aux(x_sw, example.attn_mask, key=None)
            loss = loss + gate * cfg.swap_weight * self._nce_loss(cast(Any, self.swap_head), h_aux, h_sw_aux, valid_sw)
        return loss
