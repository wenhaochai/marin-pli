# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-layer LM supervision (pls) on the muonh_qwen3 baseline: every layer's residual stream is decoded by the model's
own final RMSNorm and lm_head and trained on the next-token cross-entropy.

LLAL (Jin et al. 2026, "Mitigate Silent Expert Death in Ultra-Sparse MoE") adds lambda * CE(W_lm h^(l), y) at the first
MoE layer of an MoE model for a short early window. This is the dense, every-layer, always-on version. With L layers

    loss = CE(h_{L-1}) + pls_weight * sum_{k=0}^{L-2} CE(h_k),    CE(h) = CE(lm_head(final_norm(h)), next token),

where h_k is the residual stream after layer k and the k = L-1 term is the ordinary NTP loss, so pls_weight = 1 gives
every layer's LM loss the final loss's weight. There are no new parameters: the readouts share the final norm and the
lm_head, and their gradients flow into both (``pls_detach_head=True``: the intermediate losses train the trunk only, and
the final norm and lm_head get the final layer's gradient alone). Data, batch, schedule and optimizer stay the baseline's,
and evaluation (``key=None``) runs the baseline's loss unchanged, so eval/paloma numbers are directly comparable.

Monitoring. Training returns every layer's CE as ``train/pls/L{k}`` (k = L-1 is the NTP loss, which ``train/loss`` no
longer equals once pls_weight > 0). With pls_weight = 0 the model trains exactly as the baseline and the per-layer CEs are
stop-gradient readouts on every ``pls_monitor_stride``-th position: a logit-lens reference run. ``extra_eval_callbacks``
adds an evaluator that decodes every layer in one forward pass at the main eval's cadence and logs ``eval/L{k}/...``
with the main eval's keys (loss, macro_loss, per-dataset loss and bpb); ``eval/L{L-1}/...`` equals the main eval.

Schedule (``pls_off_start``/``pls_off_end``): the intermediate weight holds at pls_weight, falls linearly to 0 between the
two steps and stays 0 after, when the readouts are skipped (a ``lax.cond`` on the traced train step) and train/pls/L{k}
falls back to the stride-sampled monitor. Probe (``pls_probe``, diagnostic runs): every train step, the cosine between
each intermediate loss's gradient and the final loss's gradient on the trunk they share, their norm ratio, and every
readout's in-context score (late-position minus early-position CE), all stop-gradient: the candidate signals for a
metric-driven schedule.
"""

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from tqdm_loggable.auto import tqdm

import haliax as hax
import haliax.nn as hnn
from haliax import NamedArray
from haliax.jax_utils import maybe_rng_split

import levanter.tracker
from levanter.callbacks import StepInfo
from levanter.eval import EvalResult, TaggedEvaluator, _ensure_named_lm_example, _EvalRunningMeans, construct_log_dict
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty, maybe_fused_next_token_loss, next_token_loss_weight
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.trainer import current_train_step
from levanter.utils.logging import LoadingTimeTrackerIterator
from levanter.utils.tree_utils import inference_mode

logger = logging.getLogger(__name__)


@LmConfig.register_subclass("qwen3_pls")
@dataclass(frozen=True)
class PerLayerQwen3Config(Qwen3Config):
    # Weight of every intermediate layer's LM loss; the final layer's (the NTP loss) is 1. 0 = baseline training with
    # stop-gradient per-layer readouts for monitoring.
    pls_weight: float = 1.0
    # pls_weight = 0 only: the monitoring readouts use every k-th position (forward only, ~1/(3k) of a readout's cost).
    pls_monitor_stride: int = 16
    # Per-layer evaluation (eval/L{k}/...) at the main eval's cadence.
    pls_eval: bool = True
    # Intermediate layers' LM losses train the trunk only: their gradients do not reach the shared final norm or the
    # lm_head, which are then trained by the final layer's NTP loss alone. The forward pass is unchanged.
    pls_detach_head: bool = False
    # Every intermediate layer gets its own readout: its own RMSNorm and its own lm_head (initialised like the model's),
    # trained by that layer's loss alone; the final layer keeps the model's norm and lm_head, which then get the final
    # layer's gradient alone. Per-layer eval reads each layer through its own head. The heads are ``aux_lm_heads`` (AdamH,
    # like lm_head) and ``aux_norms`` (Adam, like the final norm). New parameters: (L-1) x (vocab x hidden + hidden).
    pls_separate_heads: bool = False
    # With separate heads: the intermediate losses train their own heads only; their gradient stops at the layer's
    # output, so the backbone and the main head train exactly as the baseline (online tuned-lens probes of an
    # undisturbed model).
    pls_detach_backbone: bool = False
    # Weight schedule: pls_weight until step pls_off_start, linear to 0 at step pls_off_end (start == end: a switch), 0
    # after, with the readouts skipped. pls_off_end = 0: always on.
    pls_off_start: int = 0
    pls_off_end: int = 0
    # Diagnostic: per-layer gradient alignment and in-context scores every train step (no effect on training; one extra
    # forward and L backward passes per step). In-context score = CE on every pls_monitor_stride-th position of the
    # second half of the sequence minus CE on positions [pls_probe_early[0], pls_probe_early[1]).
    pls_probe: bool = False
    pls_probe_early: tuple[int, int] = (32, 64)
    # Residual design of every layer: "baseline" is the baseline's own Stacked transformer (with hybrid_norm, the
    # launcher's default, that is Sandwich-LN); any of depth_arch_qwen3.ARCHS (DepthBench, arXiv 2609.32534) runs
    # depth_arch_qwen3.DepthArchTransformer, a checkpointed Python loop.
    depth_arch: str = "baseline"
    hc_streams: int = 4  # hc, mhc: residual streams
    attnres_blocks: int = 8  # attnres_block: blocks
    moda_chunk: int = 1024  # moda: query positions per attention chunk

    def __post_init__(self):
        super().__post_init__()
        if self.pls_weight < 0:
            raise ValueError(f"pls_weight must be >= 0, got {self.pls_weight}")
        if not self.scan_layers and self.depth_arch == "baseline":
            raise ValueError("pls reads every layer's output through Stacked.scan_via, so scan_layers must be True")
        from experiments.references.depth_arch_qwen3 import ARCHS

        if self.depth_arch != "baseline" and self.depth_arch not in ARCHS:
            raise ValueError(f"depth_arch must be 'baseline' or one of {ARCHS}, got {self.depth_arch!r}")
        if self.pls_separate_heads and self.pls_weight <= 0:
            raise ValueError("pls_separate_heads needs pls_weight > 0: with weight 0 the per-layer heads would never train")
        if self.pls_detach_backbone and not self.pls_separate_heads:
            raise ValueError("pls_detach_backbone needs pls_separate_heads: with the shared head the loss would train the baseline's lm_head")
        if self.pls_separate_heads and self.pls_detach_head:
            raise ValueError("pls_separate_heads and pls_detach_head are exclusive: detaching separate heads leaves them untrained")
        if self.pls_monitor_stride < 1 or self.max_seq_len % self.pls_monitor_stride:
            raise ValueError(f"pls_monitor_stride must be >= 1 and divide max_seq_len={self.max_seq_len}, got {self.pls_monitor_stride}")
        if self.pls_off_end and not (0 <= self.pls_off_start <= self.pls_off_end and self.pls_weight > 0):
            raise ValueError(f"a weight schedule needs 0 <= pls_off_start <= pls_off_end and pls_weight > 0, got {self.pls_off_start}, {self.pls_off_end}, {self.pls_weight}")
        if not self.pls_off_end and self.pls_off_start:
            raise ValueError("pls_off_start without pls_off_end")
        half = self.max_seq_len // 2
        a, b = self.pls_probe_early
        if self.pls_probe and (half % self.pls_monitor_stride or not 0 <= a < b <= half):
            raise ValueError(f"pls_probe needs pls_monitor_stride | max_seq_len/2 and 0 <= early start < end <= max_seq_len/2, got stride {self.pls_monitor_stride}, early {self.pls_probe_early}")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return PerLayerQwen3LMHeadModel

    def flops_per_token(self, vocab_size: int, context_length: int):
        # Each readout is one more lm_head matmul (2 * hidden * vocab forward FLOPs per token, as the base counts the head).
        # Monitoring readouts run forward only on 1/stride of the positions, i.e. 1/(3 * stride) of a trained readout.
        # With a weight schedule the count is the monitor's (after the window, i.e. most of training). The probe is not counted.
        base = super().flops_per_token(vocab_size, context_length)
        if base is None:
            return None
        share = 1.0 if self.pls_weight > 0 and not self.pls_off_end else 1.0 / (3 * self.pls_monitor_stride)
        return base + (self.num_layers - 1) * 2 * self.hidden_dim * vocab_size * share

    def extra_eval_callbacks(self, EvalBatch, tagged_eval_sets, tokenizer, device_mesh, axis_mapping, max_examples_per_dataset, *, mp):
        """Evaluators the training loop runs next to the main eval (levanter.main.train_lm)."""
        if not self.pls_eval:
            return []
        readouts = tuple(range(self.num_layers))
        return [cb_readout_evaluate(EvalBatch, tagged_eval_sets, tokenizer, device_mesh, axis_mapping, max_examples_per_dataset, readouts=readouts, mp=mp)]


def _scalar(x):
    return x.array if isinstance(x, NamedArray) else x


def _every(x: NamedArray, axis_name: str, stride: int, start: int) -> NamedArray:
    """x at positions start, start + stride, ... of axis axis_name (stride must divide the axis size minus start)."""
    if start:
        x = x[axis_name, start:]
    ax = x.resolve_axis(axis_name)
    return x.unflatten_axis(ax, (hax.Axis("pls_chunk", ax.size // stride), hax.Axis("pls_stride", stride)))["pls_stride", 0]


class PerLayerQwen3LMHeadModel(Qwen3LMHeadModel):
    aux_norms: Optional[tuple] = None  # pls_separate_heads: one RMSNorm per intermediate layer
    aux_lm_heads: Optional[tuple] = None  # pls_separate_heads: one Embed -> Vocab head per intermediate layer

    @classmethod
    def init(cls, Vocab, config: PerLayerQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)  # the baseline's parameters and initialisation, unchanged
        aux_norms = aux_lm_heads = None
        if config.pls_separate_heads:
            # Own key stream, so the baseline parameters above stay byte-identical to Qwen3LMHeadModel.init(key).
            keys = jrandom.split(jrandom.fold_in(key, 0x9A), config.num_layers - 1)
            aux_norms = tuple(config.mk_LayerNorm(config.Embed) for _ in range(config.num_layers - 1))
            aux_lm_heads = tuple(hnn.Linear.init(In=config.Embed, Out=Vocab, key=k, use_bias=False, out_first=True) for k in keys)
        transformer = base.transformer
        if config.depth_arch != "baseline":
            from experiments.references.depth_arch_qwen3 import DepthArchTransformer

            transformer = DepthArchTransformer.init(config, config.depth_arch, key=jrandom.fold_in(key, 0xD7))
        return cls(transformer, base.embeddings, base.lm_head, aux_norms, aux_lm_heads)

    def layer_outputs(self, input_ids: NamedArray, attn_mask, *, key=None) -> NamedArray:
        """The residual stream after every layer, stacked on the layer axis; the last entry is the final pre-norm state
        (the activations the baseline normalises and decodes). Same embedding, mask and per-layer keys as ``activations``."""
        tr = self.transformer
        x = self.embeddings.embed(input_ids)
        if cast(PerLayerQwen3Config, self.config).depth_arch != "baseline":
            return cast(Any, tr).outputs(x, attn_mask, key=key)
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None

        def step(layer, carry, **kw):
            y = layer(carry, **kw)
            return y, y

        _, outs = tr.layers.scan_via(step)(x, mask=attn_mask, key=keys, pos_ids=None)
        return outs

    def _separate(self, k: int) -> bool:
        return self.aux_lm_heads is not None and k < self.config.num_layers - 1

    def _layer(self, outs: NamedArray, k: int, *, detach_head: bool = False, detach_input: bool = False) -> NamedArray:
        norm = cast(Any, self.aux_norms)[k] if self._separate(k) else self.transformer.norm
        if detach_head:  # same values; no gradient into the norm's parameters (the input still gets its gradient)
            norm = jax.tree_util.tree_map(lambda a: jax.lax.stop_gradient(a) if eqx.is_array(a) else a, norm)
        h = outs[self.transformer.layers.Block.name, k]
        if detach_input:  # no gradient from this readout into the backbone
            h = jax.lax.stop_gradient(h)
        return norm(h)

    def _head(self, k: int) -> NamedArray:
        """Layer k's readout head: its own with pls_separate_heads (k < L-1), else the model's lm_head."""
        return cast(Any, self.aux_lm_heads)[k].weight if self._separate(k) else self.get_lm_head()

    def _readout_ce(self, outs: NamedArray, k: int, example: LmExample, *, detach_head: bool = False, detach_input: bool = False, **kw) -> NamedArray:
        lm_head = self._head(k)
        if detach_head:
            lm_head = jax.lax.stop_gradient(lm_head)
        return maybe_fused_next_token_loss(
            self.Pos, self.Embed, self.Vocab, self._layer(outs, k, detach_head=detach_head, detach_input=detach_input), lm_head, example.tokens, loss_weight=example.loss_weight, **kw
        )

    def _subset_ce(self, outs: NamedArray, k: int, example: LmExample, select, *, logit_soft_cap) -> jax.Array:
        """Stop-gradient CE of layer k's readout on the positions ``select`` keeps (targets x_{t+1}, the baseline's
        next-token weights); a weighted mean over those positions, like the training loss over all of them."""
        Pos = example.tokens.resolve_axis(self.Pos.name)
        h = jax.lax.stop_gradient(self._layer(outs, k))
        lm_head = jax.lax.stop_gradient(self._head(k))
        target = hax.roll(example.tokens, -1, Pos)
        weight = next_token_loss_weight(Pos, example.loss_weight)
        ce = fused_cross_entropy_loss_and_logsumexp_penalty(
            select(h),
            lm_head,
            Contract=self.Embed,
            Label=self.Vocab,
            target_y=select(target),
            reduction=hax.mean,
            reduction_axis=None,
            weight=select(weight),
            logsumexp_weight=None,
            dtype=weight.dtype,
            logit_soft_cap=logit_soft_cap,
        )
        return _scalar(ce)

    def _monitor_ce(self, outs: NamedArray, k: int, example: LmExample, *, logit_soft_cap) -> jax.Array:
        """Stop-gradient CE of layer k's readout on every pls_monitor_stride-th position."""
        s = self.config.pls_monitor_stride
        return self._subset_ce(outs, k, example, lambda x: _every(x, self.Pos.name, s, 0), logit_soft_cap=logit_soft_cap)

    def _icl_score(self, outs: NamedArray, k: int, example: LmExample, *, logit_soft_cap) -> jax.Array:
        """In-context score of layer k's readout: CE on every stride-th position of the sequence's second half minus CE
        on the early window pls_probe_early (negative once the readout uses context)."""
        cfg = cast(PerLayerQwen3Config, self.config)
        Pos = example.tokens.resolve_axis(self.Pos.name)
        a, b = cfg.pls_probe_early
        half = Pos.size // 2
        late = self._subset_ce(outs, k, example, lambda x: _every(x, Pos.name, cfg.pls_monitor_stride, half), logit_soft_cap=logit_soft_cap)
        early = self._subset_ce(outs, k, example, lambda x: x[Pos.name, a:b], logit_soft_cap=logit_soft_cap)
        return late - early

    def _probe_stats(self, outs: NamedArray, example: LmExample, *, key, kw: dict) -> dict[str, jax.Array]:
        """Gradient alignment of every intermediate loss with the final loss, on the trunk both depend on (embeddings and
        blocks 0..k; heads and norms excluded), from one VJP of all readout CEs at the current parameters:
        cos_L{k} = cos(grad CE_k, grad CE_{L-1}), gratio_L{k} = |grad CE_k| / |grad CE_{L-1}| (on embeddings + blocks
        0..k), cos_aux = cos(grad sum_{k<L-1} CE_k, grad CE_{L-1}) on the whole trunk; plus icl_L{k} for every readout.
        Everything is stop-gradient: training is unchanged."""
        cfg = cast(PerLayerQwen3Config, self.config)
        L = cfg.num_layers
        Block = self.transformer.layers.Block.name
        pkw = dict(kw, reduction=hax.mean, reduction_axis=None)
        frozen = jax.tree_util.tree_map(lambda a: jax.lax.stop_gradient(a) if eqx.is_array(a) else a, self)
        # One trunk forward; each readout's CE differentiated only back to the layer outputs; then one trunk backward per
        # gradient (a VJP of all readouts at once would run every readout's backward for every one-hot cotangent).
        o, trunk_vjp = eqx.filter_vjp(lambda m: m.layer_outputs(example.tokens, example.attn_mask, key=key), frozen)
        cots = [jax.grad(lambda o_, k=k: jnp.asarray(_scalar(frozen._readout_ce(o_, k, example, **pkw)), jnp.float32))(o) for k in range(L)]

        def trunk_grad(cot):
            g = trunk_vjp(cot)[0]
            return g.embeddings, g.transformer.layers

        def named_leaves(t):
            return [x for x in jax.tree_util.tree_leaves(t, is_leaf=lambda x: isinstance(x, NamedArray)) if isinstance(x, NamedArray)]

        def dots(g, h):
            """(per-block dot products, shape (L,); embedding dot product) of two trunk gradients."""
            per_block = jnp.zeros((L,), jnp.float32)
            for a_, b_ in zip(named_leaves(g[1]), named_leaves(h[1])):
                assert any(ax.name == Block for ax in a_.axes), a_.axes
                per_block = per_block + hax.sum(a_ * b_, axis=tuple(ax for ax in a_.axes if ax.name != Block)).rearrange((Block,)).array.astype(jnp.float32)
            emb = sum(jnp.sum((a_ * b_).array.astype(jnp.float32)) for a_, b_ in zip(named_leaves(g[0]), named_leaves(h[0])))
            return per_block, emb

        g_final = trunk_grad(cots[L - 1])
        f_blocks, f_emb = dots(g_final, g_final)
        out = {}
        for k in range(L - 1):
            g_k = trunk_grad(cots[k])  # zero on blocks above k
            d_blocks, d_emb = dots(g_k, g_final)
            n_blocks, n_emb = dots(g_k, g_k)
            dot = jnp.sum(d_blocks[: k + 1]) + d_emb
            n_k = jnp.sum(n_blocks[: k + 1]) + n_emb
            n_f = jnp.sum(f_blocks[: k + 1]) + f_emb
            out[f"pls_probe/cos_L{k}"] = dot / jnp.sqrt(n_k * n_f)
            out[f"pls_probe/gratio_L{k}"] = jnp.sqrt(n_k / n_f)
        cot_aux = cots[0]
        for c in cots[1 : L - 1]:
            cot_aux = cot_aux + c
        g_aux = trunk_grad(cot_aux)
        a_blocks, a_emb = dots(g_aux, g_final)
        na_blocks, na_emb = dots(g_aux, g_aux)
        out["pls_probe/cos_aux"] = (jnp.sum(a_blocks) + a_emb) / jnp.sqrt((jnp.sum(na_blocks) + na_emb) * (jnp.sum(f_blocks) + f_emb))
        for k in range(L):
            out[f"pls_probe/icl_L{k}"] = self._icl_score(outs, k, example, logit_soft_cap=kw.get("logit_soft_cap"))
        return out

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
        if key is None:  # evaluation (the trainer passes a key, the evaluators do not): the baseline's loss
            return super().compute_next_token_loss(
                example,
                key=None,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )
        cfg = cast(PerLayerQwen3Config, self.config)
        L = cfg.num_layers
        kw = dict(reduction=reduction, reduction_axis=reduction_axis, logsumexp_weight=logsumexp_weight, dtype=loss_dtype, logit_soft_cap=logit_soft_cap)
        outs = self.layer_outputs(example.tokens, example.attn_mask, key=key)
        final = self._readout_ce(outs, L - 1, example, **kw)

        def supervised(w):
            loss, per = final, []
            for k in range(L - 1):
                ce = self._readout_ce(outs, k, example, detach_head=cfg.pls_detach_head, detach_input=cfg.pls_detach_backbone, **kw)
                loss = loss + ce * w  # NamedArray on the left: w may be a traced scalar
                per.append(jnp.asarray(_scalar(ce), jnp.float32))
            return loss, jnp.stack(per)

        def monitored():
            return final, jnp.stack([jnp.asarray(self._monitor_ce(outs, k, example, logit_soft_cap=logit_soft_cap), jnp.float32) for k in range(L - 1)])

        weight = None
        if cfg.pls_weight == 0:
            loss, per = monitored()
        elif not cfg.pls_off_end:
            loss, per = supervised(cfg.pls_weight)
        else:
            step = current_train_step()
            if step is None:
                raise RuntimeError("the pls weight schedule follows the train step, but levanter.trainer.current_train_step() is None")
            ramp = max(cfg.pls_off_end - cfg.pls_off_start, 1)
            weight = cfg.pls_weight * jnp.clip((cfg.pls_off_end - jnp.asarray(step, jnp.float32)) / ramp, 0.0, 1.0)
            loss, per = jax.lax.cond(jnp.asarray(step) < cfg.pls_off_end, lambda: supervised(weight), monitored)
        values = {f"pls/L{k}": per[k] for k in range(L - 1)}
        values[f"pls/L{L - 1}"] = jnp.asarray(_scalar(final), jnp.float32)
        if weight is not None:
            values["pls/weight"] = weight
        if cfg.pls_probe:
            values.update(self._probe_stats(outs, example, key=key, kw=kw))
        stats = {k: Metric.from_value(jnp.asarray(v, jnp.float32), ReductionType.MEAN) for k, v in sorted(values.items())}
        return loss, stats

    def readout_losses(self, example: LmExample, readouts: Sequence[int]) -> NamedArray:
        """Per-position next-token losses of the given layers' readouts, unreduced: axes (readout, *example.tokens.axes).
        Readout L-1 is exactly the baseline's per-position eval loss."""
        outs = self.layer_outputs(example.tokens, example.attn_mask, key=None)
        per = [self._readout_ce(outs, k, example, reduction=None, reduction_axis=()) for k in readouts]
        return hax.stack("readout", per).rearrange(("readout", *example.tokens.axes))


# ---------------------------------------------------------------------------------------------------------------------
# Per-layer evaluation: one forward pass, every readout, every statistic of levanter's TaggedEvaluator per readout.


def _readout_eval_loss_fn(model, batch, *, readouts: tuple[int, ...], EvalBatch: hax.Axis, mp: jmp.Policy | None):
    # levanter.eval._default_lm_eval_loss_fn with the per-readout losses stacked on a leading axis.
    model = inference_mode(model, True)
    named = _ensure_named_lm_example(batch, EvalBatch=EvalBatch, model_pos=model.Pos)
    if mp is not None:
        model = mp.cast_to_compute(model)
    per = model.readout_losses(named, readouts)
    return per.array, named.loss_weight.array, jnp.roll(named.tokens.array, -1, axis=-1)


class ReadoutTaggedEvaluator(TaggedEvaluator):
    """levanter's TaggedEvaluator for a loss_fn that returns per-position losses of R readouts, shape [R, batch, pos]
    (weights and token ids stay [batch, pos]). Every running statistic gains a leading readout axis; the per-readout
    results are exactly what TaggedEvaluator would report for that readout alone."""

    def __init__(self, readouts: Sequence[int], **kwargs):
        self.readouts = tuple(readouts)
        super().__init__(**kwargs)

    def _make_accum_for_batch(self):
        bytes_per_token = self.bytes_per_token
        log2e = jnp.log2(jnp.e)
        mesh = self.device_mesh
        tag_sharding = None if mesh is None else NamedSharding(mesh, P(None))
        readout_tag_sharding = None if mesh is None else NamedSharding(mesh, P(None, None))
        per_pos_out_sharding = self.per_pos_out_sharding
        has_tags = len(self.dataset.tag_to_index) > 0

        @hax.named_jit(axis_resources=self.axis_mapping)
        def accum_for_batch(model, state: _EvalRunningMeans, batch, tags):
            losses, weights, token_ids = self.loss_fn(model, batch)
            if losses.ndim != 3 or weights.ndim != 2 or token_ids.ndim != 2 or tags.ndim != 2:
                raise ValueError(f"expected losses [R, b, t] and [b, t] weights/ids, got {losses.shape}, {weights.shape}, {token_ids.shape}")
            weighted_loss = losses * weights[None]  # r b t
            this_loss = jnp.sum(weighted_loss, axis=(1, 2))  # r
            this_weights = jnp.sum(weights)
            this_weights_per_tag = jnp.einsum("bt,bk->k", weights, tags, out_sharding=tag_sharding)
            this_loss_per_tag = jnp.einsum("rbt,bk->rk", weighted_loss, tags, out_sharding=readout_tag_sharding)

            state = dataclasses.replace(state, token_avg_loss=state.token_avg_loss.add(this_loss / jnp.maximum(this_weights, 1.0), this_weights))
            if has_tags:
                safe_mean = jnp.where(this_weights_per_tag > 0, this_loss_per_tag / this_weights_per_tag, 0.0)
                state = dataclasses.replace(state, loss_per_tag=state.loss_per_tag.add(safe_mean, this_weights_per_tag))

            if bytes_per_token is not None:
                bytes_per_pos = bytes_per_token.at[token_ids].get(out_sharding=per_pos_out_sharding)
                this_bytes = jnp.sum(bytes_per_pos * weights)
                bytes_per_tag = jnp.einsum("bt,bt,bk->k", bytes_per_pos, weights, tags, out_sharding=tag_sharding)
                bpb = this_loss / jnp.maximum(this_bytes, 1.0) * log2e
                bpb_per_tag = this_loss_per_tag / jnp.maximum(bytes_per_tag, 1.0) * log2e
                state = dataclasses.replace(state, bpb=state.bpb.add(bpb, this_weights))
                if has_tags:
                    state = dataclasses.replace(state, bpb_per_tag=state.bpb_per_tag.add(bpb_per_tag, this_weights_per_tag))
            return state

        return accum_for_batch

    def evaluate(self, model):  # the base signature returns one result; this evaluator returns one per readout
        raise TypeError("ReadoutTaggedEvaluator.evaluate_readouts returns one EvalResult per readout")

    def evaluate_readouts(self, model) -> list[EvalResult]:
        R, K = len(self.readouts), self.dataset.num_tags
        state = hax.shard(_EvalRunningMeans.zeros_like(jnp.zeros((R,), jnp.float32), jnp.zeros((R, K), jnp.float32)))
        iterator = LoadingTimeTrackerIterator(self.loader)
        for batch, tags in tqdm(iterator, "eval readouts", total=len(self.loader)):
            state = self.accum_for_batch(model, state, batch, tags)
        return [self._result(jax.tree.map(lambda a, r=r: a[r], state), iterator.total_time) for r in range(R)]

    def _result(self, state: _EvalRunningMeans, loading_time: float) -> EvalResult:
        # TaggedEvaluator.evaluate's aggregation, for one readout's slice of the state.
        micro_avg_loss = state.token_avg_loss.mean.item()
        macro_avg_loss = jnp.mean(state.loss_per_tag.mean).item()
        if self.bytes_per_token is not None:
            micro_bpb = state.bpb.mean.item()
            macro_avg_bpb = jnp.mean(state.bpb_per_tag.mean).item()
        else:
            micro_bpb = None
            macro_avg_bpb = None
        mean_loss_per_tag = np.array(state.loss_per_tag.mean)
        tokens_per_tag = np.array(state.loss_per_tag.total)
        mean_bits_per_tag = np.array(state.bpb_per_tag.mean)
        bytes_per_tag = np.array(state.bpb_per_tag.total)
        tag_macro_loss: dict[str, float] = {}
        tag_micro_loss: dict[str, float] = {}
        tag_macro_bpb: dict[str, float] = {}
        tag_micro_bpb: dict[str, float] = {}
        for parent, children in self.hierarchy.items():
            mask = np.zeros(self.dataset.num_tags, dtype=bool)
            mask[children] = 1
            mask = mask & (tokens_per_tag > 0)
            tag_macro_loss[parent] = np.mean(mean_loss_per_tag, where=mask)
            tag_micro_loss[parent] = np.average(mean_loss_per_tag, weights=tokens_per_tag * mask)
            if self.bytes_per_token is not None:
                tag_macro_bpb[parent] = np.mean(mean_bits_per_tag, where=mask)
                tag_micro_bpb[parent] = np.average(mean_bits_per_tag, weights=bytes_per_tag * mask)
        for tag, index in self.dataset.tag_to_index.items():
            tag_micro_loss[tag] = float(mean_loss_per_tag[index])
            if self.bytes_per_token is not None:
                tag_micro_bpb[tag] = float(mean_bits_per_tag[index])
        return EvalResult(micro_avg_loss, macro_avg_loss, tag_macro_loss, tag_micro_loss, loading_time, micro_bpb, macro_avg_bpb, tag_macro_bpb, tag_micro_bpb)


def readout_log_dict(evaluator: ReadoutTaggedEvaluator, results: Sequence[EvalResult], total_time: float, prefix: str = "eval") -> dict:
    """``construct_log_dict`` per readout under ``{prefix}/L{k}``, without its per-tag info lines (R x tags of them)."""
    eval_logger = logging.getLogger("levanter.eval")
    level = eval_logger.level
    eval_logger.setLevel(logging.WARNING)
    try:
        log_dict: dict = {}
        for k, result in zip(evaluator.readouts, results):
            log_dict.update(construct_log_dict(evaluator, result, total_time, prefix=f"{prefix}/L{k}"))
    finally:
        eval_logger.setLevel(level)
    return log_dict


def cb_readout_evaluate(
    EvalBatch: hax.Axis,
    tagged_eval_sets,
    tokenizer,
    device_mesh,
    axis_mapping,
    max_examples_per_dataset: Optional[int],
    *,
    readouts: tuple[int, ...],
    mp: jmp.Policy | None,
    prefix: str = "eval",
):
    evaluator = ReadoutTaggedEvaluator(
        readouts,
        EvalBatch=EvalBatch,
        tagged_eval_sets=tagged_eval_sets,
        loss_fn=partial(_readout_eval_loss_fn, readouts=readouts, EvalBatch=EvalBatch, mp=mp),
        tokenizer=tokenizer,
        device_mesh=device_mesh,
        axis_mapping=axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )

    def eval_callback(step: StepInfo):
        with levanter.tracker.capture_time() as time_fn:
            results = evaluator.evaluate_readouts(step.model)
        log_dict = readout_log_dict(evaluator, results, time_fn(), prefix=prefix)
        levanter.tracker.log(log_dict, step=step.step)
        for k, result in zip(readouts, results):
            logger.info(f"{prefix}/L{k} loss {result.micro_avg_loss:.4f} macro {result.macro_avg_loss:.4f} bpb {result.micro_bpb}")

    return eval_callback
