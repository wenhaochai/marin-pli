# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sampled softmax for the training loss of the muonh_qwen3 baseline (nanoGPT speedrun record #92, ANVIL2).

Through most of training each device's cross-entropy normalises over a shared candidate set of P classes instead of
the whole vocabulary: every class that is a target somewhere in the device's microbatch, plus enough negatives to
reach P, taken from a golden-ratio stride sweep of the vocabulary (a permutation, since the stride is coprime with the
vocabulary size) that starts at an offset drawn from the step key and staggered per device. The lm_head rows of the
candidates are gathered and the baseline's fused cross-entropy kernel runs on that [Embed, P] head with each target
replaced by its position in the set, so the logits GEMM, the CE pass and both gradient GEMMs shrink by V / P. The
lm_head gradient is dense over the vocabulary with zeros outside the set.

The loss is biased (its normaliser misses the non-candidate mass), so P ramps up in stages and the run ends on the
full softmax. Stages follow the trainer's step (``levanter.trainer.current_train_step``): stage i covers steps
< ``ss_stage_ends[i]`` with ``ss_candidates[i]`` classes, and steps >= ``ss_stage_ends[-1]`` run the full softmax
through the baseline's own code path. A device whose microbatch has more distinct targets than P falls back to the full
softmax for that step (``train/ss/overflow_frac``), so no target is ever dropped. Evaluation (``key=None``) is the
baseline's full-vocabulary loss, so eval/paloma numbers stay comparable. The logged train/loss is the sampled loss
while a stage is active: it sits below the full-softmax loss and steps up at each stage change.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, cast

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax.sharding import PartitionSpec

import haliax as hax
from haliax import NamedArray
from haliax.core import flatten_all_axes_but
from haliax.partitioning import _get_mesh, current_thread_local_mapping, pspec_for, shard_map
from levanter.kernels.pallas.fused_cross_entropy_loss import fused_cross_entropy_loss_and_logsumexp_penalty as fused_ce
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmConfig, LmExample, split_activations
from levanter.models.loss import next_token_loss_weight
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.trainer import current_train_step

# The candidate sweep's key stream: folded off the step key, which the trunk still receives unchanged.
_SWEEP_KEY_FOLD = 0x55


@LmConfig.register_subclass("qwen3_sampled_softmax")
@dataclass(frozen=True)
class SampledSoftmaxQwen3Config(Qwen3Config):
    # Candidate count per stage and the step at which each stage ends (exclusive); the full softmax runs from the last end.
    ss_candidates: tuple[int, ...] = ()
    ss_stage_ends: tuple[int, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if len(self.ss_candidates) != len(self.ss_stage_ends):
            raise ValueError(f"one stage end per candidate count: {self.ss_candidates} vs {self.ss_stage_ends}")
        if any(p <= 0 for p in self.ss_candidates):
            raise ValueError(f"candidate counts must be positive, got {self.ss_candidates}")
        if any(b <= a for a, b in zip(self.ss_stage_ends, self.ss_stage_ends[1:])) or any(e < 0 for e in self.ss_stage_ends):
            raise ValueError(f"stage ends must be non-negative and strictly increasing, got {self.ss_stage_ends}")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return SampledSoftmaxQwen3LMHeadModel


class SampledSoftmaxQwen3LMHeadModel(Qwen3LMHeadModel):
    @classmethod
    def init(cls, Vocab, config: SampledSoftmaxQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)  # the baseline's parameters and initialisation, unchanged
        return cls(base.transformer, base.embeddings, base.lm_head)

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
        cfg = cast(SampledSoftmaxQwen3Config, self.config)
        kw = dict(reduction=reduction, reduction_axis=reduction_axis, logsumexp_weight=logsumexp_weight, logit_soft_cap=logit_soft_cap)
        if key is None or not cfg.ss_candidates:  # evaluation (the trainer passes no key) or no stages: the baseline loss
            return super().compute_next_token_loss(example, key=key, loss_dtype=loss_dtype, **kw)
        step = current_train_step()
        if step is None:
            raise RuntimeError("sampled softmax follows the train step, but levanter.trainer.current_train_step() is None")
        activations, aux_loss = split_activations(self.activations(example.tokens, example.attn_mask, key=key))
        loss, stats = sampled_next_token_loss(
            self.Pos,
            self.Embed,
            self.Vocab,
            activations,
            self.get_lm_head(),
            example.tokens,
            loss_weight=example.loss_weight,
            step=step,
            candidates=cfg.ss_candidates,
            stage_ends=cfg.ss_stage_ends,
            key=jrandom.fold_in(key, _SWEEP_KEY_FOLD),
            dtype=loss_dtype,
            **kw,
        )
        return loss + aux_loss, stats


@lru_cache(maxsize=8)
def stride_sweep(vocab_size: int) -> np.ndarray:
    """k * s mod V for k = 0..V-1 with s coprime to V near V / golden ratio: a permutation whose windows spread over the ids."""
    stride = int(vocab_size * 0.6180339887498949) | 1
    while math.gcd(stride, vocab_size) != 1:
        stride += 2
    return ((np.arange(vocab_size, dtype=np.int64) * stride) % vocab_size).astype(np.int32)


def build_candidates(present: jax.Array, num_present: jax.Array, num_candidates: int, offset: jax.Array, sweep: jax.Array) -> jax.Array:
    """Ascending ids of the P candidates: every present class plus the first P - num_present absent classes of the sweep
    read from ``offset``. Needs num_present <= P (the caller falls back to the full softmax otherwise)."""
    vocab_size = present.shape[0]
    order = sweep[(offset + jnp.arange(vocab_size, dtype=jnp.int32)) % vocab_size]
    absent = ~present[order]
    rank = jnp.cumsum(absent.astype(jnp.int32)) - 1  # rank of each absent class in sweep order
    chosen = jnp.zeros((vocab_size,), jnp.bool_).at[order].set(absent & (rank < num_candidates - num_present), unique_indices=True)
    return jnp.nonzero(present | chosen, size=num_candidates, fill_value=0)[0].astype(jnp.int32)


def sampled_cross_entropy(
    x: jax.Array,
    labels: jax.Array,
    w: jax.Array,
    *,
    stage: jax.Array,
    offset: jax.Array,
    candidates: tuple[int, ...],
    sweep: jax.Array,
    **kernel_kw,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-example cross-entropy of ``x @ w`` on one device, over stage ``stage``'s candidate set or the full vocabulary.

    Returns (loss [B], number of distinct labels, branch taken: the stage index, or len(candidates) for the full softmax).
    """
    vocab_size = w.shape[1]
    present = jnp.zeros((vocab_size,), jnp.bool_).at[labels].set(True)
    num_present = jnp.sum(present, dtype=jnp.int32)
    stage_p = jnp.asarray((*candidates, vocab_size), dtype=jnp.int32)[stage]
    branch = jnp.where(num_present > stage_p, len(candidates), stage)

    def full(_):
        return fused_ce(x, labels, w, reduction=None, weight=None, **kernel_kw)

    def sampled(num_candidates: int):
        def run(_):
            cand = build_candidates(present, num_present, num_candidates, offset, sweep)
            position = jnp.zeros((vocab_size,), jnp.int32).at[cand].set(jnp.arange(num_candidates, dtype=jnp.int32), unique_indices=True)
            w_cand = jnp.take(w, cand, axis=1, unique_indices=True, indices_are_sorted=True)
            return fused_ce(x, position[labels], w_cand, reduction=None, weight=None, **kernel_kw)

        return run

    loss = jax.lax.switch(branch, [sampled(p) for p in candidates] + [full], None)
    return loss, num_present, branch


def sampled_next_token_loss(
    Pos: hax.AxisSelector,
    Embed: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    true_ids: NamedArray,
    *,
    loss_weight: Optional[NamedArray],
    step: jax.Array,
    candidates: tuple[int, ...],
    stage_ends: tuple[int, ...],
    key: jax.Array,
    reduction: Optional[hax.ReductionFunction] = cast(Optional[hax.ReductionFunction], hax.mean),
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
) -> tuple[NamedArray, dict[str, Metric]]:
    """levanter.models.loss.maybe_fused_next_token_loss with the per-device candidate set; returns (loss, metrics)."""
    if logsumexp_weight:
        raise NotImplementedError("a z-loss over the candidate set is not the full-softmax z-loss; not supported")
    Pos = pred_embeddings.resolve_axis(hax.axis_name(Pos))
    Embed = pred_embeddings.resolve_axis(hax.axis_name(Embed))
    Vocab = pred_lm_head.resolve_axis(hax.axis_name(Vocab))
    if max(candidates) >= Vocab.size:
        raise ValueError(f"candidate counts {candidates} must be below the vocabulary size {Vocab.size}")
    target_y = hax.roll(true_ids, -1, Pos)
    if loss_weight is not None:
        dtype = loss_weight.dtype
    loss_weight = next_token_loss_weight(Pos, loss_weight)
    lm_head = pred_lm_head.rearrange((Embed, Vocab))

    stage = jnp.sum(jnp.asarray(step) >= jnp.asarray(stage_ends, dtype=jnp.int32)).astype(jnp.int32)
    offset = jrandom.randint(key, (), 0, Vocab.size, dtype=jnp.int32)
    sweep = jnp.asarray(stride_sweep(Vocab.size))
    kernel_kw = dict(logsumexp_weight=logsumexp_weight, block_size=None, dtype=dtype, logit_soft_cap=logit_soft_cap, precision=None)

    mesh = _get_mesh()
    sharded = mesh is not None and not getattr(mesh, "empty", False)
    axis_mapping = current_thread_local_mapping() or {}
    # Mesh axes that split the batch: devices sharing a batch shard must build the same set, the others stagger the sweep.
    batch_mesh_axes: tuple[str, ...] = ()
    if sharded:
        for entry in pspec_for(target_y, axis_mapping):
            batch_mesh_axes += tuple(entry) if isinstance(entry, tuple) else ((entry,) if entry is not None else ())
    num_shards = math.prod(mesh.shape[a] for a in batch_mesh_axes) if batch_mesh_axes else 1

    def per_device(shard_embeddings: NamedArray, shard_labels: NamedArray, shard_lm_head: NamedArray, stage, offset):
        batch_axes = hax.axis.without_axes(shard_embeddings.axes, Embed)
        flat_embeddings, _ = flatten_all_axes_but(shard_embeddings, "__BATCH__", batch_axes, reorder_to_front=True)
        batch_axis = flat_embeddings.resolve_axis("__BATCH__")
        x = flat_embeddings.rearrange((batch_axis, Embed)).array
        labels = hax.flatten_axes(shard_labels, shard_labels.axes, batch_axis).array.astype(jnp.int32)
        shard = jax.lax.axis_index(batch_mesh_axes) if batch_mesh_axes else 0
        offset = (offset + shard * (Vocab.size // num_shards)) % Vocab.size
        loss, num_present, branch = sampled_cross_entropy(
            x, labels, shard_lm_head.array, stage=stage, offset=offset, candidates=candidates, sweep=sweep, **kernel_kw
        )

        def per_token(a):
            return hax.named(jnp.broadcast_to(a, labels.shape), batch_axis).unflatten_axis(batch_axis, shard_labels.axes)

        return per_token(loss), per_token(num_present), per_token(branch)

    if sharded:
        # Explicit specs: the step-derived scalars are replicated (pspec_for can give a rank-0 array a one-entry spec).
        in_specs = (*(pspec_for(a, axis_mapping) for a in (pred_embeddings, target_y, lm_head)), PartitionSpec(), PartitionSpec())
        loss, num_present, branch = shard_map(per_device, in_specs=in_specs, axis_mapping=axis_mapping, check_rep=False)(
            pred_embeddings, target_y, lm_head, stage, offset
        )
    else:
        loss, num_present, branch = per_device(pred_embeddings, target_y, lm_head, stage, offset)

    loss = hax.nn.loss.maybe_reduce_loss(loss, reduction, reduction_axis, where=None, weight=loss_weight)
    # Per-token copies of per-device values: max/mean over tokens are max/mean over devices (equal token counts).
    overflow = (branch == len(candidates)) & (stage < len(candidates))
    stats = {
        "ss/present_max": Metric.from_value(hax.max(num_present).array.astype(jnp.float32), ReductionType.MAX),
        "ss/present_mean": Metric.from_value(hax.mean(num_present.astype(jnp.float32)).array, ReductionType.MEAN),
        "ss/overflow_frac": Metric.from_value(hax.mean(overflow.astype(jnp.float32)).array, ReductionType.MEAN),
        "ss/candidates": Metric.from_value(jnp.asarray((*candidates, Vocab.size), jnp.float32)[stage], ReductionType.LAST),
    }
    return loss, stats
