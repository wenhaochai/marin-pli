# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
from functools import partial
from typing import Literal, Optional, cast, overload

import jax
import jax.numpy as jnp

import haliax as hax
from haliax import NamedArray
from haliax.core import flatten_all_axes_but
from haliax.nn import cross_entropy_loss_and_log_normalizers
from haliax.partitioning import _get_mesh, current_thread_local_mapping, shard_map
from levanter.kernels.pallas.fused_cross_entropy_loss import (
    Implementation,
    fused_cross_entropy_loss_and_logsumexp_penalty as fused_cross_entropy_loss_and_logsumexp_penalty_kernel,
)
from levanter.kernels.pallas.fused_cross_entropy_loss.batched_xla import (
    _full_vocab_b_tiled_forward_with_grads,
    _h100_full_vocab_b_tiled_block_size,
    _linear_softmax_cross_entropy_loss_full_vocab_b_tiled,
)

DEFAULT_REDUCTION = cast(hax.ReductionFunction, hax.mean)


def next_token_loss_weight(Pos: hax.Axis, loss_weight: Optional[NamedArray]) -> NamedArray:
    """Build the per-position loss weight for next-token prediction.

    The final position has no next token to predict, so it is always masked out. A
    caller-supplied ``loss_weight`` is multiplied by this mask (in its own dtype); when
    none is given the mask itself is returned as float32.
    """
    not_last_mask = hax.logical_not(hax.nn.one_hot(-1, Pos, dtype=jnp.bool_))  # type: ignore
    if loss_weight is not None:
        return loss_weight * not_last_mask.astype(loss_weight.dtype)
    return not_last_mask.astype(jnp.float32)


def maybe_fused_next_token_loss(
    Pos: hax.AxisSelector,
    Embed: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    true_ids: NamedArray,
    loss_weight: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = DEFAULT_REDUCTION,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
    block_size: int | None = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
) -> NamedArray:
    """
    Compute the next token loss using the fused kernel path.

    Args:
        Pos (hax.AxisSelector): Position axis selector.
        Vocab (hax.AxisSelector): Vocabulary axis selector.
        pred_embeddings (NamedArray): Predicted embeddings.
        pred_lm_head (NamedArray): Language model head weights.
        true_ids (NamedArray): True token IDs.
        loss_weight (Optional[NamedArray]): Mask to apply to the loss.
        reduction (Optional[hax.ReductionFunction]): Reduction function.
        reduction_axis (Optional[hax.AxisSelection]): Axis to apply reduction.
        logsumexp_weight (Optional[float]): Weight for logsumexp penalty.
        block_size (int | None): Optional vocabulary block size for processing.
        dtype (Optional[jnp.dtype]): Data type for the loss.
        logit_soft_cap (Optional[float]): Optional soft cap for logits
        precision (Optional[jax.lax.PrecisionLike]): Optional matmul precision override.
    Returns:
        NamedArray: Computed loss.
    """
    # Resolve axes
    Pos = cast(hax.Axis, pred_embeddings.resolve_axis(Pos.name))
    Vocab = pred_lm_head.resolve_axis(Vocab)

    # Shift target tokens to predict the next token
    target_y = hax.roll(true_ids, -1, Pos)

    # When a loss_weight is supplied, the fused kernel runs the loss in its dtype.
    if loss_weight is not None:
        dtype = loss_weight.dtype
    loss_weight = next_token_loss_weight(Pos, loss_weight)

    # Compute the loss with optional block-wise processing
    return fused_cross_entropy_loss_and_logsumexp_penalty(
        pred_embeddings,
        pred_lm_head,
        Contract=Embed,
        Label=Vocab,
        target_y=target_y,
        reduction=reduction,
        reduction_axis=reduction_axis,
        weight=loss_weight,
        logsumexp_weight=logsumexp_weight,
        block_size=block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )


# LEVANTER_CE_GRAD_IN_FWD=1 (speed work, 2026-10-25): a weighted-mean fused CE on the H100 b-tiled path computes its
# gradients in the forward, from the same logits GEMM per block, and its backward only scales them. The cotangent of a
# mean is weight / sum(weight) times one scalar c, so it is known before the backward up to c. The loss value is the
# usual one (computed with no gradient); the gradient reaches x and the head through a zero-valued carrier per example
# whose cotangent is c. Under plain evaluation the carrier's primal runs, which is the usual forward alone.
def _ce_grad_in_forward_enabled() -> bool:
    return os.environ.get("LEVANTER_CE_GRAD_IN_FWD", "0") == "1"


def _grad_in_forward_b_block(x: jax.Array, w: jax.Array) -> Optional[int]:
    """The batch tile of the H100 b-tiled path for these shapes, or None when the fused CE would take another path."""
    return _h100_full_vocab_b_tiled_block_size(x, w) if jax.default_backend() == "gpu" else None


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _ce_with_gradient_carrier(x, labels, w, g, b_block_size, dtype, logit_soft_cap, precision):
    """(per-example loss, zeros); the zeros carry the gradient of sum_b g_b loss_b, scaled by their cotangent."""
    loss, _ = _linear_softmax_cross_entropy_loss_full_vocab_b_tiled(
        x, labels, w, b_block_size=b_block_size, dtype=dtype, logit_soft_cap=logit_soft_cap, precision=precision
    )
    return loss, jnp.zeros(x.shape[:1], jnp.float32)


# LEVANTER_CE_SKIP_UNUSED_DX=1 (speed work, 2026-10-26; read at import): the rule learns which inputs take part in the
# differentiation (JAX symbolic zeros), so an input under stop_gradient (a probe readout of a detached backbone) gets no
# input-gradient GEMM; the gradients that are used do not change.
_CE_SKIP_UNUSED_DX = os.environ.get("LEVANTER_CE_SKIP_UNUSED_DX", "0") == "1"


def _ce_with_gradient_carrier_fwd(x, labels, w, g, b_block_size, dtype, logit_soft_cap, precision):
    need_grad_x = True
    if _CE_SKIP_UNUSED_DX:  # each array argument arrives as CustomVJPPrimal(value, perturbed)
        need_grad_x = x.perturbed
        x, labels, w, g = x.value, labels.value, w.value, g.value
    loss, _, grad_x, grad_w = _full_vocab_b_tiled_forward_with_grads(
        x, labels, w, g, b_block_size=b_block_size, dtype=dtype, logit_soft_cap=logit_soft_cap, precision=precision,
        need_grad_x=need_grad_x,
    )
    return (loss, jnp.zeros(x.shape[:1], jnp.float32)), (grad_x, grad_w, g)


def _ce_with_gradient_carrier_bwd(b_block_size, dtype, logit_soft_cap, precision, residuals, cotangents):
    grad_x, grad_w, g = residuals
    _, carrier_ct = cotangents  # the loss output is used under stop_gradient; its cotangent is zero
    if isinstance(carrier_ct, jax.custom_derivatives.SymbolicZero):  # the carrier is not differentiated either
        return None, None, None, None
    c = carrier_ct[0]  # every example's carrier gets the same cotangent (the caller sums the carriers)
    scale = lambda t: (t.astype(jnp.float32) * c).astype(t.dtype)
    return (scale(grad_x) if grad_x is not None else None), None, scale(grad_w), jnp.zeros_like(g)


_ce_with_gradient_carrier.defvjp(
    _ce_with_gradient_carrier_fwd, _ce_with_gradient_carrier_bwd, symbolic_zeros=_CE_SKIP_UNUSED_DX
)


def _mean_ce_grad_in_forward(
    pred_embeddings: NamedArray,
    lm_head: NamedArray,
    Contract: hax.Axis,
    Label: hax.Axis,
    target_y: NamedArray,
    weight: NamedArray,
    *,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
) -> Optional[NamedArray]:
    """The weighted mean of the fused CE with gradient-in-forward, or None when a shard would not take the b-tiled path."""
    denom = hax.sum(weight)  # the denominator hax.nn.loss.maybe_reduce_loss divides by
    # The mean's cotangent per example for a unit cotangent of the mean, as its VJP forms it: (1 / denom) * weight.
    g_all = hax.where(denom != 0, (1.0 / denom) * weight, hax.zeros_like(weight))
    taken: list[bool] = []

    def impl(shard_embeddings: NamedArray, shard_labels: NamedArray, shard_lm_head: NamedArray, shard_g: NamedArray):
        batch_axes = hax.axis.without_axes(shard_embeddings.axes, Contract)
        flat_embeddings, _ = flatten_all_axes_but(shard_embeddings, "__BATCH__", batch_axes, reorder_to_front=True)
        batch_axis = flat_embeddings.resolve_axis("__BATCH__")
        x = flat_embeddings.rearrange((batch_axis, Contract)).array
        labels = hax.flatten_axes(shard_labels, shard_labels.axes, batch_axis).array.astype(jnp.int32)
        g = hax.flatten_axes(shard_g.rearrange(shard_labels.axes), shard_labels.axes, batch_axis).array
        w = shard_lm_head.array
        b_block = _grad_in_forward_b_block(x, w)
        taken.append(b_block is not None)
        if b_block is None:  # not the b-tiled path: the caller falls back and discards these outputs. Plain arrays:
            # hax.zeros would add a sharding constraint, which a manual shard_map rejects (w196/w197 smokes, 2026-10-26)
            z = hax.named(jnp.zeros(tuple(a.size for a in shard_labels.axes), jnp.float32), shard_labels.axes)
            return z, z
        loss, carrier = _ce_with_gradient_carrier(x, labels, w, g, b_block, dtype, logit_soft_cap, precision)
        unflat = lambda a: hax.named(a, batch_axis).unflatten_axis(batch_axis, shard_labels.axes)
        return unflat(jax.lax.stop_gradient(loss)), unflat(carrier)

    mesh = _get_mesh()
    if mesh is None or getattr(mesh, "empty", False):
        loss, carrier = impl(pred_embeddings, target_y, lm_head, g_all)
    else:
        axis_mapping = current_thread_local_mapping() or {}
        loss, carrier = shard_map(impl, axis_mapping=axis_mapping, check_rep=False)(pred_embeddings, target_y, lm_head, g_all)
    if not all(taken):
        return None
    reduced = hax.nn.loss.maybe_reduce_loss(loss, hax.mean, None, where=None, weight=weight)
    return reduced + hax.sum(carrier)


def next_token_loss(
    Pos: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    logits: NamedArray,
    true_ids: NamedArray,
    loss_weight: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = DEFAULT_REDUCTION,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
):
    """
    Compute the next token loss with optional logsumexp penalty.

    Args:
        Pos: axis selector for the position axis
        Vocab: axis selector for the vocabulary axis
        logits: predicted logits
        true_ids: true token IDs (not shifted)
        loss_weight: mask to apply to the loss
        reduction: reduction function or None to disable reduction
        reduction_axis: axis to apply reduction. None means all axes
        logsumexp_weight: weight for the logsumexp penalty
        logit_soft_cap: optional soft cap for logits
    Returns:
        NamedArray: computed loss
    """
    Pos = logits.resolve_axis(hax.axis_name(Pos))

    target_y = hax.roll(true_ids, -1, Pos)
    target_y_full = hax.nn.one_hot(target_y, Vocab, dtype=logits.dtype)

    loss_weight = next_token_loss_weight(Pos, loss_weight)

    return cross_entropy_and_logsumexp_penalty(
        Vocab=Vocab,
        pred_y=logits,
        target_y=target_y_full,
        reduction=reduction,
        reduction_axis=reduction_axis,
        weight=loss_weight,
        logsumexp_weight=logsumexp_weight,
    )


def cross_entropy_and_logsumexp_penalty(
    Vocab: hax.Axis,
    pred_y: NamedArray,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = DEFAULT_REDUCTION,
    reduction_axis: Optional[hax.AxisSelection] = None,
    weight: Optional[NamedArray] = None,
    logsumexp_weight=0.0,
) -> NamedArray:
    """A loss function that combines cross entropy loss with a logsumexp penalty."""

    loss, log_normalizers = cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_y)

    if logsumexp_weight is not None and logsumexp_weight != 0.0:
        loss = loss + logsumexp_weight * (log_normalizers**2)

    return hax.nn.loss.reduce_loss(loss, reduction, reduction_axis, weight=weight)


@overload
def fused_cross_entropy_loss_and_logsumexp_penalty(
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    Contract: hax.AxisSelector,
    Label: hax.AxisSelector,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = DEFAULT_REDUCTION,
    reduction_axis: Optional[hax.AxisSelection] = None,
    weight: Optional[NamedArray] = None,
    logsumexp_weight: float | None = 0.0,
    block_size: int | None = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | None = None,
    return_argmax: Literal[False] = False,
) -> NamedArray: ...


@overload
def fused_cross_entropy_loss_and_logsumexp_penalty(
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    Contract: hax.AxisSelector,
    Label: hax.AxisSelector,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = DEFAULT_REDUCTION,
    reduction_axis: Optional[hax.AxisSelection] = None,
    weight: Optional[NamedArray] = None,
    logsumexp_weight: float | None = 0.0,
    block_size: int | None = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | None = None,
    return_argmax: Literal[True] = True,
) -> tuple[NamedArray, NamedArray]: ...


def fused_cross_entropy_loss_and_logsumexp_penalty(
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    Contract: hax.AxisSelector,
    Label: hax.AxisSelector,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = DEFAULT_REDUCTION,
    reduction_axis: Optional[hax.AxisSelection] = None,
    weight: Optional[NamedArray] = None,
    logsumexp_weight: float | None = 0.0,
    block_size: int | None = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
    implementation: Implementation | None = None,
    return_argmax: bool = False,
) -> NamedArray | tuple[NamedArray, NamedArray]:
    """
    Compute cross-entropy loss and logsumexp penalty using the fused Pallas kernel.

    Args:
        pred_embeddings (NamedArray): Predicted embeddings.
        pred_lm_head (NamedArray): Language model head weights.
        Contract (hax.AxisSelector): Axis to contract over.
        Label (hax.AxisSelector): Label (Vocab) axis.
        target_y (NamedArray): Target token ids.
        reduction (Optional[hax.ReductionFunction]): Reduction function.
        reduction_axis (Optional[hax.AxisSelection]): Axis to apply reduction.
        weight (Optional[NamedArray]): Sample weights to apply to the loss.
        logsumexp_weight (float): Weight for logsumexp penalty.
        block_size (int | None): Optional vocabulary block size for processing.
        dtype (Optional[jnp.dtype]): Data type for the loss.
        precision (Optional[jax.lax.PrecisionLike]): Optional matmul precision override for the fused kernel.
        implementation (Implementation | None): Backend selector ("xla", "pallas_tpu", etc.).
        return_argmax (bool): Whether to return per-position argmax token ids as well.

    Returns:
        If return_argmax=False: computed loss.
        If return_argmax=True: tuple of (loss, argmax token ids).
    """

    Contract = pred_embeddings.resolve_axis(Contract)
    Label = pred_lm_head.resolve_axis(Label)

    lm_head = pred_lm_head.rearrange((Contract, Label))

    if (
        _ce_grad_in_forward_enabled()
        and not return_argmax
        and weight is not None
        and reduction_axis is None
        and hax.nn.loss._is_mean_reduction(reduction)
        and not logsumexp_weight
        and implementation in (None, "batched_xla")
    ):
        reduced = _mean_ce_grad_in_forward(
            pred_embeddings,
            lm_head,
            Contract,
            Label,
            target_y,
            weight,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )
        if reduced is not None:
            return reduced

    def fused_impl(
        shard_embeddings: NamedArray, shard_labels: NamedArray, shard_lm_head: NamedArray
    ) -> NamedArray | tuple[NamedArray, NamedArray]:
        batch_axes = hax.axis.without_axes(shard_embeddings.axes, Contract)
        flat_embeddings, _ = flatten_all_axes_but(shard_embeddings, "__BATCH__", batch_axes, reorder_to_front=True)
        batch_axis = flat_embeddings.resolve_axis("__BATCH__")
        flat_embeddings = flat_embeddings.rearrange((batch_axis, Contract))

        flat_labels = hax.flatten_axes(shard_labels, shard_labels.axes, batch_axis)

        kernel_output = fused_cross_entropy_loss_and_logsumexp_penalty_kernel(
            flat_embeddings.array,
            flat_labels.array.astype(jnp.int32),
            shard_lm_head.array,
            reduction=None,
            weight=None,
            logsumexp_weight=logsumexp_weight,
            block_size=block_size,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
            implementation=implementation,
            return_argmax=return_argmax,
        )
        if return_argmax:
            loss_flat, argmax_flat = cast(tuple[jax.Array, jax.Array], kernel_output)
            loss = hax.named(loss_flat, batch_axis).unflatten_axis(batch_axis, shard_labels.axes)
            argmax = hax.named(argmax_flat.astype(jnp.int32), batch_axis).unflatten_axis(batch_axis, shard_labels.axes)
            return loss, argmax
        loss_flat = cast(jax.Array, kernel_output)
        return hax.named(loss_flat, batch_axis).unflatten_axis(batch_axis, shard_labels.axes)

    mesh = _get_mesh()
    if mesh is None or getattr(mesh, "empty", False):
        output = fused_impl(pred_embeddings, target_y, lm_head)
    else:
        axis_mapping = current_thread_local_mapping() or {}
        output = shard_map(fused_impl, axis_mapping=axis_mapping, check_rep=False)(pred_embeddings, target_y, lm_head)

    if return_argmax:
        loss, argmax = cast(tuple[NamedArray, NamedArray], output)
        reduced_loss = hax.nn.loss.maybe_reduce_loss(loss, reduction, reduction_axis, where=None, weight=weight)
        return reduced_loss, argmax

    loss = cast(NamedArray, output)
    return hax.nn.loss.maybe_reduce_loss(loss, reduction, reduction_axis, where=None, weight=weight)
