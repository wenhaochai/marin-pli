# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Focal loss for the training loss of the muonh_qwen3 baseline (Lin et al. 2017, as tried for LLM pretraining in MiLe,
arXiv 2310.19531). Each token's cross-entropy l = -log p becomes (1 - p)^gamma * l with p = exp(-l), the full-softmax
probability of the target, so confident tokens weigh less; the gradient flows through both factors, as in the
original definition. The per-token losses then reduce exactly as the baseline's do (weighted mean over the loss
weight), so gamma = 0 is the baseline loss. No renormalisation: MuonH and Adam(H) updates do not depend on a constant
loss scale, only the relative weights between tokens change. Evaluation (``key=None``) is the plain cross-entropy, so
eval/paloma stays comparable.

``focal_next_token_loss`` is shared with over_vocab_qwen3 (VARIANT=ovfocal: focal on both of OV's heads).
"""
from dataclasses import dataclass
from typing import Optional, cast

import jax
import jax.numpy as jnp

import haliax as hax
from haliax import NamedArray

from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmConfig, LmExample, split_activations
from levanter.models.loss import maybe_fused_next_token_loss, next_token_loss_weight
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel

# 1 - p is clamped here before the power: masked positions carry l = 0 exactly, where (1 - p)^(gamma - 1) in the
# gradient of (1 - p)^gamma would be infinite for gamma < 1 (inf * 0 = NaN).
_MIN_ONE_MINUS_P = 1e-12


def focal_transform(loss: NamedArray, gamma: float) -> NamedArray:
    """(1 - exp(-l))^gamma * l per token."""
    one_minus_p = -jnp.expm1(-loss.array)
    weight = jnp.maximum(one_minus_p, _MIN_ONE_MINUS_P) ** gamma
    return hax.named(weight * loss.array, loss.axes)


def focal_next_token_loss(
    Pos,
    Embed,
    Vocab,
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    true_ids: NamedArray,
    *,
    gamma: float,
    loss_weight: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = cast(Optional[hax.ReductionFunction], hax.mean),
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
):
    """maybe_fused_next_token_loss with each token's cross-entropy replaced by its focal loss.

    Returns (reduced focal loss, plain cross-entropy with the same reduction and no gradient) for logging.
    """
    if logsumexp_weight:
        raise ValueError("focal loss with a logsumexp penalty is not supported")
    Pos = cast(hax.Axis, pred_embeddings.resolve_axis(Pos.name))
    if loss_weight is not None:  # the fused kernel runs in the loss weight's dtype, as in maybe_fused_next_token_loss
        dtype = loss_weight.dtype
    per_token = maybe_fused_next_token_loss(
        Pos, Embed, Vocab, pred_embeddings, pred_lm_head, true_ids,
        loss_weight=None, reduction=None, dtype=dtype, logit_soft_cap=logit_soft_cap,
    )  # raw per-token cross-entropy (the window's last position is 0)
    weight = next_token_loss_weight(Pos, loss_weight)
    focal = hax.nn.loss.maybe_reduce_loss(focal_transform(per_token, gamma), reduction, reduction_axis, where=None, weight=weight)
    ce = hax.nn.loss.maybe_reduce_loss(per_token, reduction, reduction_axis, where=None, weight=weight)
    return focal, jax.lax.stop_gradient(ce.array if isinstance(ce, NamedArray) else ce)


@LmConfig.register_subclass("qwen3_focal")
@dataclass(frozen=True)
class FocalQwen3Config(Qwen3Config):
    focal_gamma: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return FocalQwen3LMHeadModel


class FocalQwen3LMHeadModel(Qwen3LMHeadModel):
    @classmethod
    def init(cls, Vocab, config: FocalQwen3Config, *, key):  # type: ignore[override]
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
        cfg = cast(FocalQwen3Config, self.config)
        if key is None:  # evaluation: the baseline's cross-entropy
            return super().compute_next_token_loss(example, key=None, reduction=reduction, reduction_axis=reduction_axis,
                                                   logsumexp_weight=logsumexp_weight, loss_dtype=loss_dtype, logit_soft_cap=logit_soft_cap)
        activations, aux_loss = split_activations(self.activations(example.tokens, example.attn_mask, key=key))
        loss, ce = focal_next_token_loss(
            self.Pos, self.Embed, self.Vocab, activations, self.get_lm_head(), example.tokens,
            gamma=cfg.focal_gamma, loss_weight=example.loss_weight, reduction=reduction, reduction_axis=reduction_axis,
            logsumexp_weight=logsumexp_weight, dtype=loss_dtype, logit_soft_cap=logit_soft_cap,
        )
        metrics = {"ce_loss": Metric.from_value(jnp.mean(ce).astype(jnp.float32), ReductionType.MEAN)}
        return loss + aux_loss, metrics
