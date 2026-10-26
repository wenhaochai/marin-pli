# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Flooding of the training loss toward a per-step level, from a given train step on (owner popup 2026-10-25, the four
extra vocab jobs: "按这 4 个提交").

Ishida et al. 2020 (Do We Need Zero Training Loss After Achieving Zero Training Error?, arXiv 2002.08709) replace the
training loss L by |L - b| + b: above the flood level b the gradient is unchanged, below it the sign flips and the step
climbs back to b. Here b is a curve over train steps, read from an .npy (scripts/della/flood_levels.py): the unique-token
run's train loss at the same step (smoothed), or a point between it and the plain 8-pass run's, so an 8-pass run cannot
fit its repeated text much better than a run that never repeats. The loss is flooded per batch (the batch-mean loss,
z-loss included, as the logged train/loss of the run that set the level). Before ``fl_from_step`` (train steps from
``levanter.trainer.current_train_step()``) and at evaluation (``key=None``) the loss is the plain one, so eval/paloma
stays comparable and a run is the baseline up to that step. The logged ``ce_loss`` is the unflooded loss throughout;
``flood_level`` is b and ``flood_ascent`` is 1 on steps where the loss sat below b.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, cast

import jax
import jax.numpy as jnp
import numpy as np

import haliax as hax

from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.trainer import current_train_step


@lru_cache(maxsize=4)
def flood_levels(path: str) -> np.ndarray:
    """The level curve at ``path``: a 1-D float array, entry s = the flood level at train step s."""
    b = np.load(path).astype(np.float32)
    if b.ndim != 1 or len(b) < 2 or not np.isfinite(b).all() or (b <= 0).any():
        raise ValueError(f"{path}: need a 1-D positive finite level curve, got shape {b.shape}")
    return b


@LmConfig.register_subclass("qwen3_flood")
@dataclass(frozen=True)
class FloodQwen3Config(Qwen3Config):
    fl_level: str = ""
    fl_from_step: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.fl_from_step < 0:
            raise ValueError(f"fl_from_step={self.fl_from_step}: need >= 0")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return FloodQwen3LMHeadModel


class FloodQwen3LMHeadModel(Qwen3LMHeadModel):
    @classmethod
    def init(cls, Vocab, config: FloodQwen3Config, *, key):  # type: ignore[override]
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
        cfg = cast(FloodQwen3Config, self.config)
        loss = super().compute_next_token_loss(example, key=key, reduction=reduction, reduction_axis=reduction_axis,
                                               logsumexp_weight=logsumexp_weight, loss_dtype=loss_dtype, logit_soft_cap=logit_soft_cap)
        step = current_train_step()
        if key is None or step is None or not cfg.fl_level or reduction is not hax.mean or reduction_axis is not None:
            return loss
        L = loss.array if isinstance(loss, hax.NamedArray) else loss
        levels = jnp.asarray(flood_levels(cfg.fl_level))
        s = jnp.asarray(step)
        b = levels[jnp.clip(s, 0, levels.shape[0] - 1)]
        on = s >= cfg.fl_from_step
        flooded = jnp.where(on, jnp.abs(L - b) + b, L)
        metrics = {
            "ce_loss": Metric.from_value(jax.lax.stop_gradient(L).astype(jnp.float32), ReductionType.MEAN),
            "flood_level": Metric.from_value(jnp.where(on, b, 0.0).astype(jnp.float32), ReductionType.MEAN),
            "flood_ascent": Metric.from_value((on & (jax.lax.stop_gradient(L) < b)).astype(jnp.float32), ReductionType.MEAN),
        }
        return flooded, metrics
