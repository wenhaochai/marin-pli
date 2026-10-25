# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Label smoothing toward the unigram distribution, from a given train step on (owner popup 2026-10-25, "单字分布平滑").

Yu et al. 2026 (Forking, arXiv 2610.00394) trace the cost of replayed data to softmax updates that raise the
continuations seen in training and push down the unseen ones. This variant gives every training position a target
q = (1 - eps) onehot(t) + eps * u, where u is the token distribution of the repeated text (a back-off prior, as
Kneser-Ney keeps mass for unseen events), so repeated updates stop pushing unseen continuations below their prior.

Per token, -sum_v q_v log p_v = CE + eps * (z_t - h . w_bar), with z_t = h . W[t] the target logit and
w_bar = sum_v u_v W[v] one vector per step: the fused cross-entropy stays as it is and the extra term costs a gather
and two dot products. The gradient with respect to the logits is p - q. Before ``us_from_step`` (train steps from
``levanter.trainer.current_train_step()``) and at evaluation (``key=None``) the loss is the plain cross-entropy, so
eval/paloma stays comparable and a run is the baseline up to that step. The logged ``ce_loss`` is the plain
cross-entropy throughout.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, cast

import jax
import jax.numpy as jnp
import numpy as np

import haliax as hax

from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmConfig, LmExample, split_activations
from levanter.models.loss import maybe_fused_next_token_loss, next_token_loss_weight
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.trainer import current_train_step


@lru_cache(maxsize=4)
def unigram(path: str, size: int) -> np.ndarray:
    """Token counts at ``path`` (an .npy of length <= size) as a distribution over ``size`` ids (padding ids get 0)."""
    c = np.load(path).astype(np.float64)
    if c.ndim != 1 or len(c) > size or c.sum() <= 0 or (c < 0).any():
        raise ValueError(f"{path}: need a non-negative 1-D count vector of length <= {size}, got shape {c.shape}")
    u = np.zeros(size, np.float32)
    u[: len(c)] = c / c.sum()
    return u


def unismooth_extra(Pos, Embed, Vocab, h, head, tokens, u):
    """Per-position z_t - h . w_bar in f32 (z_t: logit of the next token; w_bar = u . head)."""
    target = hax.roll(tokens, -1, Pos)
    rows = head.astype(jnp.float32).take(Vocab, target)                     # [.., Pos, Embed]
    hf = h.astype(jnp.float32)
    z_t = hax.dot(hf, rows, axis=Embed)
    w_bar = hax.dot(hax.named(jnp.asarray(u), Vocab), head.astype(jnp.float32), axis=Vocab)   # [Embed]
    return z_t - hax.dot(hf, w_bar, axis=Embed)


@LmConfig.register_subclass("qwen3_unismooth")
@dataclass(frozen=True)
class UniSmoothQwen3Config(Qwen3Config):
    us_eps: float = 0.0
    us_from_step: int = 0
    us_counts: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.us_eps < 1.0 or self.us_from_step < 0:
            raise ValueError(f"us_eps={self.us_eps}, us_from_step={self.us_from_step}: need 0 <= eps < 1, from_step >= 0")
        if self.us_eps > 0 and not self.us_counts:
            raise ValueError("us_eps > 0 needs us_counts (an .npy of token counts)")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return UniSmoothQwen3LMHeadModel


class UniSmoothQwen3LMHeadModel(Qwen3LMHeadModel):
    @classmethod
    def init(cls, Vocab, config: UniSmoothQwen3Config, *, key):  # type: ignore[override]
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
        cfg = cast(UniSmoothQwen3Config, self.config)
        step = current_train_step()
        if key is None or step is None or cfg.us_eps == 0 or reduction is not hax.mean or reduction_axis is not None or logsumexp_weight or logit_soft_cap:
            return super().compute_next_token_loss(example, key=key, reduction=reduction, reduction_axis=reduction_axis,
                                                   logsumexp_weight=logsumexp_weight, loss_dtype=loss_dtype, logit_soft_cap=logit_soft_cap)
        activations, aux_loss = split_activations(self.activations(example.tokens, example.attn_mask, key=key))
        head = self.get_lm_head()
        Pos = cast(hax.Axis, activations.resolve_axis(self.Pos.name))
        dtype = example.loss_weight.dtype if example.loss_weight is not None else loss_dtype
        ce = maybe_fused_next_token_loss(Pos, self.Embed, self.Vocab, activations, head, example.tokens,
                                         loss_weight=None, reduction=None, dtype=dtype)   # per-token CE (last position 0)
        extra = unismooth_extra(Pos, self.Embed, self.Vocab, activations, head, example.tokens, unigram(cfg.us_counts, self.Vocab.size))
        eps = jnp.where(jnp.asarray(step) >= cfg.us_from_step, cfg.us_eps, 0.0).astype(jnp.float32)
        weight = next_token_loss_weight(Pos, example.loss_weight)
        loss = hax.nn.loss.maybe_reduce_loss(ce + eps * extra, reduction, reduction_axis, where=None, weight=weight)
        ce_mean = hax.nn.loss.maybe_reduce_loss(ce, reduction, reduction_axis, where=None, weight=weight)
        ce_mean = ce_mean.array if isinstance(ce_mean, hax.NamedArray) else ce_mean
        metrics = {"ce_loss": Metric.from_value(jax.lax.stop_gradient(jnp.mean(ce_mean)).astype(jnp.float32), ReductionType.MEAN)}
        return loss + aux_loss, metrics
