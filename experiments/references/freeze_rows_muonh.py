# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH with the output rows of rare tokens frozen from a given step on (vocabulary Q5b, owner 2026-10-12).

The model is the muonh_qwen3 baseline unchanged. ``FreezeRowsMuonHConfig`` builds the baseline's MuonH optimizer and
wraps it: from update ``freeze_from_step`` on (the update index, read from the inject_hyperparams count, so update s is
the one applied after train step s), the rows ``freeze_lo .. V - freeze_keep_last - 1`` of ``lm_head.weight`` (the
tokens outside the K-token truncation, the 256 specials excluded) get

1. a zero gradient before MuonH sees it, so AdamH's moments, its update norm and the head's gradient clipping come from
   the other rows only (the moments of the frozen rows decay to zero);
2. a zero update after MuonH, so the rows stay exactly at their values from the end of step ``freeze_from_step - 1``.

Before ``freeze_from_step`` the wrapper changes nothing (the baseline bit for bit). The optimizer state keeps the
baseline's structure (InjectStatefulHyperparamsState), so learning-rate logging and checkpoints are unchanged. AdamH's
norm-preserving step rescales the head by norm(p) / norm(p_new); with the rare rows' update zeroed afterwards, the head's
Frobenius norm is no longer held exactly, only the norm of what moves is rescaled (a difference from the baseline after
the freeze, the same in the full-data and the repeated run).
"""

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import optax

import haliax as hax
from levanter.optim.config import OptimizerConfig
from levanter.optim.muonh import MuonHConfig


def freeze_rows(tree, lo: int, keep_last: int, frozen):
    """``tree`` with rows lo .. V - keep_last - 1 of lm_head.weight zeroed where ``frozen`` (a traced bool)."""
    w = tree.lm_head.weight
    if w is None:
        return tree
    vocab = [a for a in w.axes if a.name == "vocab"]
    if len(vocab) != 1:
        raise ValueError(f"lm_head.weight axes {w.axes}: expected one 'vocab' axis")
    V = vocab[0]
    if not 0 < lo < V.size - keep_last:
        raise ValueError(f"freeze_lo={lo} must lie in (0, {V.size - keep_last}) for a vocabulary of {V.size}")
    ids = hax.arange(V)
    rows = (ids >= lo) & (ids < V.size - keep_last) & frozen
    return eqx.tree_at(lambda t: t.lm_head.weight, tree, hax.where(rows, hax.zeros_like(w), w).rearrange(w.axes))


@OptimizerConfig.register_subclass("muonH_freeze_rows")
@dataclass(frozen=True)
class FreezeRowsMuonHConfig(MuonHConfig):
    freeze_lo: int = 8000
    freeze_keep_last: int = 256
    freeze_from_step: int = 0

    def build(self, num_train_steps):
        if self.freeze_from_step < 0:
            raise ValueError(f"freeze_from_step={self.freeze_from_step}: must be >= 0")
        inner = super().build(num_train_steps)
        lo, keep, s0 = self.freeze_lo, self.freeze_keep_last, self.freeze_from_step

        def update_fn(updates, state, params=None, **extra):
            frozen = jnp.asarray(state.count) >= s0   # the index of this update
            updates, new_state = inner.update(freeze_rows(updates, lo, keep, frozen), state, params, **extra)
            return freeze_rows(updates, lo, keep, frozen), new_state

        return optax.GradientTransformationExtraArgs(inner.init, update_fn)
