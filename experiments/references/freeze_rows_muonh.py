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

That default has a side effect (review, 2026-10-13): AdamH sizes the head's step as lr * norm(p) along u / norm(u), and
once the frozen rows' Adam ratios fade, all of norm(u) comes from the kept rows, so they move by lr * norm(p) instead of
about lr * norm(p_kept), roughly 4x their baseline rate at K = 8000 of 128256. ``freeze_block_norm=True`` instead takes
the head's step as AdamH on the kept rows alone: u is AdamH's own m_hat / (sqrt(v_hat) + eps) read from the updated
state, restricted to the kept rows, the step is lr * norm(p_kept) along u_kept / norm(u_kept), and the kept rows are
renormalised to norm(p_kept); the frozen rows do not move, so the whole head keeps its norm too.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import haliax as hax
from levanter.optim.config import OptimizerConfig
from levanter.optim.adamh import ScaleByAdamHState
from levanter.optim.muonh import MuonHConfig
from levanter.optim.util import _NORM_FLOOR


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


def kept_block_step(updates, new_state, params, lo: int, keep_last: int, eps: float, b1: float, b2: float):
    """``updates`` with lm_head's update replaced by AdamH's norm-preserving step on the kept rows alone (ids below
    ``lo`` and the last ``keep_last``): u = m_hat / (sqrt(v_hat) + eps) from the AdamH state just updated, zero on the
    frozen rows; p_new = p - lr * u * norm(p_kept) / norm(u), the kept rows rescaled to norm(p_kept)."""
    adamh = [s for s in jax.tree_util.tree_leaves(new_state, is_leaf=lambda x: isinstance(x, ScaleByAdamHState)) if isinstance(s, ScaleByAdamHState)]
    if len(adamh) != 1:
        raise ValueError(f"expected one AdamH state (the head's), found {len(adamh)}")
    st, lr = adamh[0], new_state.hyperparams["learning_rate"]
    w = params.lm_head.weight
    m, v = st.mu.lm_head.weight, st.nu.lm_head.weight
    m, v = (x.rearrange(w.axes).array if isinstance(x, hax.NamedArray) else x for x in (m, v))
    p = w.array
    vocab = [i for i, a in enumerate(w.axes) if a.name == "vocab"]
    if len(vocab) != 1 or p.ndim != 2:
        raise ValueError(f"lm_head.weight axes {w.axes}: expected a 2-d table with one 'vocab' axis")
    V = p.shape[vocab[0]]
    ids = jnp.arange(V)
    kept = (ids < lo) | (ids >= V - keep_last)
    kept = kept[:, None] if vocab[0] == 0 else kept[None, :]
    c = st.count.astype(jnp.float32)   # updates so far, this one included
    u = (m / (1 - b1 ** c)) / (jnp.sqrt(v / (1 - b2 ** c)) + eps)
    u = jnp.where(kept, u, 0.0)
    pk = jnp.where(kept, p, 0.0)
    pk_norm = jnp.linalg.norm(pk)
    new = pk - lr * u * pk_norm / jnp.maximum(jnp.linalg.norm(u), _NORM_FLOOR)
    new = new / jnp.maximum(jnp.linalg.norm(new), _NORM_FLOOR) * pk_norm
    delta = jnp.where(kept, new - p, 0.0).astype(updates.lm_head.weight.array.dtype)
    return eqx.tree_at(lambda t: t.lm_head.weight, updates, hax.named(delta, w.axes).rearrange(updates.lm_head.weight.axes))


@OptimizerConfig.register_subclass("muonH_freeze_rows")
@dataclass(frozen=True)
class FreezeRowsMuonHConfig(MuonHConfig):
    freeze_lo: int = 8000
    freeze_keep_last: int = 256
    freeze_from_step: int = 0
    freeze_block_norm: bool = False   # after the freeze, AdamH's norm-preserving step on the kept rows alone

    def build(self, num_train_steps):
        if self.freeze_from_step < 0:
            raise ValueError(f"freeze_from_step={self.freeze_from_step}: must be >= 0")
        inner = super().build(num_train_steps)
        lo, keep, s0 = self.freeze_lo, self.freeze_keep_last, self.freeze_from_step

        eps, b1, b2, block = self.epsilon, self.beta1, self.beta2, self.freeze_block_norm

        def update_fn(updates, state, params=None, **extra):
            frozen = jnp.asarray(state.count) >= s0   # the index of this update
            updates, new_state = inner.update(freeze_rows(updates, lo, keep, frozen), state, params, **extra)
            updates = freeze_rows(updates, lo, keep, frozen)
            if block:
                updates = jax.lax.cond(frozen, lambda u: kept_block_step(u, new_state, params, lo, keep, eps, b1, b2), lambda u: u, updates)
            return updates, new_state

        return optax.GradientTransformationExtraArgs(inner.init, update_fn)
