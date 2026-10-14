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

``freeze_table="embed"`` (Q4c, owner 2026-10-13) freezes the same rows of the input embedding instead: MuonH puts the
embedding in its plain-Adam group (no norm preservation), so zeroing the rows' gradient and update leaves the other rows
exactly as Adam moves them; only the Adam group's gradient clipping sees a slightly smaller norm after the freeze.

``freeze_rare_scale`` > 0 (owner, 2026-10-14; Q5b follow-up G) slows the rare output rows instead of stopping them: from
``freeze_from_step`` on they keep their gradient and their Adam moments, and the head's step is AdamH on each block
alone, the kept rows as with ``freeze_block_norm`` and the rare rows with the learning rate times ``freeze_rare_scale``,
each block renormalised to its own norm. At a scale of 0 the old ``freeze_block_norm`` path runs unchanged (rare rows'
gradient and moments zeroed); a positive scale needs ``freeze_block_norm`` and the head.
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


TABLES = {   # freeze_table -> the table's weight in a model-shaped tree (params, gradients, updates)
    "lm_head": lambda t: t.lm_head.weight,
    "embed": lambda t: t.embeddings.token_embeddings.weight,   # MuonH's plain-Adam group: no norm preservation
}


def freeze_rows(tree, lo: int, keep_last: int, frozen, table: str = "lm_head"):
    """``tree`` with rows lo .. V - keep_last - 1 of the table (lm_head.weight or the input embedding) zeroed where
    ``frozen`` (a traced bool)."""
    get = TABLES[table]
    w = get(tree)
    if w is None:
        return tree
    vocab = [a for a in w.axes if a.name == "vocab"]
    if len(vocab) != 1:
        raise ValueError(f"{table} weight axes {w.axes}: expected one 'vocab' axis")
    V = vocab[0]
    if not 0 < lo < V.size - keep_last:
        raise ValueError(f"freeze_lo={lo} must lie in (0, {V.size - keep_last}) for a vocabulary of {V.size}")
    ids = hax.arange(V)
    rows = (ids >= lo) & (ids < V.size - keep_last) & frozen
    return eqx.tree_at(get, tree, hax.where(rows, hax.zeros_like(w), w).rearrange(w.axes))


def block_delta(p, u, mask, lr):
    """AdamH's norm-preserving step on the rows in ``mask`` alone, as a delta that is zero elsewhere."""
    pb = jnp.where(mask, p, 0.0)
    ub = jnp.where(mask, u, 0.0)
    pb_norm = jnp.linalg.norm(pb)
    new = pb - lr * ub * pb_norm / jnp.maximum(jnp.linalg.norm(ub), _NORM_FLOOR)
    new = new / jnp.maximum(jnp.linalg.norm(new), _NORM_FLOOR) * pb_norm
    return jnp.where(mask, new - p, 0.0)


def kept_block_step(updates, new_state, params, lo: int, keep_last: int, eps: float, b1: float, b2: float, rare_scale: float = 0.0):
    """``updates`` with lm_head's update replaced by AdamH's norm-preserving step on the kept rows alone (ids below
    ``lo`` and the last ``keep_last``): u = m_hat / (sqrt(v_hat) + eps) from the AdamH state just updated, zero on the
    frozen rows; p_new = p - lr * u * norm(p_kept) / norm(u), the kept rows rescaled to norm(p_kept). With
    ``rare_scale`` > 0 the rare rows also take the same step on their own block, at lr * rare_scale."""
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
    delta = block_delta(p, u, kept, lr)
    if rare_scale > 0:
        delta = delta + block_delta(p, u, ~kept, lr * rare_scale)
    delta = delta.astype(updates.lm_head.weight.array.dtype)
    return eqx.tree_at(lambda t: t.lm_head.weight, updates, hax.named(delta, w.axes).rearrange(updates.lm_head.weight.axes))


@OptimizerConfig.register_subclass("muonH_freeze_rows")
@dataclass(frozen=True)
class FreezeRowsMuonHConfig(MuonHConfig):
    freeze_lo: int = 8000
    freeze_keep_last: int = 256
    freeze_from_step: int = 0
    freeze_block_norm: bool = False   # after the freeze, AdamH's norm-preserving step on the kept rows alone
    freeze_table: str = "lm_head"     # lm_head (output rows, AdamH) | embed (input rows, plain Adam; 2026-10-13, Q4c)
    freeze_rare_scale: float = 0.0    # > 0: the rare rows keep learning at this times the learning rate (2026-10-14, G)

    def build(self, num_train_steps):
        if self.freeze_from_step < 0:
            raise ValueError(f"freeze_from_step={self.freeze_from_step}: must be >= 0")
        inner = super().build(num_train_steps)
        lo, keep, s0 = self.freeze_lo, self.freeze_keep_last, self.freeze_from_step

        eps, b1, b2, block, table = self.epsilon, self.beta1, self.beta2, self.freeze_block_norm, self.freeze_table
        if table not in TABLES:
            raise ValueError(f"freeze_table={table!r}: one of {sorted(TABLES)}")
        if block and table != "lm_head":
            raise ValueError("freeze_block_norm applies to the AdamH head only (the input table is plain Adam)")
        scale = self.freeze_rare_scale
        if not 0 <= scale < 1:
            raise ValueError(f"freeze_rare_scale={scale}: must lie in [0, 1)")
        if scale > 0 and not (block and table == "lm_head"):
            raise ValueError("freeze_rare_scale needs freeze_block_norm on the head")

        def update_fn(updates, state, params=None, **extra):
            frozen = jnp.asarray(state.count) >= s0   # the index of this update
            if scale > 0:   # slowed, not stopped: the rare rows keep their gradient and moments
                updates, new_state = inner.update(updates, state, params, **extra)
                updates = jax.lax.cond(frozen, lambda u: kept_block_step(u, new_state, params, lo, keep, eps, b1, b2, scale), lambda u: u, updates)
                return updates, new_state
            updates, new_state = inner.update(freeze_rows(updates, lo, keep, frozen, table), state, params, **extra)
            updates = freeze_rows(updates, lo, keep, frozen, table)
            if block:
                updates = jax.lax.cond(frozen, lambda u: kept_block_step(u, new_state, params, lo, keep, eps, b1, b2), lambda u: u, updates)
            return updates, new_state

        return optax.GradientTransformationExtraArgs(inner.init, update_fn)
