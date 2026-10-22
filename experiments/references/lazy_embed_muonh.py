# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH with a lazy input embedding from a given step on (vocabulary goal of 2026-10-21; survey idea 6, owner popup).

MuonH trains the input embedding with plain Adam. A row whose token is absent from a batch has a zero gradient, but
Adam's first moment still moves it, in the direction of its last few updates, for many steps after the token was seen
(kexue.fm/archives/6869 and 9736: such updates of absent rows let rare embeddings overfit). ``LazyEmbedMuonHConfig``
builds the baseline MuonH optimizer unchanged and, from update ``lazy_from_step`` on, zeroes the update of every input
embedding row whose gradient in this step is exactly zero, i.e. whose token is not in the batch. Adam's moments still
decay as in the baseline (Su's form of the lazy update); only the step applied to absent rows is dropped.

Before ``lazy_from_step`` the wrapper changes nothing (the baseline bit for bit). The optimizer state keeps the
baseline's structure, so checkpoints and learning-rate logging are unchanged. Only the input embedding is lazy: the
AdamH head and the MuonH linears are untouched.
"""

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import optax

import haliax as hax
from levanter.optim.config import OptimizerConfig
from levanter.optim.muonh import MuonHConfig


def _embed(tree):
    return tree.embeddings.token_embeddings.weight


def lazy_rows(updates, grads, active):
    """``updates`` with the input-embedding rows zeroed where ``grads`` has an all-zero row and ``active`` holds."""
    g, u = _embed(grads), _embed(updates)
    if g is None or u is None:
        return updates
    embed = [a for a in g.axes if a.name != "vocab"]
    present = hax.any(g != 0, axis=embed)   # [vocab]: the token occurs in this batch
    keep = present | ~active
    return eqx.tree_at(_embed, updates, hax.where(keep, u, hax.zeros_like(u)).rearrange(u.axes))


@OptimizerConfig.register_subclass("muonH_lazy_embed")
@dataclass(frozen=True)
class LazyEmbedMuonHConfig(MuonHConfig):
    lazy_from_step: int = 0   # first update with a lazy input embedding

    def build(self, num_train_steps):
        if self.lazy_from_step < 0:
            raise ValueError(f"lazy_from_step={self.lazy_from_step}: must be >= 0")
        inner = super().build(num_train_steps)
        s0 = self.lazy_from_step

        def update_fn(updates, state, params=None, **extra):
            active = jnp.asarray(state.count) >= s0   # the index of this update
            new_updates, new_state = inner.update(updates, state, params, **extra)
            return lazy_rows(new_updates, updates, active), new_state

        return optax.GradientTransformationExtraArgs(inner.init, update_fn)
