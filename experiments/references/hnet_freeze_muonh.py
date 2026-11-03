# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH with the H-Net chunker frozen from a given update on (owner popup 2026-11-03, HNET_FREEZE_FROM_PASS).

The chunker of experiments/references/hnet_qwen3.py is the byte embedding, the encoder, q_proj and k_proj: everything
that decides the boundaries and the chunk vectors. From update ``freeze_from_step`` on (the inject_hyperparams count, as
freeze_rows_muonh.py reads it), these parameters get

1. a zero gradient before MuonH sees it, so the moments, the gradient clipping of each group (MuonH's matrices, the Adam
   group of the embedding and the norms) and AdamH's head come from the other parameters only;
2. a zero update after MuonH, so they stay exactly at their values from the end of step ``freeze_from_step - 1``.

A zero gradient alone is not enough: MuonH's Nesterov momentum keeps moving a matrix after its gradient stops, and
Adam's moments keep moving the embedding. The groups follow MuonHConfig.create_mask unchanged: the encoder, q_proj and
k_proj matrices are MuonH, the byte embedding and the encoder's norms Adam. Before ``freeze_from_step`` the wrapper changes
nothing (bit for bit), and the optimizer state keeps the baseline's structure.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from levanter.optim.config import OptimizerConfig
from levanter.optim.muonh import MuonHConfig

CHUNKER = ("embeddings", "encoder", "q_proj", "k_proj")   # HNetQwen3LMHeadModel fields


def zero_chunker(tree, frozen):
    """``tree`` (model-shaped: gradients or updates) with every array of the chunker zeroed where ``frozen`` (a traced
    bool)."""
    for name in CHUNKER:
        sub = getattr(tree, name)
        sub = jax.tree.map(lambda x: jnp.where(frozen, jnp.zeros_like(x), x), sub)
        tree = eqx.tree_at(lambda t, n=name: getattr(t, n), tree, sub, is_leaf=lambda x: x is None)
    return tree


@OptimizerConfig.register_subclass("muonH_freeze_chunker")
@dataclass(frozen=True)
class FreezeChunkerMuonHConfig(MuonHConfig):
    freeze_from_step: int = 0

    def build(self, num_train_steps):
        if self.freeze_from_step < 1:
            raise ValueError(f"freeze_from_step={self.freeze_from_step}: the chunker learns first, so 1 or later")
        inner = super().build(num_train_steps)
        s0 = self.freeze_from_step

        def update_fn(updates, state, params=None, **extra):
            frozen = jnp.asarray(state.count) >= s0   # the index of this update
            updates, new_state = inner.update(zero_chunker(updates, frozen), state, params, **extra)
            return zero_chunker(updates, frozen), new_state

        return optax.GradientTransformationExtraArgs(inner.init, update_fn)
