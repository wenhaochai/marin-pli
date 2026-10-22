# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH with decoupled weight decay on the input embedding from a given step on (vocabulary goal of 2026-10-21; survey
idea 9, owner popup "embedding 权重衰减").

MuonH trains the input embedding with plain Adam and no weight decay (the launcher's weight_decay is never applied), so
the embedding's memory of past updates never fades. Viewing decoupled weight decay as an exponential moving average, a
row keeps about 1 / (lr * wd) steps of updates (kexue.fm/archives/11459); Kim et al. 2025 (arXiv 2509.14786) find a far
larger weight decay than usual optimal when data repeats. ``EmbedWDMuonHConfig`` builds the baseline MuonH optimizer
unchanged and, from update ``ewd_from_step`` on, adds -adam_lr * ewd * w to the update of the input embedding table only
(adam_lr is the Adam group's learning rate held in the optimizer state, the schedule's value one update behind). Before that update nothing changes (the
baseline bit for bit); the head, the linears and every other Adam parameter are untouched.
"""

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import optax

from levanter.optim.config import OptimizerConfig
from levanter.optim.muonh import MuonHConfig


def _embed(tree):
    return tree.embeddings.token_embeddings.weight


@OptimizerConfig.register_subclass("muonH_embed_wd")
@dataclass(frozen=True)
class EmbedWDMuonHConfig(MuonHConfig):
    ewd: float = 0.0            # decoupled weight decay of the input embedding
    ewd_from_step: int = 0      # first update that decays

    def build(self, num_train_steps):
        if self.ewd <= 0 or self.ewd_from_step < 0:
            raise ValueError(f"ewd={self.ewd}, ewd_from_step={self.ewd_from_step}: need ewd > 0 and a step >= 0")
        inner = super().build(num_train_steps)
        wd, s0 = self.ewd, self.ewd_from_step

        def update_fn(updates, state, params=None, **extra):
            active = jnp.asarray(state.count) >= s0   # the index of this update
            lr = state.hyperparams["adam_lr"]          # the Adam learning rate in the state, one update behind the schedule
            new_updates, new_state = inner.update(updates, state, params, **extra)
            u, w = _embed(new_updates), _embed(params)
            if u is None or w is None:
                return new_updates, new_state
            decay = jnp.where(active, lr * wd, 0.0).astype(w.dtype)
            return eqx.tree_at(_embed, new_updates, u - decay * w), new_state

        return optax.GradientTransformationExtraArgs(inner.init, update_fn)
