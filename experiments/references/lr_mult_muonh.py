# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH with its main learning rate multiplied from a given step on (vocabulary goal of 2026-10-21; survey idea 10,
owner popup "第 2 遍起提高学习率").

On MuonH's norm-preserving sphere a weight keeps roughly 2 / lr^2 steps of past directions (the survey's estimate from
kexue.fm/archives/11459): about 20,000 steps at the 300m peak lr 0.01, longer than the whole run and far longer than one
pass (1,430 steps) of an 8-pass run, so the weights can store every pass. ``LrMultMuonHConfig`` builds the baseline
MuonH optimizer with the main schedule (``learning_rate``: the MuonH linears and the AdamH head) multiplied by
``lr_mult`` from update ``lr_mult_from_step`` on; the cosine shape and its decay to zero stay. The Adam group (input
embedding, norms; ``adam_lr``) is unchanged. Before that update nothing changes (the baseline bit for bit).
"""

from dataclasses import dataclass

import jax.numpy as jnp

from levanter.optim.config import OptimizerConfig
from levanter.optim.muonh import MuonHConfig


@OptimizerConfig.register_subclass("muonH_lr_mult")
@dataclass(frozen=True)
class LrMultMuonHConfig(MuonHConfig):
    lr_mult: float = 1.0          # factor on the main learning rate
    lr_mult_from_step: int = 0    # first update with the factor

    def lr_scheduler(self, num_train_steps, override_lr=None):
        base = super().lr_scheduler(num_train_steps, override_lr=override_lr)
        if override_lr is not None:   # the Adam group's schedule (adam_lr): unchanged
            return base
        if self.lr_mult <= 0 or self.lr_mult_from_step < 0:
            raise ValueError(f"lr_mult={self.lr_mult}, lr_mult_from_step={self.lr_mult_from_step}: need a factor > 0 and a step >= 0")
        m, s0 = self.lr_mult, self.lr_mult_from_step

        def schedule(count):
            return base(count) * jnp.where(jnp.asarray(count) >= s0, m, 1.0)

        return schedule
