# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonH with the rare input rows decayed toward the mean of their pieces from a given step on (vocabulary goal of
2026-10-21; survey idea 7, owner popup "向片段均值衰减").

Repetition moves a rare token's input row away from the mean of its K-token pieces' rows, and cvd, which trains such a
row in place of the own one, undoes that drift (the project's Q10i/Q10n). This acts on the drift directly. From update
``dtp_from_step`` on, ``DecayToPiecesMuonHConfig`` adds -adam_lr * dtp * (w_t - c_t) to the input-embedding update of
each rare token t (ids K .. V-257), where c_t is the mean of t's pieces' current rows (``compose`` of
composed_vocab_qwen3 on the table itself, a constant here: the optimizer has no gradient). A row keeps about
1 / (adam_lr * dtp) updates of its own offset from c_t (kexue.fm/archives/11459), and decays toward c_t, not toward
zero, so its norm is not the target. adam_lr is the Adam group's learning rate in the optimizer state, one update behind
the schedule, as in embed_wd_muonh.py. Before that update nothing changes (the baseline bit for bit); rows below K, the
specials, the head and the linears are untouched.
"""

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

import haliax as hax
from haliax import Axis
from levanter.optim.config import OptimizerConfig
from levanter.optim.muonh import MuonHConfig

from experiments.references.composed_vocab_qwen3 import SMALL, compose, load_map, own_rows

N_SPECIAL = 256


def _embed(tree):
    return tree.embeddings.token_embeddings.weight


def pull_to_pieces(w, path: str):
    """[vocab, embed] array: w_t - mean of t's pieces' rows for the rare tokens (ids K .. V-257), zero elsewhere."""
    Vocab, Embed = w.resolve_axis("vocab"), [a for a in w.axes if a.name != "vocab"][0]
    W = w.rearrange((Vocab, Embed))
    own = own_rows(path, Vocab.size)
    small = hax.named(W.array[jnp.asarray(own)], (Axis(SMALL, len(own)), Embed))
    comp = compose(small, Vocab, Embed, path).array.astype(jnp.float32)
    k0 = load_map(path)[3] - N_SPECIAL
    ids = np.arange(Vocab.size)
    rare = jnp.asarray((ids >= k0) & (ids < Vocab.size - N_SPECIAL))[:, None]
    diff = jnp.where(rare, W.array.astype(jnp.float32) - comp, 0.0)
    return hax.named(diff, (Vocab, Embed)).rearrange(w.axes)


@OptimizerConfig.register_subclass("muonH_decay_to_pieces")
@dataclass(frozen=True)
class DecayToPiecesMuonHConfig(MuonHConfig):
    dtp_map: str = ""         # expand_full.npz of the K-token truncation
    dtp: float = 0.0          # decay rate toward the pieces' mean
    dtp_from_step: int = 0    # first update that decays

    def build(self, num_train_steps):
        if not self.dtp_map or self.dtp <= 0 or self.dtp_from_step < 0:
            raise ValueError(f"dtp_map={self.dtp_map!r}, dtp={self.dtp}, dtp_from_step={self.dtp_from_step}: need a map, dtp > 0, a step >= 0")
        inner = super().build(num_train_steps)
        rate, s0, path = self.dtp, self.dtp_from_step, self.dtp_map

        def update_fn(updates, state, params=None, **extra):
            active = jnp.asarray(state.count) >= s0   # the index of this update
            lr = state.hyperparams["adam_lr"]
            new_updates, new_state = inner.update(updates, state, params, **extra)
            u, w = _embed(new_updates), _embed(params)
            if u is None or w is None:
                return new_updates, new_state
            pull = pull_to_pieces(w, path)
            step = jnp.where(active, lr * rate, 0.0).astype(jnp.float32)
            return eqx.tree_at(_embed, new_updates, (u - step * pull).astype(u.dtype)), new_state

        return optax.GradientTransformationExtraArgs(inner.init, update_fn)
