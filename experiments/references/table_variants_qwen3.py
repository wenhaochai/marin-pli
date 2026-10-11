# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two variants of the muonh_qwen3 baseline's token tables for the vocabulary experiment (blogs/vocab-overfitting.html;
owner, 2026-10-11). Both keep the 128K Marin tokenizer, so sequences, steps and the softmax over all 128,256 classes are
the baseline's; only the tables change.

``FrozenOutQwen3Config`` (Q5, causal test of the output rows): the output rows of the tokens outside the K-token
truncation (ids K .. V-257, "rare" here) stay at their initial values. The trainable head keeps the K rows below K and
the 256 special rows (ids V-256 .. V-1) in ``lm_head`` (MuonH's AdamH group, as the baseline's head); the rare rows live
in a separate leaf ``fixed_out_rows`` that the forward pass wraps in ``stop_gradient``. Its gradient is exactly zero,
and MuonH labels it "adam" (no lm_head or Embedding in its path), whose update m / (sqrt(v) + eps) is then exactly zero,
so the rows never move. Both start as the baseline's own initial rows (same key, same draw), so the model at step 0 is
the baseline bit for bit. One difference stays: AdamH keeps a head's Frobenius norm at its initial value, so the
trainable rows now share the norm of their own initial rows instead of the whole table's.

``LowRankQwen3Config`` (Q9, the parameter-count control): the input and the output table are each the product of a
[V, r] and an [r, D] matrix, so at r = 48 the 300m model has about the 8K truncation's 115M parameters instead of 300M.
Every token keeps its own row (row t = A[t] @ B). The input factors sit where the embedding was and in
``embeddings_b`` (MuonH's "adam" group, as the baseline's embedding); the output factors sit in ``lm_head`` and
``lm_head_b`` (AdamH, as the baseline's head). The factors start as Gaussians whose product has the baseline table's
standard deviation (each factor std = sqrt(sigma / sqrt(r)), sigma measured on the baseline's own initial table).

Evaluation goes through the same entry points (``activations``, ``get_lm_head``), so eval/paloma numbers are these
models'. The full tables are rebuilt in every forward pass (a concatenation, or a [V, r] x [r, D] product in f32).
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel

SMALL = "fo_small"   # axis of the trainable head rows (not "vocab", so no vocab sharding rule applies)
RARE = "fo_rare"     # axis of the fixed rare rows
RANK = "lr_rank"     # axis of the low-rank factors
N_SPECIAL = 256


def _check_untied(cfg):
    if cfg.tie_word_embeddings:
        raise ValueError("the table variants assume untied input and output tables")


@LmConfig.register_subclass("qwen3_frozen_out")
@dataclass(frozen=True)
class FrozenOutQwen3Config(Qwen3Config):
    fo_k: int = 8000   # rows below this id (and the 256 specials) train; the rest stay at their initial values

    def __post_init__(self):
        super().__post_init__()
        _check_untied(self)
        if self.fo_k <= 0:
            raise ValueError(f"fo_k={self.fo_k}: must be positive")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return FrozenOutQwen3LMHeadModel


def frozen_split(V: int, k: int) -> tuple[int, int]:
    """(first rare id, first special id): ids [0, k) and [V - 256, V) train, [k, V - 256) stay fixed."""
    if not 0 < k < V - N_SPECIAL:
        raise ValueError(f"fo_k={k} must lie in (0, {V - N_SPECIAL}) for a vocabulary of {V}")
    return k, V - N_SPECIAL


class FrozenOutQwen3LMHeadModel(Qwen3LMHeadModel):
    fixed_out_rows: NamedArray
    full_vocab: Axis = eqx.field(static=True)

    @property
    def Vocab(self) -> Axis:  # type: ignore[override]
        return self.full_vocab

    @classmethod
    def init(cls, Vocab: Axis, config: FrozenOutQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)   # the baseline's trunk, embedding and initial head
        r0, s0 = frozen_split(Vocab.size, config.fo_k)
        w = base.lm_head.weight.rearrange((Vocab.name, config.Embed.name)).array
        small = jnp.concatenate([w[:r0], w[s0:]], axis=0)
        lm_head = dataclasses.replace(base.lm_head, weight=hax.named(small, (Axis(SMALL, small.shape[0]), config.Embed)),
                                      Out=Axis(SMALL, small.shape[0]))
        fixed = hax.named(w[r0:s0], (Axis(RARE, s0 - r0), config.Embed))
        return cls(base.transformer, base.embeddings, lm_head, fixed, Vocab)

    def _head(self) -> NamedArray:
        """[Vocab, Embed] head: trainable rows below K, the fixed rare rows (no gradient), trainable specials."""
        cfg: FrozenOutQwen3Config = self.config  # type: ignore[assignment]
        r0, s0 = frozen_split(self.full_vocab.size, cfg.fo_k)
        small = self.lm_head.weight.rearrange((SMALL, cfg.Embed.name)).array
        fixed = jax.lax.stop_gradient(self.fixed_out_rows.rearrange((RARE, cfg.Embed.name)).array).astype(small.dtype)
        if small.shape[0] != r0 + (self.full_vocab.size - s0) or fixed.shape[0] != s0 - r0:
            raise ValueError(f"head rows {small.shape[0]} + {fixed.shape[0]} do not split a vocabulary of {self.full_vocab.size} at {r0}, {s0}")
        full = jnp.concatenate([small[:r0], fixed, small[r0:]], axis=0)
        return hax.named(full, (self.full_vocab, cfg.Embed))

    def _plain(self) -> Qwen3LMHeadModel:
        lm_head = dataclasses.replace(self.lm_head, weight=self._head(), Out=self.full_vocab)
        return Qwen3LMHeadModel(self.transformer, self.embeddings, lm_head)

    def activations(self, input_ids, attn_mask=None, *, key=None, pos_ids: Optional[NamedArray] = None):  # type: ignore[override]
        return self._plain().activations(input_ids, attn_mask, key=key, pos_ids=pos_ids)

    def get_lm_head(self) -> NamedArray:  # type: ignore[override]
        return self._head()

    def __call__(self, input_ids, attn_mask=None, pos_ids=None, *, key=None):  # type: ignore[override]
        return self._plain()(input_ids, attn_mask, pos_ids, key=key)


@LmConfig.register_subclass("qwen3_low_rank")
@dataclass(frozen=True)
class LowRankQwen3Config(Qwen3Config):
    lr_rank: int = 48

    def __post_init__(self):
        super().__post_init__()
        _check_untied(self)
        if self.lr_rank <= 0:
            raise ValueError(f"lr_rank={self.lr_rank}: must be positive")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return LowRankQwen3LMHeadModel


class LowRankQwen3LMHeadModel(Qwen3LMHeadModel):
    embeddings_b: NamedArray   # [lr_rank, Embed]; the input table is embeddings.token_embeddings.weight @ embeddings_b
    lm_head_b: NamedArray      # [lr_rank, Embed]; the output table is lm_head.weight @ lm_head_b
    full_vocab: Axis = eqx.field(static=True)

    @property
    def Vocab(self) -> Axis:  # type: ignore[override]
        return self.full_vocab

    @classmethod
    def init(cls, Vocab: Axis, config: LowRankQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)   # the baseline's trunk; its tables only give the scale
        Rank, E = Axis(RANK, config.lr_rank), config.Embed
        k1, k2, k3, k4 = jax.random.split(jax.random.fold_in(key, 0x10A7), 4)

        def factors(table: NamedArray, ka, kb):
            w = table.rearrange((Vocab.name, E.name)).array
            sigma = jnp.std(w.astype(jnp.float32))
            s = jnp.sqrt(sigma / jnp.sqrt(config.lr_rank))   # std(A @ B) = sqrt(r) s^2 = sigma
            a = (jax.random.normal(ka, (Vocab.size, config.lr_rank), jnp.float32) * s).astype(w.dtype)
            b = (jax.random.normal(kb, (config.lr_rank, E.size), jnp.float32) * s).astype(w.dtype)
            return hax.named(a, (Vocab, Rank)), hax.named(b, (Rank, E))

        te = base.embeddings.token_embeddings
        a_in, b_in = factors(te.weight, k1, k2)
        a_out, b_out = factors(base.lm_head.weight, k3, k4)
        embeddings = dataclasses.replace(base.embeddings, token_embeddings=dataclasses.replace(te, weight=a_in))
        lm_head = dataclasses.replace(base.lm_head, weight=a_out.rearrange((Vocab.name, RANK)), In=Rank)
        return cls(base.transformer, embeddings, lm_head, b_in, b_out, Vocab)

    def _table(self, a: NamedArray, b: NamedArray) -> NamedArray:
        """[Vocab, Embed] = a @ b, accumulated in f32."""
        E = self.config.Embed
        aa = a.rearrange((self.full_vocab.name, RANK)).array
        bb = b.rearrange((RANK, E.name)).array
        full = jnp.matmul(aa.astype(jnp.float32), bb.astype(jnp.float32)).astype(aa.dtype)
        return hax.named(full, (self.full_vocab, E))

    def _plain(self) -> Qwen3LMHeadModel:
        te = self.embeddings.token_embeddings
        embeddings = dataclasses.replace(self.embeddings, token_embeddings=dataclasses.replace(te, weight=self._table(te.weight, self.embeddings_b)))
        lm_head = dataclasses.replace(self.lm_head, weight=self._table(self.lm_head.weight, self.lm_head_b), In=self.config.Embed)
        return Qwen3LMHeadModel(self.transformer, embeddings, lm_head)

    def activations(self, input_ids, attn_mask=None, *, key=None, pos_ids: Optional[NamedArray] = None):  # type: ignore[override]
        return self._plain().activations(input_ids, attn_mask, key=key, pos_ids=pos_ids)

    def get_lm_head(self) -> NamedArray:  # type: ignore[override]
        return self._table(self.lm_head.weight, self.lm_head_b)

    def __call__(self, input_ids, attn_mask=None, pos_ids=None, *, key=None):  # type: ignore[override]
        return self._plain()(input_ids, attn_mask, pos_ids, key=key)
