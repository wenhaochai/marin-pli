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

``RareTargetMaskQwen3Config`` (Q10, owner 2026-10-11): from train step ``rm_from_step`` on (pass 5 of 8), positions whose
target is a rare token (ids rm_k .. V-257) get no loss. The remaining positions keep the baseline's gradient scale: the
masked weighted mean is rescaled by the (gradient-free) kept share of the weight, so the loss is
sum(w m l) / sum(w). The step comes from ``levanter.trainer.current_train_step()``; evaluation (no train step) is never
masked.

``ComposedResidualQwen3Config`` (Q10): the composed input table of ``ComposedVocabQwen3Config`` (cv_input; rows of tokens
below K are their own, a rare token's row is the mean of its K-token pieces' rows) plus, for each rare token, its own
residual row scaled by ``cv_res_scale`` (0.1): row_t = mean(pieces) + 0.1 * r_t. The residuals start at zero (step 0 is
the composed model) and sit in ``embeddings_res`` (MuonH "adam", no weight decay), so under Adam a residual moves at a
tenth of an ordinary row's rate.

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
from levanter.models.loss import next_token_loss_weight
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.trainer import current_train_step

from experiments.references.composed_vocab_qwen3 import ComposedVocabQwen3Config, ComposedVocabQwen3LMHeadModel, load_map

SMALL = "fo_small"   # axis of the trainable head rows (not "vocab", so no vocab sharding rule applies)
RARE = "fo_rare"     # axis of the fixed rare rows
RANK = "lr_rank"     # axis of the low-rank factors
RES = "cvr_rare"     # axis of the composed-input residual rows
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


@LmConfig.register_subclass("qwen3_rare_target_mask")
@dataclass(frozen=True)
class RareTargetMaskQwen3Config(Qwen3Config):
    rm_k: int = 8000          # targets with ids rm_k .. V-257 are "rare"
    rm_from_step: int = 0     # train steps >= this drop the loss of rare targets

    def __post_init__(self):
        super().__post_init__()
        if self.rm_k <= 0 or self.rm_from_step < 0:
            raise ValueError(f"rm_k={self.rm_k}, rm_from_step={self.rm_from_step}: need rm_k > 0 and rm_from_step >= 0")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return RareTargetMaskQwen3LMHeadModel


class RareTargetMaskQwen3LMHeadModel(Qwen3LMHeadModel):
    @classmethod
    def init(cls, Vocab: Axis, config: RareTargetMaskQwen3Config, *, key):  # type: ignore[override]
        # Qwen3LMHeadModel.init returns the base class, which would silently drop the mask: rebuild as this class
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)
        return cls(base.transformer, base.embeddings, base.lm_head)

    def compute_next_token_loss(self, example, *, key=None, reduction=hax.mean, reduction_axis=None, **kw):  # type: ignore[override]
        step = current_train_step()
        if step is None or reduction is not hax.mean or reduction_axis is not None:   # evaluation, or a caller-specific reduction
            return super().compute_next_token_loss(example, key=key, reduction=reduction, reduction_axis=reduction_axis, **kw)
        cfg: RareTargetMaskQwen3Config = self.config  # type: ignore[assignment]
        V = self.Vocab.size
        if not 0 < cfg.rm_k < V - N_SPECIAL:
            raise ValueError(f"rm_k={cfg.rm_k} must lie in (0, {V - N_SPECIAL})")
        tgt = hax.roll(example.tokens, -1, self.Pos).array   # the loss at position t predicts token t+1
        rare = (tgt >= cfg.rm_k) & (tgt < V - N_SPECIAL)
        keep = jnp.where(jnp.asarray(step) >= cfg.rm_from_step, ~rare, True)
        w = example.loss_weight if example.loss_weight is not None else hax.ones(example.tokens.axes, dtype=jnp.float32)
        w2 = w * hax.named(keep, example.tokens.axes).astype(w.dtype)
        loss = super().compute_next_token_loss(dataclasses.replace(example, loss_weight=w2), key=key, reduction=reduction, reduction_axis=None, **kw)
        # sum(w m l) / sum(w m) x sum(w m) / sum(w) = sum(w m l) / sum(w): the kept positions' gradients keep their scale
        wa, wb = next_token_loss_weight(self.Pos, w2), next_token_loss_weight(self.Pos, w)
        scale = jax.lax.stop_gradient(jnp.sum(wa.array.astype(jnp.float32)) / jnp.maximum(jnp.sum(wb.array.astype(jnp.float32)), 1e-9))
        return loss * scale


@LmConfig.register_subclass("qwen3_composed_residual")
@dataclass(frozen=True)
class ComposedResidualQwen3Config(ComposedVocabQwen3Config):
    cv_res_scale: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if not self.cv_input or self.cv_output:
            raise ValueError("the residual variant composes the input table only (cv_input=True, cv_output=False)")
        if not self.cv_res_scale > 0:
            raise ValueError(f"cv_res_scale={self.cv_res_scale}: must be positive")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return ComposedResidualQwen3LMHeadModel


class ComposedResidualQwen3LMHeadModel(ComposedVocabQwen3LMHeadModel):
    embeddings_res: NamedArray   # [cvr_rare, Embed]: the residual rows of tokens K .. V-257

    @classmethod
    def init(cls, Vocab: Axis, config: ComposedResidualQwen3Config, *, key):  # type: ignore[override]
        base = ComposedVocabQwen3LMHeadModel.init(Vocab, config, key=key)   # the composed model, bit for bit
        k = load_map(config.cv_map)[3] - N_SPECIAL
        res = hax.zeros((Axis(RES, Vocab.size - N_SPECIAL - k), config.Embed), dtype=base.embeddings.token_embeddings.weight.dtype)
        return cls(transformer=base.transformer, embeddings=base.embeddings, lm_head=base.lm_head, full_vocab=Vocab, embeddings_res=res)

    def _full(self) -> Qwen3LMHeadModel:
        plain = super()._full()
        cfg: ComposedResidualQwen3Config = self.config  # type: ignore[assignment]
        k = load_map(cfg.cv_map)[3] - N_SPECIAL
        te = plain.embeddings.token_embeddings
        w = te.weight.rearrange((self.full_vocab.name, cfg.Embed.name)).array
        r = self.embeddings_res.rearrange((RES, cfg.Embed.name)).array
        if r.shape[0] != self.full_vocab.size - N_SPECIAL - k:
            raise ValueError(f"{r.shape[0]} residual rows for ids {k} .. {self.full_vocab.size - N_SPECIAL - 1}")
        w = w.at[k:self.full_vocab.size - N_SPECIAL].add((cfg.cv_res_scale * r.astype(jnp.float32)).astype(w.dtype))
        te = dataclasses.replace(te, weight=hax.named(w, (self.full_vocab, cfg.Embed)))
        return Qwen3LMHeadModel(plain.transformer, dataclasses.replace(plain.embeddings, token_embeddings=te), plain.lm_head)
