# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Over-vocabulary (OV) Qwen3: the Over-Tokenized Transformer's OT configuration on the muonh_qwen3 baseline.

Huang et al. 2025, "Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling" (arXiv 2501.16975). Their OT
model is over-encoding (OE) plus over-decoding in its MTP-DS form (DeepSeek-V3's conditional multi-token prediction):
"combining OE-12.8M with MTP-DS, which we refer as OT-12.8M". We call it OV.

* Over-encoding: the input embedding is the baseline's 1-gram token embedding plus hashed 2-gram and 3-gram
  embeddings, OE(x) = E(x_t) + sum_{i=2..n} sum_{j=1..k} E_ij(h_i) W_ij, divided by 1 + k(n - 1). The i-gram index is
  h_i = (x_t + x_{t-1} V + ... + x_{t-i+1} V^{i-1}) mod m_ij (p = V as in the paper). n = 3, m = 12.8M, k chosen so
  that d_model / (n k) ~ 256 (k = 1 at 130m/300m/520m), each table d_model // (n k) wide, projected to d_model by W_ij.
  Tokens before the window start are 0 (the paper's zero padding); tokens of the previous document count as out of
  range too, because the baseline blocks cross-document attention. Each table gets its own modulus (m + 4 t: the
  paper's "m + 2" trick for distinct collisions, kept divisible by the 4 devices the rows are sharded over).
* MTP-DS, depth 1, weight 0.1: u_t = M [RMSNorm(h_t); RMSNorm(e_{t+1})] with M: 2d -> d, one more decoder layer,
  RMSNorm, and the shared lm_head predicts x_{t+2}; e_{t+1} is the (over-encoded) input embedding of the next token.
  Positions whose x_{t+1} or x_{t+2} lies past the window or in another document carry no MTP loss.

Everything else is the baseline's: data, steps, batch, schedule, and the optimizer, which labels the new parameters
by its own rules (tables are plain arrays -> Adam, like the token embedding; W_ij, M and the extra layer are Linears
-> MuonH, like every other Linear). Evaluation (``key=None``) is next-token loss on the main head with the full softmax.
``ss_candidates`` (VARIANT=ovss) puts both heads' training cross-entropy through the sampled softmax of
experiments.references.sampled_softmax_qwen3, each head with its own candidate sets.
"""

import math
from dataclasses import dataclass
from typing import Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import NamedArray
from levanter.layers.attention import AttentionMask
from levanter.metrics import Metric, ReductionType
from levanter.models.llama import LlamaDecoderLayer
from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.trainer import current_train_step

from experiments.references.sampled_softmax_qwen3 import sampled_next_token_loss

ROWS = "oe_rows"  # the tables' row axis: sharded over the data axis for both parameters and compute (launcher)


@LmConfig.register_subclass("qwen3_over_vocab")
@dataclass(frozen=True)
class OverVocabQwen3Config(Qwen3Config):
    oe_m: int = 12_800_000
    oe_n: int = 3
    oe_k: int = 0  # 0: derived so that hidden_dim / (n k) ~ 256
    mtp_weight: float = 0.1  # 0 turns MTP-DS off
    # sampled softmax for both heads (empty: full softmax); see experiments.references.sampled_softmax_qwen3
    ss_candidates: tuple[int, ...] = ()
    ss_stage_ends: tuple[int, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if self.oe_n < 2 or self.oe_m < 1 or self.oe_k < 0 or self.mtp_weight < 0:
            raise ValueError("oe_n >= 2, oe_m >= 1, oe_k >= 0 and mtp_weight >= 0 are required")
        if len(self.ss_candidates) != len(self.ss_stage_ends):
            raise ValueError("one stage end per candidate count")

    @property
    def k(self) -> int:
        return self.oe_k or max(1, round(self.hidden_dim / (self.oe_n * 256)))

    @property
    def table_dim(self) -> int:
        return self.hidden_dim // (self.oe_n * self.k)

    def moduli(self) -> list[tuple[int, int]]:
        """(order i, modulus) per table, orders 2..n with k tables each."""
        out, t = [], 0
        for i in range(2, self.oe_n + 1):
            for _ in range(self.k):
                out.append((i, self.oe_m + 4 * t)); t += 1
        return out

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return OverVocabQwen3LMHeadModel


def _mulmod(a: jax.Array, b: int, m: int) -> jax.Array:
    """(a * b) mod m in int32 for 0 <= a < 2^17, 0 <= b < m < 2^24: Horner over 6-bit chunks of b, every partial < 2^31."""
    assert m < (1 << 24) and 0 <= b < m
    r = jnp.zeros_like(a)
    for shift in (18, 12, 6, 0):
        r = (r * 64 + a * ((b >> shift) & 63)) % m
    return r


def ngram_index(prev: list[jax.Array], vocab_size: int, m: int) -> jax.Array:
    """(z_1 + z_2 V + ... + z_i V^{i-1}) mod m for token arrays prev = [z_1 = x_t, z_2 = x_{t-1}, ...], exact in int32."""
    idx = prev[0] % m
    for p, z in enumerate(prev[1:], start=1):
        idx = (idx + _mulmod(z, pow(vocab_size, p, m), m)) % m
    return idx


def shifted_tokens(tokens: NamedArray, seg: Optional[NamedArray], Pos: hax.Axis, n: int) -> list[NamedArray]:
    """[x_t, x_{t-1}, ..., x_{t-n+1}], with 0 for positions before the window or in an earlier document."""
    out = [tokens]
    pos = hax.arange(Pos)
    for s in range(1, n):
        z = hax.roll(tokens, s, Pos)
        ok = pos >= s
        if seg is not None:
            ok = ok & (hax.roll(seg, s, Pos) == seg)
        out.append(hax.where(ok, z, 0))
    return out


def _segment_ids(attn_mask) -> Optional[NamedArray]:
    return attn_mask.segment_ids[0] if isinstance(attn_mask, AttentionMask) and attn_mask.segment_ids is not None else None


class OverVocabQwen3LMHeadModel(Qwen3LMHeadModel):
    oe_tables: list  # NamedArray [oe_rows, oe_dim] per table (Adam group, like the token embedding)
    oe_proj: list  # hnn.Linear oe_dim -> embed per table (MuonH group, like every Linear)
    mtp_norm_h: Optional[hnn.RmsNorm]
    mtp_norm_e: Optional[hnn.RmsNorm]
    mtp_proj: Optional[hnn.Linear]  # mtp_in (2 d) -> embed
    mtp_layer: Optional[LlamaDecoderLayer]
    mtp_norm_out: Optional[hnn.RmsNorm]

    @classmethod
    def init(cls, Vocab, config: OverVocabQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)  # the baseline's parameters and initialisation, unchanged
        k_t, k_p, k_m, k_l = jrandom.split(jrandom.fold_in(key, 0x0E), 4)
        Dim = hax.Axis("oe_dim", config.table_dim)
        tables, projs = [], []
        for t, (_, m) in enumerate(config.moduli()):
            # Same initialisation as the token embedding table (hnn.Embedding.init).
            tables.append(hnn.Embedding.init(hax.Axis(ROWS, m), Dim, key=jrandom.fold_in(k_t, t)).weight)
            projs.append(hnn.Linear.init(In=Dim, Out=config.Embed, key=jrandom.fold_in(k_p, t), use_bias=False, out_first=True))
        mtp = config.mtp_weight > 0
        In2 = hax.Axis("mtp_in", 2 * config.hidden_dim)
        return cls(
            base.transformer, base.embeddings, base.lm_head, tables, projs,
            config.mk_LayerNorm(config.Embed) if mtp else None,
            config.mk_LayerNorm(config.Embed) if mtp else None,
            hnn.Linear.init(In=In2, Out=config.Embed, key=k_m, use_bias=False, out_first=True) if mtp else None,
            LlamaDecoderLayer.init(config, key=k_l) if mtp else None,
            config.mk_LayerNorm(config.Embed) if mtp else None,
        )

    def embed(self, input_ids: NamedArray, attn_mask) -> NamedArray:
        """Over-encoded input embedding (the baseline's embed() with the hashed n-gram terms added)."""
        cfg = cast(OverVocabQwen3Config, self.config)
        Pos = input_ids.resolve_axis(self.Pos.name)
        x = self.embeddings.token_embeddings(input_ids)
        prev = shifted_tokens(input_ids, _segment_ids(attn_mask), Pos, cfg.oe_n)
        for table, proj, (order, m) in zip(self.oe_tables, self.oe_proj, cfg.moduli()):
            idx = hax.named(ngram_index([z.array for z in prev[:order]], self.Vocab.size, m), input_ids.axes)
            x = x + proj(table.take(ROWS, idx))
        x = x / (1 + cfg.k * (cfg.oe_n - 1))
        if self.embeddings.norm is not None:
            x = self.embeddings.norm(x)
        return x

    def activations(self, input_ids, attn_mask=None, *, key=None, pos_ids=None):  # type: ignore[override]
        return self.transformer(self.embed(input_ids, attn_mask), attn_mask=attn_mask, key=key, pos_ids=pos_ids)

    def __call__(self, input_ids, attn_mask=None, pos_ids=None, *, key=None):  # type: ignore[override]
        # The baseline's __call__ embeds with self.embeddings directly; logits must see the over-encoded input too.
        return hax.dot(self.activations(input_ids, attn_mask, key=key, pos_ids=pos_ids), self.get_lm_head(), axis=self.Embed)

    def compute_next_token_loss(  # type: ignore[override]
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = cast(Optional[hax.ReductionFunction], hax.mean),
        reduction_axis: Optional[hax.AxisSelection] = None,
        logsumexp_weight: Optional[float] = None,
        loss_dtype: Optional[jnp.dtype] = jnp.float32,
        logit_soft_cap: Optional[float] = None,
    ):
        cfg = cast(OverVocabQwen3Config, self.config)
        kw = dict(reduction=reduction, reduction_axis=reduction_axis, logsumexp_weight=logsumexp_weight, logit_soft_cap=logit_soft_cap)
        if key is None:  # evaluation: main head, full softmax (LmHeadModel's loss over this model's activations)
            return super().compute_next_token_loss(example, key=None, loss_dtype=loss_dtype, **kw)
        k_main, k_mtp, k_s1, k_s2 = jrandom.split(key, 4)
        e = self.embed(example.tokens, example.attn_mask)
        h = self.transformer(e, attn_mask=example.attn_mask, key=k_main)
        sampled = bool(cfg.ss_candidates)
        step = current_train_step() if sampled else None
        if sampled and step is None:
            raise RuntimeError("the sampled softmax follows the train step, but current_train_step() is None")

        def head_loss(states, true_ids, weight, k_s):
            if sampled:
                return sampled_next_token_loss(self.Pos, self.Embed, self.Vocab, states, self.get_lm_head(), true_ids, loss_weight=weight,
                                               step=step, candidates=cfg.ss_candidates, stage_ends=cfg.ss_stage_ends, key=k_s, dtype=loss_dtype, **kw)
            return maybe_fused_next_token_loss(self.Pos, self.Embed, self.Vocab, states, self.get_lm_head(), true_ids, loss_weight=weight, dtype=loss_dtype, **kw), {}

        ntp, stats = head_loss(h, example.tokens, example.loss_weight, k_s1)
        loss = ntp
        metrics = {"ntp_loss": Metric.from_value(_scalar(ntp), ReductionType.MEAN)}
        if cfg.mtp_weight > 0:
            Pos = self.Pos
            e_next = hax.roll(e, -1, Pos)  # embedding of x_{t+1}
            u = hax.concatenate("mtp_in", [cast(hnn.RmsNorm, self.mtp_norm_h)(h).rename({self.Embed.name: "mtp_in"}),
                                           cast(hnn.RmsNorm, self.mtp_norm_e)(e_next).rename({self.Embed.name: "mtp_in"})])
            u = cast(hnn.Linear, self.mtp_proj)(u)
            u = cast(LlamaDecoderLayer, self.mtp_layer)(u, example.attn_mask, key=k_mtp)
            h2 = cast(hnn.RmsNorm, self.mtp_norm_out)(u)
            # h2_t predicts x_{t+2}: pass x_{t+1} as the "input ids" (the loss shifts once more) and drop positions whose
            # x_{t+1} or x_{t+2} is past the window or in another document, on top of the example's own weights.
            pos = hax.arange(Pos)
            ok = pos < Pos.size - 2
            seg = _segment_ids(example.attn_mask)
            if seg is not None:
                ok = ok & (hax.roll(seg, -1, Pos) == seg) & (hax.roll(seg, -2, Pos) == seg)
            w = example.loss_weight * hax.roll(example.loss_weight, -1, Pos) * ok.astype(example.loss_weight.dtype)
            mtp, mtp_stats = head_loss(h2, hax.roll(example.tokens, -1, Pos), w, k_s2)
            loss = loss + cfg.mtp_weight * mtp
            metrics["mtp_loss"] = Metric.from_value(_scalar(mtp), ReductionType.MEAN)
            stats.update({"mtp_" + name: v for name, v in mtp_stats.items()})
        metrics.update(stats)
        return loss, metrics


def _scalar(x):
    x = x.array if isinstance(x, NamedArray) else x
    return jax.lax.stop_gradient(jnp.mean(x)).astype(jnp.float32)
