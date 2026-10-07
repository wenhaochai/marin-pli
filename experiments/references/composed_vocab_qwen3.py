# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The muonh_qwen3 baseline with the input embedding or the output head of the 128K Marin tokenizer composed from the
rows of its K-token truncation (the vocabulary experiment's Q4 and Q5, blogs/vocab-overfitting.html).

The tokenization stays the Marin tokenizer's, so sequences, steps and the softmax over all 128,256 classes are the
baseline's. Only the table changes: a composed side keeps a table of the truncation's K ordinary rows plus the 256
special rows, and the full table row of token t is the mean of the rows of t's pieces under the K-token BPE
(``small_vocab_tokenizer.expansion``): a token below K is its own piece, so its row is unchanged; a rarer, longer token
(' repeated' -> ' repe' + 'ated') has no row of its own and shares the rows of its pieces with every other token that
uses them. The question is whether the rows of rare tokens are where repeated data is memorised.

``cv_input`` composes the input embedding (Q4), ``cv_output`` the output head (Q5). The small tables live where the full
ones did (``embeddings.token_embeddings`` and ``lm_head``), so MuonH's mask keeps them on Adam and AdamH, as for the
baseline. The full table is rebuilt every forward pass from the small one (a gather of every token's pieces and a
segment mean, about 0.3M rows), and the gradient flows back through the mean. Evaluation goes through the same two
entry points (``activations`` and ``get_lm_head``), so eval/paloma numbers are the composed model's.

The map is ``expand_full.npz`` (flat piece ids and per-token offsets) in the K-token tokenizer's directory, written by
``python -m experiments.references.small_vocab_tokenizer expansion-npz K OUT_DIR``.
"""

import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel

SMALL = "small_vocab"   # axis name of the small tables (not "vocab", so no vocab sharding rule applies to them)


@LmConfig.register_subclass("qwen3_composed_vocab")
@dataclass(frozen=True)
class ComposedVocabQwen3Config(Qwen3Config):
    cv_map: str = ""          # expand_full.npz of the K-token truncation
    cv_input: bool = False    # compose the input embedding (Q4)
    cv_output: bool = False   # compose the output head (Q5)

    def __post_init__(self):
        super().__post_init__()
        if not self.cv_map:
            raise ValueError("cv_map: the expand_full.npz path is required")
        if not (self.cv_input or self.cv_output):
            raise ValueError("compose the input embedding, the output head, or both")
        if self.tie_word_embeddings:
            raise ValueError("the composed tables assume an untied head")

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return ComposedVocabQwen3LMHeadModel


@lru_cache(maxsize=4)
def load_map(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(piece ids, full-token id of each piece, pieces per full token, small table rows) from the npz."""
    z = np.load(path)
    flat, offs = z["flat"].astype(np.int32), z["offs"].astype(np.int64)
    counts = np.diff(offs)
    if (counts <= 0).any():
        raise ValueError(f"{path}: a token without pieces")
    rows = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    return flat, rows, counts.astype(np.float32), int(flat.max()) + 1


def compose(small: NamedArray, Full: Axis, Embed: Axis, path: str) -> NamedArray:
    """[Full, Embed] table whose row t is the mean of the small rows of t's pieces."""
    flat, rows, counts, n_small = load_map(path)
    if Full.size != len(counts):
        raise ValueError(f"vocabulary axis has {Full.size} entries, the map {len(counts)}")
    w = small.rearrange((SMALL, Embed.name)).array
    if w.shape[0] != n_small:
        raise ValueError(f"small table has {w.shape[0]} rows, the map uses {n_small}")
    summed = jax.ops.segment_sum(w[jnp.asarray(flat)], jnp.asarray(rows), num_segments=Full.size, indices_are_sorted=True)
    return hax.named(summed / jnp.asarray(counts, dtype=summed.dtype)[:, None], (Full, Embed))


class ComposedVocabQwen3LMHeadModel(Qwen3LMHeadModel):
    full_vocab: Axis = eqx.field(static=True)

    @property
    def Vocab(self) -> Axis:  # type: ignore[override]
        return self.full_vocab

    @classmethod
    def init(cls, Vocab: Axis, config: ComposedVocabQwen3Config, *, key):  # type: ignore[override]
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)   # the baseline's trunk and initialisation
        _, _, counts, n_small = load_map(config.cv_map)
        if Vocab.size != len(counts):
            raise ValueError(f"vocabulary axis has {Vocab.size} entries, the map {len(counts)}")
        Small = Axis(SMALL, n_small)
        k_in, k_out = jrandom.split(jrandom.fold_in(key, 0xC0), 2)
        embeddings, lm_head = base.embeddings, base.lm_head
        if config.cv_input:
            embeddings = dataclasses.replace(embeddings, token_embeddings=hnn.Embedding.init(Small, config.Embed, key=k_in))
        if config.cv_output:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Small, key=k_out, use_bias=False, out_first=True)
        return cls(base.transformer, embeddings, lm_head, Vocab)

    def _full(self) -> Qwen3LMHeadModel:
        """The plain model with the full tables composed from the small ones."""
        cfg: ComposedVocabQwen3Config = self.config  # type: ignore[assignment]
        embeddings, lm_head = self.embeddings, self.lm_head
        if cfg.cv_input:
            te = embeddings.token_embeddings
            w = compose(te.weight, self.full_vocab, cfg.Embed, cfg.cv_map)
            embeddings = dataclasses.replace(embeddings, token_embeddings=dataclasses.replace(te, weight=w, Vocab=self.full_vocab))
        if cfg.cv_output:
            w = compose(lm_head.weight, self.full_vocab, cfg.Embed, cfg.cv_map)
            lm_head = dataclasses.replace(lm_head, weight=w, Out=self.full_vocab)
        return Qwen3LMHeadModel(self.transformer, embeddings, lm_head)

    def activations(self, input_ids, attn_mask=None, *, key=None, pos_ids: Optional[NamedArray] = None):  # type: ignore[override]
        if not self.config.cv_input:  # type: ignore[attr-defined]
            return super().activations(input_ids, attn_mask, key=key, pos_ids=pos_ids)
        return self._full().activations(input_ids, attn_mask, key=key, pos_ids=pos_ids)

    def get_lm_head(self) -> NamedArray:  # type: ignore[override]
        if not self.config.cv_output:  # type: ignore[attr-defined]
            return super().get_lm_head()
        return self._full().get_lm_head()

    def __call__(self, input_ids, attn_mask=None, pos_ids=None, *, key=None):  # type: ignore[override]
        return self._full()(input_ids, attn_mask, pos_ids, key=key)
