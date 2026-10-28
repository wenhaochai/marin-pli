"""A one-stage H-Net on bytes for the vocabulary page (owner 2026-10-28, after Cartesia's "From Tokens to Bytes I/II";
popup "再加一个 H-Net 实验"): does a byte model that learns its own chunks repeat data like the bytes (8-pass cost ~0) or
like a BPE tokenizer (128K: +0.08)?

Port of goombalab/hnet (MIT; Hwang, Wang and Gu 2025, arXiv 2507.07955) to Levanter, one stage, with Qwen3 decoder layers
in every network: the original's outer networks are Mamba-2 layers, which Levanter lacks, so encoder and decoder here are
full-attention Qwen3 layers within each document. The pieces follow hnet/models/hnet.py and hnet/modules/dc.py:

1. Byte embedding, then the encoder (``enc_layers`` layers and a final RMS norm) at every byte position.
2. Routing: q = W_q h, k = W_k h (both start as the identity); the boundary probability of byte t is
   p_t = (1 - cos(q_{t-1}, k_t)) / 2, forced to 1 at the first byte of every document; byte t opens a chunk when p_t > 0.5.
3. Chunking: the opening bytes' encoder states, in order, fill ``chunk_slots`` slots (JAX needs a static length; a row
   with more boundaries than slots keeps its first ``chunk_slots``, and the later bytes join the last chunk).
4. The main network (``num_layers`` layers and a final norm) on the chunks, causal within each document.
5. Dechunking: an exponential moving average over the chunks, z_j = P_j c_j + (1 - P_j) z_{j-1}, with P_j the opening
   byte's p clipped to [1e-4, 1 - 1e-4]; every byte takes the z of the chunk it belongs to.
6. Residual: y_t = z_t * STE(s_t) + R h_t, with s_t = p_t at a boundary and 1 - p_t elsewhere, STE(s) = 1 forward and the
   identity backward, and R a zero-initialised matrix applied in float32 (an Adam parameter, so MuonH leaves it alone).
7. The decoder (``dec_layers`` layers and a final norm) at every byte position, then the byte head.

Training adds the ratio loss alpha * N / (N - 1) * ((N - 1) F G + (1 - F)(1 - G)), F the share of boundaries and G the
mean of p over the batch's bytes, N = ``chunk_ratio`` (target bytes per chunk), alpha = ``ratio_loss_weight`` (0.03 in
the paper). Evaluation (no key or no train step) returns the byte states alone, so eval losses are the plain byte loss.
"""
import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from levanter.layers.attention import AttentionMask
from levanter.models.llama import LlamaEmbedding, LlamaTransformer
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.trainer import current_train_step


@LmConfig.register_subclass("qwen3_hnet")
@dataclass(frozen=True)
class HNetQwen3Config(Qwen3Config):
    """``num_layers`` is the main network's depth; ``max_seq_len`` the byte window."""

    enc_layers: int = 2
    dec_layers: int = 2
    chunk_ratio: float = 4.74      # N: target bytes per chunk (the Marin tokenizer writes fineweb-edu at 4.744 bytes per token)
    chunk_slots: int = 0           # main-network positions per row; 0: max_seq_len / chunk_ratio x slot_margin, rounded up to 128
    slot_margin: float = 1.25
    ratio_loss_weight: float = 0.03

    def __post_init__(self):
        super().__post_init__()
        if self.enc_layers < 1 or self.dec_layers < 1:
            raise ValueError(f"enc_layers={self.enc_layers}, dec_layers={self.dec_layers}: at least one each")
        if not self.chunk_ratio > 1:
            raise ValueError(f"chunk_ratio={self.chunk_ratio}: must exceed 1")
        if self.slots > self.max_seq_len:
            raise ValueError(f"{self.slots} chunk slots for a {self.max_seq_len}-byte window")

    @property
    def slots(self) -> int:
        return self.chunk_slots or 128 * math.ceil(self.max_seq_len / self.chunk_ratio * self.slot_margin / 128)

    def stack(self, n: int) -> "HNetQwen3Config":
        """This config with n layers, for the encoder or the decoder (the main network uses the config itself, so the
        model's config, read from its main network, is this one)."""
        return dataclasses.replace(self, num_layers=n)

    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return HNetQwen3LMHeadModel


def _ste(x: jax.Array) -> jax.Array:
    """1 in the forward pass, the identity in the backward pass (hnet.py STE)."""
    return x + jax.lax.stop_gradient(1.0 - x)


def _ema(p: jax.Array, c: jax.Array, first: jax.Array) -> jax.Array:
    """z_j = p_j c_j + (1 - p_j) z_{j-1} along axis -2, restarted (z_{j-1} = 0) where ``first`` marks a document's first
    chunk, as the original's packed scan does with seq_idx; an associative scan of affine maps."""
    a, b = jnp.where(first, 0.0, 1.0 - p)[..., None], p[..., None] * c

    def combine(left, right):
        (a1, b1), (a2, b2) = left, right
        return a1 * a2, a2 * b1 + b2

    return jax.lax.associative_scan(combine, (a, b), axis=-2)[1]


class HNetQwen3LMHeadModel(Qwen3LMHeadModel):
    """``transformer`` is the main network (so the trainer's per-layer settings reach it); ``embeddings`` and ``lm_head``
    are the byte embedding and head."""

    encoder: LlamaTransformer
    decoder: LlamaTransformer
    q_proj: hnn.Linear
    k_proj: hnn.Linear
    residual_proj: NamedArray

    @classmethod
    def init(cls, Vocab: Axis, config: HNetQwen3Config, *, key):  # type: ignore[override]
        k_t, k_emb, k_enc, k_dec = jrandom.split(key, 4)
        transformer = LlamaTransformer.init(config, key=k_t)
        encoder = LlamaTransformer.init(config.stack(config.enc_layers), key=k_enc)
        decoder = LlamaTransformer.init(config.stack(config.dec_layers), key=k_dec)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = None if config.tie_word_embeddings else hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        E = config.Embed
        Eo = E.alias("embed_out")
        eye = hax.named(jnp.eye(E.size, dtype=jnp.float32), (Eo, E))
        ident = lambda k: eqx.tree_at(lambda m: m.weight, hnn.Linear.init(In=E, Out=Eo, key=k, use_bias=False, out_first=True), eye)  # noqa: E731
        q_proj, k_proj = ident(k_t), ident(k_t)
        residual_proj = hax.zeros((Eo, E), dtype=jnp.float32)
        return cls(transformer, embeddings, lm_head, encoder, decoder, q_proj, k_proj, residual_proj)

    def chunked(self, h: NamedArray, attn_mask: Optional[AttentionMask | NamedArray]):
        """Routing and chunking of the encoder states h: (boundary probabilities p, boundaries b, chunk order, number
        of valid chunks, chunk states, chunk mask), the arrays raw, batch axes first, then position."""
        cfg: HNetQwen3Config = self.config  # type: ignore[assignment]
        Pos = h.resolve_axis(cfg.max_Pos.name)
        E = cfg.Embed
        L, M = Pos.size, cfg.slots
        batch_axes = tuple(ax for ax in h.axes if ax.name not in (Pos.name, E.name))
        hr = h.rearrange((*batch_axes, Pos, E)).array
        q = self.q_proj(h).rearrange((*batch_axes, Pos, self.q_proj.Out)).array.astype(jnp.float32)
        k = self.k_proj(h).rearrange((*batch_axes, Pos, self.k_proj.Out)).array.astype(jnp.float32)
        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
        cos = jnp.sum(q[..., :-1, :] * k[..., 1:, :], axis=-1)
        p = jnp.clip((1.0 - cos) / 2.0, 0.0, 1.0)
        p = jnp.concatenate([jnp.ones_like(p[..., :1]), p], axis=-1)   # (batch..., L)
        seg = None
        if isinstance(attn_mask, AttentionMask) and attn_mask.segment_ids is not None:
            seg = attn_mask.segment_ids[0].broadcast_to((*batch_axes, Pos)).rearrange((*batch_axes, Pos)).array
            start = jnp.concatenate([jnp.ones_like(seg[..., :1], dtype=bool), seg[..., 1:] != seg[..., :-1]], axis=-1)
            p = jnp.where(start, 1.0, p)
        b_raw = p > 0.5
        b = b_raw & (jnp.cumsum(b_raw, axis=-1) <= M)   # chunks past the slots join the last one
        t = jnp.arange(L)
        order = jnp.argsort(jnp.where(b, t, t + L), axis=-1)[..., :M]
        n = jnp.sum(b, axis=-1)
        hc = jnp.take_along_axis(hr, order[..., None], axis=-2)
        valid = jnp.arange(M) < n[..., None]
        segc = jnp.take_along_axis(seg, order, axis=-1) if seg is not None else None
        if segc is not None:
            segc = jnp.where(valid, segc, jnp.max(seg) + 1)   # empty slots: one segment of their own after the last document
        return p, b_raw, b, order, valid, hc, segc, batch_axes

    def activations(self, input_ids, attn_mask=None, *, key=None, pos_ids: Optional[NamedArray] = None):  # type: ignore[override]
        cfg: HNetQwen3Config = self.config  # type: ignore[assignment]
        k_enc, k_main, k_dec = jrandom.split(key, 3) if key is not None else (None, None, None)
        Pos = input_ids.resolve_axis(cfg.max_Pos.name)
        E = cfg.Embed
        x = self.embeddings.embed(input_ids)
        h = self.encoder(x, attn_mask=attn_mask, key=k_enc, pos_ids=pos_ids)
        p, b_raw, b, order, valid, hc, segc, batch_axes = self.chunked(h, attn_mask)
        Chunk = Pos.resize(cfg.slots)
        hc_n = hax.named(hc, (*batch_axes, Chunk, E))
        mask_c = AttentionMask.causal()
        if segc is not None:
            mask_c = mask_c.with_segment_ids(hax.named(segc, (*batch_axes, Chunk)))
        zc = self.transformer(hc_n, attn_mask=mask_c, key=k_main).rearrange((*batch_axes, Chunk, E)).array
        pc = jnp.clip(jnp.take_along_axis(p, order, axis=-1), 1e-4, 1 - 1e-4)
        first = jnp.arange(cfg.slots) == 0
        if segc is not None:
            first = first | jnp.concatenate([jnp.ones_like(segc[..., :1], dtype=bool), segc[..., 1:] != segc[..., :-1]], axis=-1)
        z = _ema(pc, zc.astype(jnp.float32), first)
        idx = jnp.cumsum(b, axis=-1) - 1
        out = jnp.take_along_axis(z, idx[..., None], axis=-2)   # (batch..., L, E)
        sel = jnp.where(b, p, 1.0 - p)
        hr = h.rearrange((*batch_axes, Pos, E)).array.astype(jnp.float32)
        res = jnp.einsum("...le,oe->...lo", hr, self.residual_proj.rearrange((self.q_proj.Out, E)).array)
        y = out * _ste(sel)[..., None] + res
        y = hax.named(y.astype(h.dtype), (*batch_axes, Pos, E))
        y = self.decoder(y, attn_mask=attn_mask, key=k_dec, pos_ids=pos_ids)
        if key is None or current_train_step() is None:
            return y
        N = cfg.chunk_ratio
        F = jnp.mean(b_raw.astype(jnp.float32))
        G = jnp.mean(p)
        aux = cfg.ratio_loss_weight * N / (N - 1) * ((N - 1) * F * G + (1 - F) * (1 - G))
        return y, aux
