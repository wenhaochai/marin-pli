# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The residual designs compared by DepthBench (arXiv 2609.32534, appendix A, and its code), on the muonh_qwen3 layer,
for the per-layer contribution runs (blog: wenhaochai.com/blogs/per-layer-contribution.html, Figure 7).

``depth_arch`` picks how the two sublayers of every layer (attention, then MLP; F below, each with its own weights)
meet the residual stream. l is the layer index (1-based in lns's factor, 0-based in the sublayer index k = 2l + j of hc
and mhc, j = 0 attention, 1 MLP), L the number of layers, RN an RMSNorm with a learned scale:

    preln          h <- h + F(RN(h))
    sandwich       h <- h + RN_out(F(RN_in(h)))                   the baseline's design (hybrid_norm); this loop computes
                                                                  the baseline's Stacked transformer
    lns            h <- h + F(RN(h) / sqrt(l))                    LayerNorm Scaling; the same factor in both sublayers
    deepnorm       h <- LN(a h + F(h)),  a = (2L)^(1/4)           LN a mean-centring LayerNorm with a scale and no bias;
                                                                  v, o and the three MLP weights scaled by (8L)^(-1/4)
    keel           h <- RN_post(a h + F(RN_pre(h))),  a = 2L       the number of sublayers; the first layer is special:
                                                                  its attention is a plain Pre-LN update and its MLP
                                                                  drops the gain (RN_post(h + F(RN_pre(h))))
    hc             m = 4 residual streams H; per sublayer x = sum_s r_s H_s, H <- H A + w (x) F(RN(x)), with read r,
                   mixing A and write w each a static part plus a dynamic part s * tanh(RMS(H_s) W) per stream
                   (Zhu et al. 2024, dynamic HC; static init r = e_{k mod m}, A = I, w = 1, W = 0, s = 0.01);
                   embedding copied to every stream
    mhc            per sublayer one projection phi of the RMS-normalised concatenated streams (m x d -> m + m + m^2)
                   gives read, write and residual logits, each gain * projection + bias (three gains, init 0.01; phi
                   init 0); r = sigmoid, w = 2 sigmoid, A = Sinkhorn (Liger Kernel 0.8.0: a softmax over the inputs,
                   then column and row normalisation, 20 rounds ending on columns, eps 1e-6). Bias init: read +8 on
                   stream k mod m and -8 elsewhere, write 0, residual 0 on the diagonal and -8 off it. Routing
                   and mixing in float32
    hc, mhc        the attention output and MLP down projections are scaled by 1/sqrt(m) at init
    attnres        every sublayer's input is a softmax mix of the embedding and all earlier sublayer outputs,
                   sum_i softmax_i(q . (g * RMS(v_i))) v_i with a learned pseudo-query q (init 0, a uniform mean) and a
                   key gain g (init 1) per sublayer, and a final pair for the head
    attnres_block  the same mix over the embedding, the finished blocks' summed outputs (8 blocks) and the current block's
                   running sum
    moda           preln, with one softmax over the causal sequence keys and values and the same token's depth keys and
                   values: two per earlier layer, its attention's (key after the QK-norm, before RoPE) and its MLP
                   input's from extra key and value projections kv_k, kv_v (key through the attention's QK-norm). The
                   query is rotated.
                   The last layer's MLP entry is never read.

The state read out after layer k (the per-layer heads and the final norm see it): h for the residual designs and moda;
the sum of the streams for hc and mhc; for attnres the mix the next layer's attention reads (after the last layer, the
head's mix). Every design stores its layers stacked on the layer axis (Stacked), as the baseline does, so MuonH's
norm-preserving projection treats every weight the same way in every design (it also keeps the init scales of deepnorm,
hc and mhc for the whole run). Designs with a fixed-size state (preln ... mhc) scan over the stack. attnres and moda
carry every earlier source in buffers of fixed size (2L + 1 sources; 2L key/value slots) through a two-level scan whose
backward pass recomputes one group of layers at a time; attnres_block loops in Python over layer slices of the stack.

Optimizer (DepthArchMuonHConfig): the baseline's parameters keep the baseline's MuonH / AdamH / Adam; the new small
parameters (HC / mHC weights, attnres pseudo-queries and key gains) get their own Adam with eps 1e-8. The 130m recipe's
Adam eps is 1e-20, and at the symmetric hc / mhc init the true gradient of some of these weights is 0, leaving float
rounding noise whose square underflows: Adam then divides noise by 1e-20 and moves a weight by ~80 x the learning rate in
one step. moda's kv_k and kv_v are Linears and train with MuonH like the attention's k_proj and v_proj, one matrix each.
Recipe, not design: under MuonH (no weight decay anywhere; the new parameters at the recipe's Adam learning rate, no
warmup at 130m), where DepthBench trains every design with AdamW (weight decay 0.1, warmup 10%).
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split
from haliax.nn.scan import ScanCheckpointPolicy, Stacked

from levanter.layers.attention import Attention
from levanter.layers.attention_mask import materialize_mask
from levanter.models.llama import LlamaConfig, LlamaMlp
from levanter.optim.config import OptimizerConfig
from levanter.optim.muonh import MuonHConfig, scale_by_adamh, scale_with_muonh
from levanter.utils.jax_utils import leaf_key_paths

ARCHS = ("preln", "sandwich", "lns", "deepnorm", "keel", "hc", "mhc", "attnres", "attnres_block", "moda")
_POST = ("sandwich", "deepnorm", "keel")
SCANNED = ("preln", "sandwich", "lns", "deepnorm", "keel", "hc", "mhc")  # fixed-size state: one compiled layer
MHC_READ_BIAS, MHC_RES_OFF_DIAGONAL = 8.0, -8.0  # DepthBench's "0/-8 gap-8" init


def _rms(x: NamedArray, Embed: Axis, eps: float = 1e-6) -> NamedArray:
    """Parameter-free RMS normalisation over Embed, in float32, returned in x's dtype."""
    xf = x.astype(jnp.float32)
    return (xf * hax.rsqrt(hax.mean(xf * xf, axis=Embed) + eps)).astype(x.dtype)


class HyperParams(eqx.Module):
    """One sublayer's hyper-connection weights (hc): static read / mixing / write plus their dynamic projections."""

    read: NamedArray  # (stream,)  offset from the static read constant
    res: NamedArray  # (stream, stream_out)  offset from the static mixing constant
    write: NamedArray  # (stream_out,)  offset from the static write constant
    w_read: NamedArray  # (embed,)
    w_res: NamedArray  # (embed, stream_out)
    w_write: NamedArray  # (embed,)
    scale: NamedArray  # (hp_scale=2,): dynamic scales of read/mixing and of write

    @staticmethod
    def init(Embed: Axis, m: int) -> "HyperParams":
        S, So = Axis("stream", m), Axis("stream_out", m)
        z = hax.zeros
        return HyperParams(z(S), z((S, So)), z(So), z(Embed), z((Embed, So)), z(Embed), hax.full(Axis("hp_scale", 2), 0.01))


class MhcParams(eqx.Module):
    """One sublayer's mHC routing (DepthBench, Liger Kernel 0.8.0 mhc): one projection of the RMS-normalised concatenated
    streams to m read, m write and m x m residual logits, three gains, and the logits' biases stored as offsets from
    their init constants (added in float32, so bf16 compute keeps the small learned changes)."""

    phi: NamedArray  # (stream, embed, mhc)
    bias: NamedArray  # (mhc,)
    gain: NamedArray  # (hp_scale=3,): read, write, residual

    @staticmethod
    def init(Embed: Axis, m: int) -> "MhcParams":
        Mh = Axis("mhc", m * m + 2 * m)
        return MhcParams(hax.zeros((Axis("stream", m), Embed, Mh)), hax.zeros(Mh), hax.full(Axis("hp_scale", 3), 0.01))


def _hc_constants(m: int, sublayer):
    """hc's static read / mixing / write values at init, in float32; sublayer = 2 l + j (traced)."""
    S, So = Axis("stream", m), Axis("stream_out", m)
    hot = (jnp.arange(m) == jnp.mod(sublayer, m)).astype(jnp.float32)
    return hax.named(hot, S), hax.named(jnp.eye(m), (S, So)), hax.named(jnp.ones((m,)), So)


def _mhc_constants(m: int, sublayer) -> NamedArray:
    """mhc's bias constants (mhc,), float32: read +8 on stream sublayer mod m and -8 elsewhere; write 0; residual 0 on the
    diagonal and -8 off it, row-major (out, in)."""
    hot = (jnp.arange(m) == jnp.mod(sublayer, m)).astype(jnp.float32)
    read = MHC_READ_BIAS * (2.0 * hot - 1.0)
    res = MHC_RES_OFF_DIAGONAL * (1.0 - jnp.eye(m, dtype=jnp.float32)).reshape(-1)
    return hax.named(jnp.concatenate([read, jnp.zeros((m,), jnp.float32), res]), Axis("mhc", m * m + 2 * m))


def _sinkhorn_liger(logits: NamedArray, iters: int = 20, eps: float = 1e-6) -> NamedArray:
    """Liger Kernel 0.8.0's mHC Sinkhorn on logits (..., stream_out, stream), i.e. (out, in), in float32: a softmax over
    the inputs of every output, a column normalisation (over the outputs of every input), then iters - 1 rounds of row
    and column normalisation; eps keeps every division finite."""
    lg = logits.astype(jnp.float32)
    mat = hax.exp(lg - hax.max(lg, axis="stream"))
    mat = mat / hax.sum(mat, axis="stream") + eps
    mat = mat / (hax.sum(mat, axis="stream_out") + eps)
    for _ in range(iters - 1):
        mat = mat / (hax.sum(mat, axis="stream") + eps)
        mat = mat / (hax.sum(mat, axis="stream_out") + eps)
    return mat


def _qk_normed_qkv(attn: Attention, x: NamedArray, pos_ids):
    """Attention._compute_qkv (projections, QK-norm, RoPE), also returning the key before RoPE: MoDA's depth key."""
    q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    if attn.config.qk_norm is not None:
        q, k = cast(Any, attn.q_norm)(q), cast(Any, attn.k_norm)(k)
    k_depth = k
    if attn.rot_embs is not None:
        pos = pos_ids if pos_ids is not None else hax.arange(x.resolve_axis("position"))
        q, k = attn.rot_embs(q, pos).astype(q.dtype), attn.rot_embs(k, pos).astype(k.dtype)
    return q, k, v, k_depth


class DepthArchLayer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    arch: str = eqx.field(static=True)
    self_attn: Attention
    mlp: LlamaMlp
    ln_1: hnn.RmsNorm  # before the attention sublayer (keel: RN_pre; unused by deepnorm)
    ln_2: hnn.RmsNorm  # before the MLP sublayer
    post_1: Optional[Any] = None  # sandwich: on the branch output; deepnorm (LayerNorm), keel: after the add
    post_2: Optional[Any] = None
    hyper: Optional[tuple] = None  # hc: (HyperParams, HyperParams); mhc: (MhcParams, MhcParams), one per sublayer
    query: Optional[NamedArray] = None  # attnres*: (sub=2, embed) pseudo-queries of the two sublayers
    key_gain: Optional[NamedArray] = None  # attnres*: (sub=2, embed) key gains of the two sublayers
    kv_k: Optional[hnn.Linear] = None  # moda: the MLP input's depth key, Embed -> (kv_head, head_size)
    kv_v: Optional[hnn.Linear] = None  # moda: the MLP input's depth value

    @staticmethod
    def init(config: LlamaConfig, arch: str, *, key) -> "DepthArchLayer":
        k_attn, k_mlp = jrandom.split(key, 2)
        attn = Attention.init(config.attention_config(), key=k_attn)
        mlp = LlamaMlp.init(config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias)

        def scale(lin, b):
            return eqx.tree_at(lambda l: l.weight, lin, lin.weight * b)

        if arch == "deepnorm":
            b = (8 * config.num_layers) ** -0.25
            attn = eqx.tree_at(lambda a: (a.v_proj, a.o_proj), attn, (scale(attn.v_proj, b), scale(attn.o_proj, b)))
            mlp = eqx.tree_at(lambda f: (f.gate_proj, f.up_proj, f.down_proj), mlp, (scale(mlp.gate_proj, b), scale(mlp.up_proj, b), scale(mlp.down_proj, b)))
        hyper = None
        if arch in ("hc", "mhc"):
            m = cast(Any, config).hc_streams
            b = 1.0 / math.sqrt(m)
            attn = eqx.tree_at(lambda a: a.o_proj, attn, scale(attn.o_proj, b))
            mlp = eqx.tree_at(lambda f: f.down_proj, mlp, scale(mlp.down_proj, b))
            hyper = tuple((HyperParams if arch == "hc" else MhcParams).init(config.Embed, m) for _ in range(2))
        if arch == "deepnorm":
            post = tuple(hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_weight=True, use_bias=False) for _ in range(2))
        elif arch in _POST:
            post = (config.mk_LayerNorm(config.Embed), config.mk_LayerNorm(config.Embed))
        else:
            post = (None, None)
        attnres = arch.startswith("attnres")
        query = hax.zeros((Axis("sub", 2), config.Embed)) if attnres else None
        key_gain = hax.ones((Axis("sub", 2), config.Embed)) if attnres else None
        kv_k = kv_v = None
        if arch == "moda":
            acfg = config.attention_config()
            kk, kv = jrandom.split(jrandom.fold_in(key, 1), 2)
            kv_k = hnn.Linear.init(In=config.Embed, Out=(acfg.KVHeads, acfg.HeadSize), key=kk, use_bias=config.use_bias, out_first=True)
            kv_v = hnn.Linear.init(In=config.Embed, Out=(acfg.KVHeads, acfg.HeadSize), key=kv, use_bias=config.use_bias, out_first=True)
        return DepthArchLayer(config=config, arch=arch, self_attn=attn, mlp=mlp, ln_1=config.mk_LayerNorm(config.Embed), ln_2=config.mk_LayerNorm(config.Embed),
                              post_1=post[0], post_2=post[1], hyper=hyper, query=query, key_gain=key_gain, kv_k=kv_k, kv_v=kv_v)

    # ---- the two branches ------------------------------------------------------------------------------------------
    def branch(self, j: int, x: NamedArray, mask, *, key, pos_ids, layer=None) -> NamedArray:
        """Sublayer j's branch on its input x (j = 0 attention, 1 MLP), including the pre-norm where the design has one.
        layer: the 0-based layer index (traced, from the scan); lns needs it."""
        arch = self.arch
        if arch != "deepnorm":
            x = (self.ln_1 if j == 0 else self.ln_2)(x)
        if arch == "lns":
            x = x * jax.lax.rsqrt(jnp.asarray(layer.array if isinstance(layer, NamedArray) else layer, jnp.float32) + 1.0).astype(x.dtype)
        if j == 1:
            return self.mlp(x, key=key)
        if arch == "moda":
            raise ValueError("moda's attention reads the depth keys and values: use moda_step")
        return self.self_attn(x=x, mask=mask, key=key, pos_ids=pos_ids)

    def residual_step(self, h: NamedArray, mask, *, key, pos_ids, layer=None) -> NamedArray:
        L = self.config.num_layers
        keys = maybe_rng_split(key, 2)
        first = None
        if self.arch == "keel":
            first = jnp.asarray(layer.array if isinstance(layer, NamedArray) else layer) == 0
        for j in range(2):
            f = self.branch(j, h, mask, key=keys[j], pos_ids=pos_ids, layer=layer)
            post = self.post_1 if j == 0 else self.post_2
            if self.arch == "sandwich":
                h = h + cast(Any, post)(f)
            elif self.arch == "deepnorm":
                h = cast(Any, post)(h * ((2 * L) ** 0.25) + f)
            elif self.arch == "keel":
                # the first layer (KEEL sec. 3, DepthBench app. A): a plain Pre-LN attention update, an MLP without the gain
                if j == 0:
                    h = hax.where(first, h + f, cast(Any, post)(h * float(2 * L) + f))
                else:
                    h = cast(Any, post)(h * jnp.where(first, 1.0, float(2 * L)).astype(h.dtype) + f)
            else:
                h = h + f
        return h

    def hyper_step(self, H: NamedArray, mask, *, key, pos_ids, layer) -> NamedArray:
        """layer: the 0-based layer index (traced, from the scan)."""
        if self.arch == "mhc":
            return self._mhc_step(H, mask, key=key, pos_ids=pos_ids, layer=layer)
        Embed = self.config.Embed
        m = H.resolve_axis("stream").size
        li = layer.array if isinstance(layer, NamedArray) else layer
        keys = maybe_rng_split(key, 2)
        for j in range(2):
            p = cast(HyperParams, cast(tuple, self.hyper)[j])
            c_read, c_res, c_write = _hc_constants(m, 2 * li + j)
            f32 = lambda a: a.astype(jnp.float32)  # noqa: E731
            s_rr, s_w = f32(p.scale)["hp_scale", 0], f32(p.scale)["hp_scale", 1]
            Hn = f32(_rms(H, Embed))
            d_read = hax.dot(Hn, f32(p.w_read), axis=Embed)  # (..., stream)
            d_res = hax.dot(Hn, f32(p.w_res), axis=Embed)  # (..., stream, stream_out)
            d_write = hax.dot(Hn, f32(p.w_write), axis=Embed).rename({"stream": "stream_out"})
            read = c_read + f32(p.read) + s_rr * hax.tanh(d_read)
            res = c_res + f32(p.res) + s_rr * hax.tanh(d_res)
            write = c_write + f32(p.write) + s_w * hax.tanh(d_write)
            x = hax.dot(read.astype(H.dtype), H, axis="stream")
            f = self.branch(j, x, mask, key=keys[j], pos_ids=pos_ids)
            upd = write.astype(H.dtype) * f.broadcast_axis(write.resolve_axis("stream_out"))
            H = (hax.dot(res.astype(H.dtype), H, axis="stream") + upd).rename({"stream_out": "stream"})
        return H

    def _mhc_step(self, H: NamedArray, mask, *, key, pos_ids, layer) -> NamedArray:
        """mHC (DepthBench app. A; Liger Kernel 0.8.0 mhc) for both sublayers: x = sum_i r_i H_i, H_o <- sum_i A[o, i] H_i
        + w_o f, with the routing from one projection of the RMS-normalised concatenated streams, all in float32."""
        Embed = self.config.Embed
        S = H.resolve_axis("stream")
        m, So = S.size, Axis("stream_out", S.size)
        li = layer.array if isinstance(layer, NamedArray) else layer
        keys = maybe_rng_split(key, 2)
        f32 = lambda a: a.astype(jnp.float32)  # noqa: E731
        for j in range(2):
            p = cast(MhcParams, cast(tuple, self.hyper)[j])
            Hf = f32(H)
            inv = hax.rsqrt(hax.mean(Hf * Hf, axis=(S, Embed)) + 1e-6)  # RMS over the concatenated streams
            mix = hax.dot(H, p.phi.astype(H.dtype), axis=(S, Embed), preferred_element_type=jnp.float32) * inv  # (..., mhc)
            g = f32(p.gain)
            c = _mhc_constants(m, 2 * li + j) + f32(p.bias)
            pre = mix["mhc", 0:m] * g["hp_scale", 0] + c["mhc", 0:m]
            post = mix["mhc", m : 2 * m] * g["hp_scale", 1] + c["mhc", m : 2 * m]
            res = mix["mhc", 2 * m :] * g["hp_scale", 2] + c["mhc", 2 * m :]
            read = hax.nn.sigmoid(pre.rename({"mhc": "stream"}))
            write = 2.0 * hax.nn.sigmoid(post.rename({"mhc": "stream_out"}))
            A = _sinkhorn_liger(res.unflatten_axis("mhc", (So, S)))  # (..., stream_out, stream) = (out, in)
            x = hax.dot(read, Hf, axis=S)
            f = self.branch(j, x.astype(H.dtype), mask, key=keys[j], pos_ids=pos_ids)
            H = (hax.dot(A, Hf, axis=S) + write * f32(f).broadcast_axis(So)).rename({"stream_out": "stream"}).astype(H.dtype)
        return H

    # ---- moda ------------------------------------------------------------------------------------------------------
    def ffn_depth_kv(self, x: NamedArray):
        """The MLP input's depth key and value (x = ln_2(h)): kv_k and kv_v, the key through the attention's QK-norm, no RoPE."""
        k, v = cast(hnn.Linear, self.kv_k)(x), cast(hnn.Linear, self.kv_v)(x)
        if self.self_attn.config.qk_norm is not None:
            k = cast(Any, self.self_attn.k_norm)(k)
        return k, v

    def moda_step(self, h: NamedArray, mask, depth, *, key, pos_ids, chunk: int):
        """One Pre-LN MoDA layer. depth: the earlier slots as _moda_attention takes them. Returns h and this layer's two
        depth entries: its attention's (k, v) and its MLP input's (k, v)."""
        k2 = maybe_rng_split(key, 2)[1]
        o, ka, va = _moda_attention(self.self_attn, self.ln_1(h), mask, depth, pos_ids=pos_ids, chunk=chunk)
        h = h + o
        xf = self.ln_2(h)
        kf, vf = self.ffn_depth_kv(xf)
        return h + self.mlp(xf, key=k2), (ka, va), (kf, vf)


def _mix(query: NamedArray, values: list, keys: list) -> NamedArray:
    """sum_i softmax_i(query . keys_i) values_i, in float32 weights."""
    logits = [hax.dot(k.astype(jnp.float32), query.astype(jnp.float32), axis=query.axes[0]) for k in keys]
    Src = Axis("src", len(values))
    w = hax.nn.softmax(hax.stack(Src, logits), axis=Src)
    out = values[0] * w[Src, 0].astype(values[0].dtype)
    for i in range(1, len(values)):
        out = out + values[i] * w[Src, i].astype(values[i].dtype)
    return out


def _put(V: NamedArray, R: NamedArray, i, f: NamedArray, Embed: Axis):
    """Write source f into slot i of the buffer V and 1 / RMS(f) (float32, eps as _rms) into slot i of R."""
    ff = f.astype(jnp.float32)
    r = hax.rsqrt(hax.mean(ff * ff, axis=Embed) + 1e-6)
    return V.at["src", i].set(f.astype(V.dtype)), R.at["src", i].set(r)


def _mix_buf(query: NamedArray, V: NamedArray, R: NamedArray, n) -> NamedArray:
    """_mix over the first n sources of the buffer V (axis src; later slots masked out). RMS(V_i) = V_i R_i, so the
    logits are (query . V_i) R_i, accumulated in float32, and the weighted sum accumulates in float32."""
    Src = V.resolve_axis("src")
    logits = hax.dot(V, query.astype(V.dtype), axis=query.axes[0], preferred_element_type=jnp.float32) * R
    w = hax.nn.softmax(hax.where(hax.arange(Src) < n, logits, -1e30), axis=Src)
    return hax.dot(w.astype(V.dtype), V, axis=Src, preferred_element_type=jnp.float32).astype(V.dtype)


def _moda_attention(attn: Attention, x: NamedArray, mask, depth, *, pos_ids, chunk: int):
    """Attention over the causal sequence keys/values and the same token's depth keys/values, one softmax. depth: (dk, dv,
    valid), dk and dv on a "depth" axis and valid a boolean over it (None: all valid), or None (no earlier slot). The
    query is rotated, the depth keys are not (DepthBench and the MoDA code cache the key after the QK-norm, before
    RoPE). Returns the projected output and this layer's depth entry (key before RoPE, value)."""
    q, k, v, k_depth = _qk_normed_qkv(attn, x, pos_ids)  # q (.., position, kv_head, q_heads_per_group, head_size)
    cfg = attn.config
    scale = cfg.scaling_factor if cfg.scaling_factor is not None else 1.0 / math.sqrt(cfg.HeadSize.size)
    Pos = q.resolve_axis("position")
    KPos = Pos.alias("key_position")
    ks, vs = k.rename({"position": "key_position"}), v.rename({"position": "key_position"})
    dk, dv, valid = depth if depth is not None else (None, None, None)
    outs = []
    for start in range(0, Pos.size, chunk):
        sl = hax.dslice(start, min(chunk, Pos.size - start))
        m = materialize_mask(mask, Pos, KPos, q_slice=sl)

        def part(qc, ks, vs, dkc, dvc, m):
            s = hax.dot(qc.astype(jnp.float32), ks.astype(jnp.float32), axis="head_size") * scale  # (.., position, kv_head, qh, key_position)
            if m is not None:
                s = hax.where(m, s, -1e30)
            mx = hax.max(s, axis="key_position")
            if dkc is not None:
                sd = hax.dot(qc.astype(jnp.float32), dkc.astype(jnp.float32), axis="head_size") * scale  # (.., depth, position, ...)
                if valid is not None:
                    sd = hax.where(valid, sd, -1e30)
                mx = hax.maximum(mx, hax.max(sd, axis="depth"))
            ps = hax.exp(s - mx)
            z = hax.sum(ps, axis="key_position")
            o = hax.dot(ps, vs.astype(jnp.float32), axis="key_position")
            if dkc is not None:
                pd = hax.exp(sd - mx)
                z = z + hax.sum(pd, axis="depth")
                o = o + hax.dot(pd, dvc.astype(jnp.float32), axis="depth")
            return (o / z).astype(x.dtype)

        qc = q["position", sl]
        dkc = dk["position", sl] if dk is not None else None
        dvc = dv["position", sl] if dv is not None else None
        outs.append(eqx.filter_checkpoint(part)(qc, ks, vs, dkc, dvc, m))
    o = hax.concatenate("position", outs) if len(outs) > 1 else outs[0]
    o = o.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
    return attn.o_proj(o), k_depth, v


class DepthArchTransformer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    arch: str = eqx.field(static=True)
    layers: Stacked  # every design: layers stacked on the layer axis, like the baseline
    norm: hnn.RmsNorm
    final_query: Optional[NamedArray] = None  # attnres*: the head's mix
    final_key_gain: Optional[NamedArray] = None

    @staticmethod
    def init(config: LlamaConfig, arch: str, *, key) -> "DepthArchTransformer":
        if arch not in ARCHS:
            raise ValueError(f"depth_arch must be one of {ARCHS}, got {arch!r}")
        keys = jrandom.split(key, config.num_layers)
        blocks = [DepthArchLayer.init(config, arch, key=keys[i]) for i in range(config.num_layers)]
        attnres = arch.startswith("attnres")
        is_named = lambda x: isinstance(x, NamedArray)  # noqa: E731
        stacked = jax.tree_util.tree_map(lambda *xs: hax.stack(config.Layers, xs), *blocks, is_leaf=is_named)
        layers = Stacked(stacked, config.Layers, ScanCheckpointPolicy._mk(config.gradient_checkpointing))
        return DepthArchTransformer(config, arch, layers, config.mk_LayerNorm(config.Embed), hax.zeros(config.Embed) if attnres else None, hax.ones(config.Embed) if attnres else None)

    def layer(self, i: int) -> "DepthArchLayer":
        """Layer i, sliced out of the stack (the loop designs)."""
        return hax.tree_util.tree_map(lambda a: a[self.config.Layers.name, i], cast(Stacked, self.layers).stacked)

    def outputs(self, x: NamedArray, attn_mask, *, key=None, pos_ids=None) -> NamedArray:
        """The state read out after every layer, stacked on the layer axis (see the module docstring); the last entry
        goes to the final norm."""
        cfg, arch, L = self.config, self.arch, self.config.num_layers
        if arch in SCANNED:
            skeys = maybe_rng_split(key, L) if key is not None else None
            if arch in ("hc", "mhc"):
                m = cast(Any, cfg).hc_streams

                def hstep(layer, H, i, *, key):
                    H = layer.hyper_step(H, attn_mask, key=key, pos_ids=pos_ids, layer=i).rearrange(H.axes)  # the scan carry keeps its axis order
                    return H, hax.sum(H, axis="stream")

                _, outs = self.layers.scan_via(hstep)(hax.stack(Axis("stream", m), [x] * m), hax.arange(cfg.Layers), key=skeys)
                return outs

            def rstep(layer, h, i, *, key):
                h = layer.residual_step(h, attn_mask, key=key, pos_ids=pos_ids, layer=i)
                return h, h

            _, outs = self.layers.scan_via(rstep)(x, hax.arange(cfg.Layers), key=skeys)
            return outs
        if arch == "attnres":
            return self._attnres_scan(x, attn_mask, key=key, pos_ids=pos_ids)
        if arch == "moda":
            return self._moda_scan(x, attn_mask, key=key, pos_ids=pos_ids)
        return hax.stack(cfg.Layers, self._loop_outputs(x, attn_mask, key=key, pos_ids=pos_ids))

    # ---- attnres and moda as a two-level layer scan ------------------------------------------------------------------
    def _two_level_scan(self, fn, init, *xs, key=None):
        """Scan fn(layer, carry, *x, key=k) -> (carry, out) over the layer stack in two rolled levels: an outer scan over
        G groups of layers whose checkpointed body saves only its carry, and an inner scan over the L / G layers of one
        group, recomputed (each layer checkpointed) in the backward pass. A carry holding every earlier layer's outputs
        is then saved G + L / G times, not L times, and the backward pass recomputes one group at a time. Returns the
        final carry and the outputs stacked on the layer axis."""
        cfg, L = self.config, self.config.num_layers
        G = next(g for g in range(int(L**0.5), 0, -1) if L % g == 0)
        Outer, Inner = Axis("layer_group", G), cfg.Layers.resize(L // G)
        is_named = lambda a: isinstance(a, NamedArray)  # noqa: E731

        def split(a):
            return hax.unflatten_axis(a, cfg.Layers.name, (Outer, Inner)) if is_named(a) else a.reshape((G, L // G) + a.shape[1:])

        stacked = jax.tree.map(split, cast(Stacked, self.layers).stacked, is_leaf=is_named)
        gxs = jax.tree.map(split, xs, is_leaf=is_named)
        keys = split(maybe_rng_split(key, L)) if key is not None else None
        policy = ScanCheckpointPolicy(simple=True)

        def group(carry, layers, gx, gkeys):
            def one(c, layer, *x, key):
                return fn(layer, c, *x, key=key)

            return hax.scan(one, Inner, remat=policy)(carry, layers, *gx, key=gkeys)

        carry, ys = hax.scan(group, Outer, remat=policy)(init, stacked, gxs, keys)
        return carry, jax.tree.map(lambda a: hax.flatten_axes(a, (Outer, Inner), cfg.Layers.name), ys, is_leaf=is_named)

    def _final_query(self) -> NamedArray:
        return cast(NamedArray, self.final_query).astype(jnp.float32) * cast(NamedArray, self.final_key_gain).astype(jnp.float32)

    def _attnres_scan(self, x: NamedArray, attn_mask, *, key, pos_ids) -> NamedArray:
        """attnres: the carry holds the next sublayer's input and every source so far (the embedding, then each sublayer's
        output) in a buffer of 2L + 1 slots with 1 / RMS of each; layer i writes slots 2i + 1 and 2i + 2. The read-out
        after layer i is the mix the next layer's attention reads, so it is also that layer's input."""
        cfg, L = self.config, self.config.num_layers
        Embed, Sub = cfg.Embed, Axis("sub", 2)
        stacked = cast(DepthArchLayer, cast(Stacked, self.layers).stacked)
        V = hax.zeros((Axis("src", 2 * L + 1),) + x.axes, dtype=x.dtype)
        R = hax.zeros((Axis("src", 2 * L + 1),) + tuple(a for a in x.axes if a.name != Embed.name), dtype=jnp.float32)
        V, R = _put(V, R, 0, x, Embed)
        q_eff = cast(NamedArray, stacked.query).astype(jnp.float32) * cast(NamedArray, stacked.key_gain).astype(jnp.float32)  # (layer, sub, embed)
        nxt = hax.concatenate(cfg.Layers.name, [q_eff[Sub, 0][cfg.Layers.name, 1:], self._final_query().broadcast_axis(cfg.Layers.resize(1))])

        def step(layer, carry, i, q_next, *, key):
            h_in, V, R = carry
            n = i.array if isinstance(i, NamedArray) else i
            k1, k2 = maybe_rng_split(key, 2)
            V, R = _put(V, R, 2 * n + 1, layer.branch(0, h_in, attn_mask, key=k1, pos_ids=pos_ids), Embed)
            q1 = cast(NamedArray, layer.query)[Sub, 1].astype(jnp.float32) * cast(NamedArray, layer.key_gain)[Sub, 1].astype(jnp.float32)
            V, R = _put(V, R, 2 * n + 2, layer.branch(1, _mix_buf(q1, V, R, 2 * n + 2), attn_mask, key=k2, pos_ids=pos_ids), Embed)
            out = _mix_buf(q_next, V, R, 2 * n + 3)
            return (out, V, R), out

        _, outs = self._two_level_scan(step, (x, V, R), hax.arange(cfg.Layers), nxt, key=key)
        return outs

    def _moda_scan(self, x: NamedArray, attn_mask, *, key, pos_ids) -> NamedArray:
        """moda: the carry holds h and the depth keys and values in buffers of 2L slots (slot 2i: layer i's attention,
        2i + 1: its MLP input); layer i's attention reads slots < 2i."""
        cfg, L = self.config, self.config.num_layers
        acfg = cfg.attention_config()
        Depth = Axis("depth", 2 * L)
        buf = hax.zeros((Depth,) + tuple(a for a in x.axes if a.name != cfg.Embed.name) + (acfg.KVHeads, acfg.HeadSize), dtype=x.dtype)
        chunk = cast(Any, cfg).moda_chunk

        def step(layer, carry, i, *, key):
            h, dk, dv = carry
            n = i.array if isinstance(i, NamedArray) else i
            h, (ka, va), (kf, vf) = layer.moda_step(h, attn_mask, (dk, dv, hax.arange(Depth) < 2 * n), key=key, pos_ids=pos_ids, chunk=chunk)
            dk = dk.at["depth", 2 * n].set(ka.astype(dk.dtype)).at["depth", 2 * n + 1].set(kf.astype(dk.dtype))
            dv = dv.at["depth", 2 * n].set(va.astype(dv.dtype)).at["depth", 2 * n + 1].set(vf.astype(dv.dtype))
            return (h, dk, dv), h

        _, outs = self._two_level_scan(step, (x, buf, buf), hax.arange(cfg.Layers), key=key)
        return outs

    # ---- the Python loops: attnres_block, and the reference the scans are tested against ---------------------------
    def _loop_outputs(self, x: NamedArray, attn_mask, *, key=None, pos_ids=None) -> list:
        cfg, arch, L = self.config, self.arch, self.config.num_layers
        keys = maybe_rng_split(key, L) if key is not None else [None] * L
        if arch.startswith("attnres"):
            return self._attnres_outputs(x, attn_mask, keys, pos_ids)
        if arch == "moda":
            blocks = [self.layer(i) for i in range(L)]
            chunk = cast(Any, cfg).moda_chunk
            dks: list = []
            dvs: list = []
            outs = []
            for i, layer in enumerate(blocks):

                def step(layer, h, dks, dvs, k):
                    Dd = Axis("depth", len(dks))
                    depth = (hax.stack(Dd, dks), hax.stack(Dd, dvs), None) if dks else None
                    return layer.moda_step(h, attn_mask, depth, key=k, pos_ids=pos_ids, chunk=chunk)

                x, (ka, va), (kf, vf) = eqx.filter_checkpoint(step)(layer, x, dks, dvs, keys[i])
                dks, dvs = dks + [ka.astype(x.dtype), kf.astype(x.dtype)], dvs + [va.astype(x.dtype), vf.astype(x.dtype)]
                outs.append(x)
            return outs
        raise AssertionError(arch)

    def _attnres_outputs(self, x: NamedArray, attn_mask, keys, pos_ids) -> list:
        cfg, L = self.config, self.config.num_layers
        Embed, Sub = cfg.Embed, Axis("sub", 2)
        blocks = [self.layer(i) for i in range(L)]
        block = self.arch == "attnres_block"
        per_block = max(1, (2 * L) // cast(Any, cfg).attnres_blocks)  # sublayers per block
        values, rkeys = [x], [_rms(x, Embed)]  # full: every source; block: finished blocks (+ the running sum below)
        partial: Optional[NamedArray] = None
        n_in_block = 0
        outs = []

        def sources():
            if block and partial is not None:
                return values + [partial], rkeys + [_rms(partial, Embed)]
            return values, rkeys

        def query(i: int, j: int) -> NamedArray:
            lay = blocks[i]
            return cast(NamedArray, lay.query)[Sub, j].astype(jnp.float32) * cast(NamedArray, lay.key_gain)[Sub, j].astype(jnp.float32)

        for i, layer in enumerate(blocks):
            ks = maybe_rng_split(keys[i], 2)
            for j in range(2):
                vals, rk = sources()

                def sub(layer, q, vals, rk, k, j=j):
                    return layer.branch(j, _mix(q, vals, rk), attn_mask, key=k, pos_ids=pos_ids)

                f = eqx.filter_checkpoint(sub)(layer, query(i, j), vals, rk, ks[j])
                if block:
                    partial = f if partial is None else partial + f
                    n_in_block += 1
                    if n_in_block == per_block:
                        values, rkeys = values + [partial], rkeys + [_rms(partial, Embed)]
                        partial, n_in_block = None, 0
                else:
                    values, rkeys = values + [f], rkeys + [_rms(f, Embed)]
            vals, rk = sources()
            outs.append(_mix(query(i + 1, 0) if i + 1 < L else self._final_query(), vals, rk))
        return outs

    def __call__(self, x: NamedArray, attn_mask, *, key=None, pos_ids: NamedArray | None = None) -> NamedArray:
        return self.norm(self.outputs(x, attn_mask, key=key, pos_ids=pos_ids)[self.config.Layers.name, -1])


_NEW_PARAMS = ("hyper", "query", "key_gain")  # path fragments of the parameters the designs add


@OptimizerConfig.register_subclass("muonH_depth_arch")
@dataclass(frozen=True)
class DepthArchMuonHConfig(MuonHConfig):
    """MuonH exactly as the baseline for the baseline's parameters; the parameters a design adds (paths containing
    "hyper", "query" or "key_gain") get their own Adam with ``new_param_epsilon`` (see the module docstring for why)."""

    new_param_epsilon: float = 1e-8

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def clip():
                return [optax.clip_by_global_norm(self.max_grad_norm)] if self.max_grad_norm else []

            def adam(eps):
                return optax.chain(*clip(), optax.scale_by_adam(self.beta1, self.beta2, eps), optax.scale(-adam_lr))

            transformations = {
                "muonh": optax.chain(*clip(), scale_with_muonh(self.momentum, self.nesterov, self.backend_steps, self.muon_epsilon, learning_rate, self.coefficient_type)),
                "adamh": optax.chain(*clip(), scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate)),
                "adam": adam(self.epsilon),
                "adam_new": adam(self.new_param_epsilon),
            }
            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        base = super().create_mask(params)
        paths = leaf_key_paths(params, is_leaf=lambda x: isinstance(x, hnn.Linear))

        def relabel(label, path):
            if isinstance(path, NamedArray):  # a NamedArray leaf's path arrives wrapped in a NamedArray
                path = path.array
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            return "adam_new" if label == "adam" and any(f in path_str for f in _NEW_PARAMS) else label

        return jax.tree_util.tree_map(relabel, base, paths, is_leaf=lambda x: isinstance(x, (str, hnn.Linear)))
