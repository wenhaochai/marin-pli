# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The residual designs compared by DepthBench (arXiv 2609.32534) on the muonh_qwen3 layer, for the per-layer
contribution runs (blog: wenhaochai.com/blogs/per-layer-contribution.html, Figure 6).

``depth_arch`` picks how the two sublayers of every layer (attention, then MLP; F below, each with its own weights)
meet the residual stream. l is the 1-based layer index, L the number of layers, LN an RMSNorm with a learned scale:

    preln          h <- h + F(LN(h))
    sandwich       h <- h + LN_out(F(LN_in(h)))                   the baseline's own design (hybrid_norm); this loop
                                                                  reproduces the baseline's Stacked transformer exactly
    lns            h <- h + F(LN(h) / sqrt(l))                    LayerNorm Scaling
    deepnorm       h <- LN(a h + F(h)),  a = (2L)^(1/4)            and v, o, MLP weights scaled by b = (8L)^(-1/4) at init
    keel           h <- LN_post(a h + F(LN_pre(h))),  a = 2L        the number of sublayers
    hc             m = 4 residual streams H; per sublayer x = sum_s r_s H_s, H <- H A + w (x) F(LN(x)), with read r,
                   mixing A and write w each a static part plus a dynamic part s * tanh(RMS(H) W) (Zhu et al. 2024, dynamic
                   HC; static init r = e_{l mod m}, A = I, w = 1, W = 0, s = 0.01); embedding copied to every stream
    mhc            hc with r = sigmoid(.), w = 2 sigmoid(.), A = Sinkhorn(exp(.)) (doubly stochastic, 20 iterations);
                   static logits r: +2 on stream l mod m, -2 elsewhere; A: +2 on the diagonal, -2 elsewhere; w: 0
    attnres        every sublayer's input is a softmax mix of the embedding and all earlier sublayer outputs,
                   sum_i softmax_i(q . RMS(v_i)) v_i with a learned pseudo-query q per sublayer (init 0, a uniform mean)
    attnres_block  the same mix over the embedding, the finished blocks' summed outputs (8 blocks) and the current block's
                   running sum
    moda           preln, with attention over the causal sequence keys and values plus the same token's keys and values
                   from every earlier layer, in one softmax (Mixture-of-Depths Attention)

The state read out after layer k (the per-layer heads and the final norm see it): h for the residual designs and moda;
the sum of the streams for hc and mhc; for attnres the mix the next layer's attention would read (after the last layer, a
mix with a final pseudo-query). The layers run as a Python loop (BlockSeq), each checkpointed, since attnres and moda carry
every earlier layer's outputs. New small parameters (norm scales, pseudo-queries, HC weights) are not Linear layers, so
MuonH leaves them to Adam, as it does the baseline's norms.
"""

import math
from typing import Any, Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split
from haliax.nn.scan import BlockSeq

from levanter.layers.attention import Attention
from levanter.layers.attention_mask import materialize_mask
from levanter.models.llama import LlamaConfig, LlamaMlp

ARCHS = ("preln", "sandwich", "lns", "deepnorm", "keel", "hc", "mhc", "attnres", "attnres_block", "moda")
_POST = ("sandwich", "deepnorm", "keel")


def _rms(x: NamedArray, Embed: Axis, eps: float = 1e-6) -> NamedArray:
    """Parameter-free RMS normalisation over Embed, in float32, returned in x's dtype."""
    xf = x.astype(jnp.float32)
    return (xf * hax.rsqrt(hax.mean(xf * xf, axis=Embed) + eps)).astype(x.dtype)


class HyperParams(eqx.Module):
    """One sublayer's hyper-connection weights: static read / mixing / write plus their dynamic projections."""

    read: NamedArray  # (stream,)
    res: NamedArray  # (stream, stream_out)
    write: NamedArray  # (stream_out,)
    w_read: NamedArray  # (embed,)
    w_res: NamedArray  # (embed, stream_out)
    w_write: NamedArray  # (embed,)
    scale: jax.Array  # (2,): dynamic scales of read/mixing and of write

    @staticmethod
    def init(Embed: Axis, m: int, layer: int, manifold: bool) -> "HyperParams":
        S, So = Axis("stream", m), Axis("stream_out", m)
        hot = jnp.zeros((m,)).at[layer % m].set(1.0)
        if manifold:
            read, res, write = 4.0 * hot - 2.0, 4.0 * jnp.eye(m) - 2.0, jnp.zeros((m,))
        else:
            read, res, write = hot, jnp.eye(m), jnp.ones((m,))
        z = jnp.zeros
        return HyperParams(
            hax.named(read, S), hax.named(res, (S, So)), hax.named(write, So),
            hax.named(z((Embed.size,)), Embed), hax.named(z((Embed.size, m)), (Embed, So)), hax.named(z((Embed.size,)), Embed),
            jnp.full((2,), 0.01),
        )


def _sinkhorn(logits: NamedArray, iters: int = 20) -> NamedArray:
    m = hax.exp(logits - hax.max(logits, axis=("stream", "stream_out")))
    for _ in range(iters):
        m = m / hax.sum(m, axis="stream_out")
        m = m / hax.sum(m, axis="stream")
    return m


class DepthArchLayer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    arch: str = eqx.field(static=True)
    index: int = eqx.field(static=True)  # 0-based
    self_attn: Attention
    mlp: LlamaMlp
    ln_1: hnn.RmsNorm  # before the attention sublayer (keel: LN_pre; unused by deepnorm)
    ln_2: hnn.RmsNorm  # before the MLP sublayer
    post_1: Optional[hnn.RmsNorm] = None  # sandwich: on the branch output; deepnorm, keel: after the add
    post_2: Optional[hnn.RmsNorm] = None
    hyper: Optional[tuple] = None  # hc, mhc: (HyperParams, HyperParams), one per sublayer
    query: Optional[NamedArray] = None  # attnres*: (sub=2, embed) pseudo-queries of the two sublayers

    @staticmethod
    def init(config: LlamaConfig, arch: str, index: int, *, key) -> "DepthArchLayer":
        k_attn, k_mlp = jrandom.split(key, 2)
        attn = Attention.init(config.attention_config(), key=k_attn)
        mlp = LlamaMlp.init(config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias)
        if arch == "deepnorm":
            b = (8 * config.num_layers) ** -0.25
            scale = lambda lin: eqx.tree_at(lambda l: l.weight, lin, lin.weight * b)  # noqa: E731
            attn = eqx.tree_at(lambda a: (a.v_proj, a.o_proj), attn, (scale(attn.v_proj), scale(attn.o_proj)))
            mlp = eqx.tree_at(lambda f: (f.gate_proj, f.up_proj, f.down_proj), mlp, (scale(mlp.gate_proj), scale(mlp.up_proj), scale(mlp.down_proj)))
        post = (config.mk_LayerNorm(config.Embed), config.mk_LayerNorm(config.Embed)) if arch in _POST else (None, None)
        hyper = None
        if arch in ("hc", "mhc"):
            m = cast(Any, config).hc_streams
            hyper = tuple(HyperParams.init(config.Embed, m, 2 * index + j, arch == "mhc") for j in range(2))
        query = hax.zeros((Axis("sub", 2), config.Embed)) if arch.startswith("attnres") else None
        return DepthArchLayer(config, arch, index, attn, mlp, config.mk_LayerNorm(config.Embed), config.mk_LayerNorm(config.Embed), post[0], post[1], hyper, query)

    # ---- the two branches ------------------------------------------------------------------------------------------
    def branch(self, j: int, x: NamedArray, mask, *, key, pos_ids, depth_kv=None) -> NamedArray:
        """Sublayer j's branch on its input x (j = 0 attention, 1 MLP), including the pre-norm where the design has one."""
        arch, l = self.arch, self.index + 1
        if arch != "deepnorm":
            x = (self.ln_1 if j == 0 else self.ln_2)(x)
        if arch == "lns":
            x = x * (1.0 / math.sqrt(l))
        if j == 1:
            return self.mlp(x, key=key)
        if arch == "moda":
            return _moda_attention(self.self_attn, x, mask, depth_kv, pos_ids=pos_ids, chunk=cast(Any, self.config).moda_chunk)
        return self.self_attn(x=x, mask=mask, key=key, pos_ids=pos_ids)

    def residual_step(self, h: NamedArray, mask, *, key, pos_ids) -> NamedArray:
        L = self.config.num_layers
        keys = maybe_rng_split(key, 2)
        for j in range(2):
            f = self.branch(j, h, mask, key=keys[j], pos_ids=pos_ids)
            post = self.post_1 if j == 0 else self.post_2
            if self.arch == "sandwich":
                h = h + cast(hnn.RmsNorm, post)(f)
            elif self.arch == "deepnorm":
                h = cast(hnn.RmsNorm, post)(h * ((2 * L) ** 0.25) + f)
            elif self.arch == "keel":
                h = cast(hnn.RmsNorm, post)(h * float(2 * L) + f)
            else:
                h = h + f
        return h

    def hyper_step(self, H: NamedArray, mask, *, key, pos_ids) -> NamedArray:
        Embed = self.config.Embed
        keys = maybe_rng_split(key, 2)
        for j in range(2):
            p = cast(HyperParams, cast(tuple, self.hyper)[j])
            Hn = _rms(H, Embed).astype(jnp.float32)
            d_read = hax.dot(Hn, p.w_read.astype(jnp.float32), axis=Embed)  # (..., stream)
            d_res = hax.dot(Hn, p.w_res.astype(jnp.float32), axis=Embed)  # (..., stream, stream_out)
            d_write = hax.dot(Hn, p.w_write.astype(jnp.float32), axis=Embed).rename({"stream": "stream_out"})
            if self.arch == "hc":
                read = p.read + p.scale[0] * hax.tanh(d_read)
                res = p.res + p.scale[0] * hax.tanh(d_res)
                write = p.write + p.scale[1] * hax.tanh(d_write)
            else:
                read = hax.nn.sigmoid(p.read + p.scale[0] * d_read)
                res = _sinkhorn(p.res + p.scale[0] * d_res)
                write = 2.0 * hax.nn.sigmoid(p.write + p.scale[1] * d_write)
            x = hax.dot(read.astype(H.dtype), H, axis="stream")
            f = self.branch(j, x, mask, key=keys[j], pos_ids=pos_ids)
            upd = write.astype(H.dtype) * f.broadcast_axis(write.resolve_axis("stream_out"))
            H = (hax.dot(res.astype(H.dtype), H, axis="stream") + upd).rename({"stream_out": "stream"})
        return H


def _mix(query: NamedArray, values: list, keys: list) -> NamedArray:
    """sum_i softmax_i(query . keys_i) values_i, in float32 weights."""
    logits = [hax.dot(k.astype(jnp.float32), query.astype(jnp.float32), axis=query.axes[0]) for k in keys]
    Src = Axis("src", len(values))
    w = hax.nn.softmax(hax.stack(Src, logits), axis=Src)
    out = values[0] * w[Src, 0].astype(values[0].dtype)
    for i in range(1, len(values)):
        out = out + values[i] * w[Src, i].astype(values[i].dtype)
    return out


def _moda_attention(attn: Attention, x: NamedArray, mask, depth_kv, *, pos_ids, chunk: int) -> NamedArray:
    """Attention over the causal sequence keys/values and the same token's keys/values from every earlier layer, one
    softmax. Returns the projected output; appends this layer's (k, v) to depth_kv (a list) for the layers above."""
    q, k, v = attn._compute_qkv(x, key=None, pos_ids=pos_ids)  # q (.., position, kv_head, q_heads_per_group, head_size)
    cfg = attn.config
    scale = cfg.scaling_factor if cfg.scaling_factor is not None else 1.0 / math.sqrt(cfg.HeadSize.size)
    Pos = q.resolve_axis("position")
    KPos = Pos.alias("key_position")
    ks, vs = k.rename({"position": "key_position"}), v.rename({"position": "key_position"})
    D = Axis("depth", len(depth_kv)) if depth_kv else None
    dk = hax.stack(D, [a for a, _ in depth_kv]) if D else None
    dv = hax.stack(D, [b for _, b in depth_kv]) if D else None
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
    depth_kv.append((k, v))
    o = o.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
    return attn.o_proj(o)


class DepthArchTransformer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    arch: str = eqx.field(static=True)
    layers: BlockSeq
    norm: hnn.RmsNorm
    final_query: Optional[NamedArray] = None  # attnres*: the mix read after the last layer

    @staticmethod
    def init(config: LlamaConfig, arch: str, *, key) -> "DepthArchTransformer":
        if arch not in ARCHS:
            raise ValueError(f"depth_arch must be one of {ARCHS}, got {arch!r}")
        keys = jrandom.split(key, config.num_layers)
        blocks = [DepthArchLayer.init(config, arch, i, key=keys[i]) for i in range(config.num_layers)]
        fq = hax.zeros(config.Embed) if arch.startswith("attnres") else None
        return DepthArchTransformer(config, arch, BlockSeq(blocks, config.Layers, cast(Any, None)), config.mk_LayerNorm(config.Embed), fq)

    def outputs(self, x: NamedArray, attn_mask, *, key=None, pos_ids=None) -> list:
        """The state read out after every layer (see the module docstring); the last one goes to the final norm."""
        cfg, arch, L = self.config, self.arch, self.config.num_layers
        Embed = cfg.Embed
        keys = maybe_rng_split(key, L) if key is not None else [None] * L
        blocks = cast(list, self.layers.blocks)
        outs = []
        if arch in ("hc", "mhc"):
            m = cast(Any, cfg).hc_streams
            H = hax.stack(Axis("stream", m), [x] * m)
            step = eqx.filter_checkpoint(lambda layer, H, k: layer.hyper_step(H, attn_mask, key=k, pos_ids=pos_ids))
            for i, layer in enumerate(blocks):
                H = step(layer, H, keys[i])
                outs.append(hax.sum(H, axis="stream"))
            return outs
        if arch.startswith("attnres"):
            return self._attnres_outputs(x, attn_mask, keys, pos_ids)
        if arch == "moda":
            depth_kv: list = []
            for i, layer in enumerate(blocks):

                def step(layer, h, kv_k, kv_v, k):
                    kv = list(zip(kv_k, kv_v))
                    k1, k2 = maybe_rng_split(k, 2)
                    h = h + layer.branch(0, h, attn_mask, key=k1, pos_ids=pos_ids, depth_kv=kv)
                    h = h + layer.branch(1, h, attn_mask, key=k2, pos_ids=pos_ids)
                    return h, kv[-1]

                x, new_kv = eqx.filter_checkpoint(step)(layer, x, [a for a, _ in depth_kv], [b for _, b in depth_kv], keys[i])
                depth_kv.append(new_kv)
                outs.append(x)
            return outs
        step = eqx.filter_checkpoint(lambda layer, h, k: layer.residual_step(h, attn_mask, key=k, pos_ids=pos_ids))
        for i, layer in enumerate(blocks):
            x = step(layer, x, keys[i])
            outs.append(x)
        return outs

    def _attnres_outputs(self, x: NamedArray, attn_mask, keys, pos_ids) -> list:
        cfg, L = self.config, self.config.num_layers
        Embed, Sub = cfg.Embed, Axis("sub", 2)
        blocks = cast(list, self.layers.blocks)
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

        def next_query(i: int, j: int) -> NamedArray:
            return cast(NamedArray, blocks[i].query)[Sub, j]

        for i, layer in enumerate(blocks):
            ks = maybe_rng_split(keys[i], 2)
            for j in range(2):
                vals, rk = sources()

                def sub(layer, q, vals, rk, k, j=j):
                    return layer.branch(j, _mix(q, vals, rk), attn_mask, key=k, pos_ids=pos_ids)

                f = eqx.filter_checkpoint(sub)(layer, next_query(i, j), vals, rk, ks[j])
                if block:
                    partial = f if partial is None else partial + f
                    n_in_block += 1
                    if n_in_block == per_block:
                        values, rkeys = values + [partial], rkeys + [_rms(partial, Embed)]
                        partial, n_in_block = None, 0
                else:
                    values, rkeys = values + [f], rkeys + [_rms(f, Embed)]
            vals, rk = sources()
            q = next_query(i + 1, 0) if i + 1 < L else cast(NamedArray, self.final_query)
            outs.append(_mix(q, vals, rk))
        return outs

    def __call__(self, x: NamedArray, attn_mask, *, key=None, pos_ids: NamedArray | None = None) -> NamedArray:
        return self.norm(self.outputs(x, attn_mask, key=key, pos_ids=pos_ids)[-1])
