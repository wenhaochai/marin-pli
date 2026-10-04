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
                   HC; static init r = e_{(2l+j) mod m}, A = I, w = 1, W = 0, s = 0.01); embedding copied to every stream
    mhc            hc with r = sigmoid(.), w = 2 sigmoid(.), A = Sinkhorn(exp(.)) (doubly stochastic, 20 log-domain
                   iterations); static logits r: +2 on stream (2l+j) mod m, -2 elsewhere; A: +2 on the diagonal, -2
                   elsewhere; w: 0
                   The static parts are stored as offsets from these constants (init 0), so bf16 compute keeps the small
                   learned changes, and the constants are added in float32.
    attnres        every sublayer's input is a softmax mix of the embedding and all earlier sublayer outputs,
                   sum_i softmax_i(q . RMS(v_i)) v_i with a learned pseudo-query q per sublayer (init 0, a uniform mean)
    attnres_block  the same mix over the embedding, the finished blocks' summed outputs (8 blocks) and the current block's
                   running sum
    moda           preln, with attention over the causal sequence keys and values plus the same token's keys and values
                   from every earlier layer, in one softmax (Mixture-of-Depths Attention)

The state read out after layer k (the per-layer heads and the final norm see it): h for the residual designs and moda;
the sum of the streams for hc and mhc; for attnres the mix the next layer's attention would read (after the last layer, a
mix with a final pseudo-query). Every design stores its layers stacked on the layer axis (Stacked), as the baseline
does, so MuonH's norm-preserving projection treats every weight the same way in every design. Designs with a fixed-size
state (preln ... mhc) scan over the stack (one compiled layer); attnres and moda carry every earlier layer's outputs, so
they loop in Python over layer slices of the stack, each layer checkpointed.

Optimizer (DepthArchMuonHConfig): the baseline's parameters keep the baseline's MuonH / AdamH / Adam; the new small
parameters (HC weights, attnres pseudo-queries) get their own Adam with eps 1e-8. The 130m recipe's Adam eps is 1e-20, and
at the symmetric hc / mhc init the true gradient of some of these weights is 0, leaving float rounding noise whose
square underflows: Adam then divides noise by 1e-20 and moves a weight by ~80 x the learning rate in one step.
"""

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


def _rms(x: NamedArray, Embed: Axis, eps: float = 1e-6) -> NamedArray:
    """Parameter-free RMS normalisation over Embed, in float32, returned in x's dtype."""
    xf = x.astype(jnp.float32)
    return (xf * hax.rsqrt(hax.mean(xf * xf, axis=Embed) + eps)).astype(x.dtype)


class HyperParams(eqx.Module):
    """One sublayer's hyper-connection weights: static read / mixing / write plus their dynamic projections."""

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


def _hyper_constants(m: int, sublayer, manifold: bool):
    """The static read / mixing / write values at init, in float32; sublayer = 2 l + j (traced)."""
    S, So = Axis("stream", m), Axis("stream_out", m)
    hot = (jnp.arange(m) == jnp.mod(sublayer, m)).astype(jnp.float32)
    if manifold:
        return hax.named(4.0 * hot - 2.0, S), hax.named(4.0 * jnp.eye(m) - 2.0, (S, So)), hax.named(jnp.zeros((m,)), So)
    return hax.named(hot, S), hax.named(jnp.eye(m), (S, So)), hax.named(jnp.ones((m,)), So)


def _sinkhorn(logits: NamedArray, iters: int = 20) -> NamedArray:
    """Sinkhorn-Knopp of exp(logits) over (stream, stream_out), in the log domain (no row can underflow to 0/0)."""
    lg = logits.astype(jnp.float32)
    for _ in range(iters):
        lg = lg - hax.nn.logsumexp(lg, axis="stream_out")
        lg = lg - hax.nn.logsumexp(lg, axis="stream")
    return hax.exp(lg)


class DepthArchLayer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    arch: str = eqx.field(static=True)
    self_attn: Attention
    mlp: LlamaMlp
    ln_1: hnn.RmsNorm  # before the attention sublayer (keel: LN_pre; unused by deepnorm)
    ln_2: hnn.RmsNorm  # before the MLP sublayer
    post_1: Optional[hnn.RmsNorm] = None  # sandwich: on the branch output; deepnorm, keel: after the add
    post_2: Optional[hnn.RmsNorm] = None
    hyper: Optional[tuple] = None  # hc, mhc: (HyperParams, HyperParams), one per sublayer
    query: Optional[NamedArray] = None  # attnres*: (sub=2, embed) pseudo-queries of the two sublayers

    @staticmethod
    def init(config: LlamaConfig, arch: str, *, key) -> "DepthArchLayer":
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
            hyper = tuple(HyperParams.init(config.Embed, m) for _ in range(2))
        query = hax.zeros((Axis("sub", 2), config.Embed)) if arch.startswith("attnres") else None
        return DepthArchLayer(config, arch, attn, mlp, config.mk_LayerNorm(config.Embed), config.mk_LayerNorm(config.Embed), post[0], post[1], hyper, query)

    # ---- the two branches ------------------------------------------------------------------------------------------
    def branch(self, j: int, x: NamedArray, mask, *, key, pos_ids, depth_kv=None, layer=None) -> NamedArray:
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
            return _moda_attention(self.self_attn, x, mask, depth_kv, pos_ids=pos_ids, chunk=cast(Any, self.config).moda_chunk)
        return self.self_attn(x=x, mask=mask, key=key, pos_ids=pos_ids)

    def residual_step(self, h: NamedArray, mask, *, key, pos_ids, layer=None) -> NamedArray:
        L = self.config.num_layers
        keys = maybe_rng_split(key, 2)
        for j in range(2):
            f = self.branch(j, h, mask, key=keys[j], pos_ids=pos_ids, layer=layer)
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

    def hyper_step(self, H: NamedArray, mask, *, key, pos_ids, layer) -> NamedArray:
        """layer: the 0-based layer index (traced, from the scan)."""
        Embed = self.config.Embed
        m = H.resolve_axis("stream").size
        li = layer.array if isinstance(layer, NamedArray) else layer
        keys = maybe_rng_split(key, 2)
        for j in range(2):
            p = cast(HyperParams, cast(tuple, self.hyper)[j])
            c_read, c_res, c_write = _hyper_constants(m, 2 * li + j, self.arch == "mhc")
            f32 = lambda a: a.astype(jnp.float32)  # noqa: E731
            s_rr, s_w = f32(p.scale)["hp_scale", 0], f32(p.scale)["hp_scale", 1]
            Hn = f32(_rms(H, Embed))
            d_read = hax.dot(Hn, f32(p.w_read), axis=Embed)  # (..., stream)
            d_res = hax.dot(Hn, f32(p.w_res), axis=Embed)  # (..., stream, stream_out)
            d_write = hax.dot(Hn, f32(p.w_write), axis=Embed).rename({"stream": "stream_out"})
            if self.arch == "hc":
                read = c_read + f32(p.read) + s_rr * hax.tanh(d_read)
                res = c_res + f32(p.res) + s_rr * hax.tanh(d_res)
                write = c_write + f32(p.write) + s_w * hax.tanh(d_write)
            else:
                read = hax.nn.sigmoid(c_read + f32(p.read) + s_rr * d_read)
                res = _sinkhorn(c_res + f32(p.res) + s_rr * d_res)
                write = 2.0 * hax.nn.sigmoid(c_write + f32(p.write) + s_w * d_write)
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
    layers: Stacked  # every design: layers stacked on the layer axis, like the baseline
    norm: hnn.RmsNorm
    final_query: Optional[NamedArray] = None  # attnres*: the mix read after the last layer

    @staticmethod
    def init(config: LlamaConfig, arch: str, *, key) -> "DepthArchTransformer":
        if arch not in ARCHS:
            raise ValueError(f"depth_arch must be one of {ARCHS}, got {arch!r}")
        keys = jrandom.split(key, config.num_layers)
        blocks = [DepthArchLayer.init(config, arch, key=keys[i]) for i in range(config.num_layers)]
        fq = hax.zeros(config.Embed) if arch.startswith("attnres") else None
        is_named = lambda x: isinstance(x, NamedArray)  # noqa: E731
        stacked = jax.tree_util.tree_map(lambda *xs: hax.stack(config.Layers, xs), *blocks, is_leaf=is_named)
        layers = Stacked(stacked, config.Layers, ScanCheckpointPolicy._mk(config.gradient_checkpointing))
        return DepthArchTransformer(config, arch, layers, config.mk_LayerNorm(config.Embed), fq)

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
        return hax.stack(cfg.Layers, self._loop_outputs(x, attn_mask, key=key, pos_ids=pos_ids))

    def _loop_outputs(self, x: NamedArray, attn_mask, *, key=None, pos_ids=None) -> list:
        cfg, arch, L = self.config, self.arch, self.config.num_layers
        keys = maybe_rng_split(key, L) if key is not None else [None] * L
        blocks = [self.layer(i) for i in range(L)]
        outs = []
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
        return self.norm(self.outputs(x, attn_mask, key=key, pos_ids=pos_ids)[self.config.Layers.name, -1])


_NEW_PARAMS = ("hyper", "query")  # path fragments of the parameters the designs add (HyperParams, pseudo-queries)


@OptimizerConfig.register_subclass("muonH_depth_arch")
@dataclass(frozen=True)
class DepthArchMuonHConfig(MuonHConfig):
    """MuonH exactly as the baseline for the baseline's parameters; the parameters a design adds (paths containing
    "hyper" or "query") get their own Adam with ``new_param_epsilon`` (see the module docstring for why)."""

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
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            return "adam_new" if label == "adam" and any(f in path_str for f in _NEW_PARAMS) else label

        return jax.tree_util.tree_map(relabel, base, paths, is_leaf=lambda x: isinstance(x, (str, hnn.Linear)))
