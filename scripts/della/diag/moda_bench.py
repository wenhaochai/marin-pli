"""Time one forward + backward of the 48-layer backbone (width 512, the 130m layer, seq 4096, one sequence) per design
and per moda variant, on one GPU. Backbone only: no per-layer heads (they cost the same in every design). Loss = mean
square of every layer's output. Usage: VARIANTS="preln attnres moda moda:bf16 ..." python moda_bench.py"""
import os, time, math, sys
import jax, jax.numpy as jnp, jax.random as jrandom, equinox as eqx
import haliax as hax
from haliax import Axis
from levanter.layers.attention import AttentionBackend, AttentionMask
import experiments.references.depth_arch_qwen3 as D
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config

T, L = int(os.environ.get("T", 4096)), int(os.environ.get("L", 48))
orig_attn = D._moda_attention

def attn_bf16(attn, x, mask, depth, *, pos_ids, chunk):
    """_moda_attention with bf16 matmul inputs and float32 accumulation (softmax still float32)."""
    q, k, v, k_depth = D._qk_normed_qkv(attn, x, pos_ids)
    cfg = attn.config
    scale = cfg.scaling_factor if cfg.scaling_factor is not None else 1.0 / math.sqrt(cfg.HeadSize.size)
    Pos = q.resolve_axis("position"); KPos = Pos.alias("key_position")
    ks, vs = k.rename({"position": "key_position"}), v.rename({"position": "key_position"})
    dk, dv, valid = depth if depth is not None else (None, None, None)
    outs = []
    for start in range(0, Pos.size, chunk):
        sl = hax.dslice(start, min(chunk, Pos.size - start))
        m = D.materialize_mask(mask, Pos, KPos, q_slice=sl)
        def part(qc, ks, vs, dkc, dvc, m):
            s = hax.dot(qc, ks, axis="head_size", preferred_element_type=jnp.float32) * scale
            if m is not None: s = hax.where(m, s, -1e30)
            mx = hax.max(s, axis="key_position")
            if dkc is not None:
                sd = hax.dot(qc, dkc, axis="head_size", preferred_element_type=jnp.float32) * scale
                if valid is not None: sd = hax.where(valid, sd, -1e30)
                mx = hax.maximum(mx, hax.max(sd, axis="depth"))
            ps = hax.exp(s - mx); z = hax.sum(ps, axis="key_position")
            o = hax.dot(ps.astype(vs.dtype), vs, axis="key_position", preferred_element_type=jnp.float32)
            if dkc is not None:
                pd = hax.exp(sd - mx); z = z + hax.sum(pd, axis="depth")
                o = o + hax.dot(pd.astype(dvc.dtype), dvc, axis="depth", preferred_element_type=jnp.float32)
            return (o / z).astype(x.dtype)
        qc = q["position", sl]
        dkc = dk["position", sl] if dk is not None else None
        dvc = dv["position", sl] if dv is not None else None
        outs.append(eqx.filter_checkpoint(part)(qc, ks, vs, dkc, dvc, m))
    o = hax.concatenate("position", outs) if len(outs) > 1 else outs[0]
    return attn.o_proj(o.flatten_axes(("kv_head", "q_heads_per_group"), "heads")), k_depth, v

def attn_nodepth(attn, x, mask, depth, *, pos_ids, chunk):
    return orig_attn(attn, x, mask, None, pos_ids=pos_ids, chunk=chunk)

def attn_flash(attn, x, mask, depth, *, pos_ids, chunk):
    """No depth part: the baseline's fused attention (JAX_FLASH) for the sequence, plus the depth key / value outputs."""
    _, _, v, k_depth = D._qk_normed_qkv(attn, x, pos_ids)
    return attn(x=x, mask=mask, key=None, pos_ids=pos_ids), k_depth, v


def make_variant(kind):
    """kind ew: depth logits and output as elementwise products and sums in float32 (no batched dot); sg: the depth
    part reads stop_gradient(dk, dv) (diagnostic: no gradient to the buffer)."""
    def attn(attn, x, mask, depth, *, pos_ids, chunk):
        q, k, v, k_depth = D._qk_normed_qkv(attn, x, pos_ids)
        cfg = attn.config
        scale = cfg.scaling_factor if cfg.scaling_factor is not None else 1.0 / math.sqrt(cfg.HeadSize.size)
        Pos = q.resolve_axis("position"); KPos = Pos.alias("key_position")
        ks, vs = k.rename({"position": "key_position"}), v.rename({"position": "key_position"})
        dk, dv, valid = depth if depth is not None else (None, None, None)
        if kind == "sg" and dk is not None:
            dk, dv = jax.lax.stop_gradient(dk), jax.lax.stop_gradient(dv)
        outs = []
        for start in range(0, Pos.size, chunk):
            sl = hax.dslice(start, min(chunk, Pos.size - start))
            m = D.materialize_mask(mask, Pos, KPos, q_slice=sl)
            def part(qc, ks, vs, dkc, dvc, m):
                s = hax.dot(qc.astype(jnp.float32), ks.astype(jnp.float32), axis="head_size") * scale
                if m is not None: s = hax.where(m, s, -1e30)
                mx = hax.max(s, axis="key_position")
                if dkc is not None:
                    if kind == "ew":
                        qr = qc.rearrange((..., "position", "kv_head", "q_heads_per_group", "head_size"))
                        da = dkc.rearrange((..., "position", "kv_head", "head_size", "depth")).array.astype(jnp.float32)
                        sd = hax.named((qr.array.astype(jnp.float32)[..., None] * da[..., None, :, :]).sum(-2) * scale, qr.axes[:-1] + (dkc.resolve_axis("depth"),))
                    else:
                        sd = hax.dot(qc.astype(jnp.float32), dkc.astype(jnp.float32), axis="head_size") * scale
                    if valid is not None: sd = hax.where(valid, sd, -1e30)
                    mx = hax.maximum(mx, hax.max(sd, axis="depth"))
                ps = hax.exp(s - mx); z = hax.sum(ps, axis="key_position")
                o = hax.dot(ps, vs.astype(jnp.float32), axis="key_position")
                if dkc is not None:
                    pd = hax.exp(sd - mx); z = z + hax.sum(pd, axis="depth")
                    if kind == "ew":
                        pr = pd.rearrange((..., "position", "kv_head", "q_heads_per_group", "depth"))
                        va = dvc.rearrange((..., "position", "kv_head", "depth", "head_size")).array.astype(jnp.float32)
                        od = hax.named((pr.array[..., None] * va[..., None, :, :]).sum(-2), pr.axes[:-1] + (dvc.resolve_axis("head_size"),))
                        o = o + od.rearrange(o.axes)
                    else:
                        o = o + hax.dot(pd, dvc.astype(jnp.float32), axis="depth")
                return (o / z).astype(x.dtype)
            qc = q["position", sl]
            dkc = dk["position", sl] if dk is not None else None
            dvc = dv["position", sl] if dv is not None else None
            outs.append(eqx.filter_checkpoint(part)(qc, ks, vs, dkc, dvc, m))
        o = hax.concatenate("position", outs) if len(outs) > 1 else outs[0]
        return attn.o_proj(o.flatten_axes(("kv_head", "q_heads_per_group"), "heads")), k_depth, v
    return attn


def moda_scan_layout(self, x, attn_mask, *, key, pos_ids):
    """_moda_scan with the depth buffers laid out (.., position, kv_head, depth, head_size): the same values, depth
    next to head_size in memory."""
    from typing import Any, cast
    cfg, L = self.config, self.config.num_layers
    acfg = cfg.attention_config()
    Depth = Axis("depth", 2 * L)
    buf = hax.zeros(tuple(a for a in x.axes if a.name != cfg.Embed.name) + (acfg.KVHeads, Depth, acfg.HeadSize), dtype=x.dtype)
    chunk = cast(Any, cfg).moda_chunk
    def step(layer, carry, i, *, key):
        h, dk, dv = carry
        n = i.array if isinstance(i, hax.NamedArray) else i
        h, (ka, va), (kf, vf) = layer.moda_step(h, attn_mask, (dk, dv, hax.arange(Depth) < 2 * n), key=key, pos_ids=pos_ids, chunk=chunk)
        dk = dk.at["depth", 2 * n].set(ka.astype(dk.dtype)).at["depth", 2 * n + 1].set(kf.astype(dk.dtype))
        dv = dv.at["depth", 2 * n].set(va.astype(dv.dtype)).at["depth", 2 * n + 1].set(vf.astype(dv.dtype))
        return (h, dk, dv), h
    _, outs = self._two_level_scan(step, (x, buf, buf), hax.arange(cfg.Layers), key=key)
    return outs
orig_scan = D.DepthArchTransformer._moda_scan

import inspect, textwrap
_src = textwrap.dedent(inspect.getsource(orig_attn))
for _a, _b in [('qc = q["position", sl]', 'qc = q["position", ss]'), ('dk["position", sl]', 'dk["position", ss]'), ('dv["position", sl]', 'dv["position", ss]'),
               ("m = materialize_mask(", "ss = slice(start, min(start + chunk, Pos.size)); m = materialize_mask(")]:
    assert _a in _src, _a; _src = _src.replace(_a, _b)
_ns = dict(D.__dict__); exec(_src.replace("def _moda_attention", "def static_attn"), _ns); static_attn = _ns["static_attn"]

def run(name):
    arch, _, var = name.partition(":")
    D._moda_attention = {"": orig_attn, "bf16": attn_bf16, "nodepth": attn_nodepth, "flash": attn_flash, "c4096": orig_attn, "c512": orig_attn, "ew": make_variant("ew"), "sg": make_variant("sg"), "layout": orig_attn, "layout_static": orig_attn}[var]
    D.DepthArchTransformer._moda_scan = moda_scan_layout if var in ("layout", "layout_static") else orig_scan
    if var in ("layout", "layout_static"): D._moda_attention = static_attn if var == "layout_static" else orig_attn
    chunk = {"c4096": 4096, "c512": 512}.get(var, 1024)
    cfg = PerLayerQwen3Config(max_seq_len=T, hidden_dim=512, intermediate_dim=1792, num_layers=L, num_heads=8, num_kv_heads=8, hybrid_norm=True,
                              attn_backend=AttentionBackend.JAX_FLASH, pls_weight=1.0, pls_separate_heads=True, depth_arch=arch, scan_layers=False, moda_chunk=chunk)
    model = D.DepthArchTransformer.init(cfg, arch, key=jrandom.PRNGKey(0))
    Pos = Axis("position", T)
    x = hax.random.normal(jrandom.PRNGKey(1), (Pos, cfg.Embed)).astype(jnp.float32 if os.environ.get("F32") else jnp.bfloat16)
    mask = AttentionMask.causal()
    def loss(m):
        mb = m if os.environ.get("F32") else jax.tree.map(lambda a: a.astype(jnp.bfloat16) if eqx.is_inexact_array(a) else a, m)
        outs = mb.outputs(x, mask, key=None, pos_ids=None)
        return hax.mean(hax.square(outs.astype(jnp.float32))).scalar()
    f = eqx.filter_jit(loss) if os.environ.get("FWD") else eqx.filter_jit(eqx.filter_value_and_grad(loss))
    t0 = time.time(); r = f(model); jax.block_until_ready(r); lo = r if os.environ.get("FWD") else r[0]; tc = time.time() - t0
    ts = []
    for _ in range(3):
        t0 = time.time(); r = f(model); jax.block_until_ready(r); lo = r if os.environ.get("FWD") else r[0]; ts.append(time.time() - t0)
    if os.environ.get("PROF"):
        d = os.environ["PROF"] + "/" + name.replace(":", "_")
        with jax.profiler.trace(d):
            r = f(model); jax.block_until_ready(r)
    mem = jax.local_devices()[0].memory_stats() or {}
    if not os.environ.get("FWD"):
        gl = jax.tree.leaves(eqx.filter(r[1], eqx.is_inexact_array)); import numpy as _np
        _np.save(f"/scratch/gpfs/GROUP/USER/tmp/pls_diag/grad_{name.replace(':', '_')}.npy", _np.concatenate([_np.asarray(g, _np.float32).ravel() for g in gl]))
    print(f"{name:16s} step {min(ts)*1e3:8.1f} ms  first {tc:6.1f} s  peak {mem.get('peak_bytes_in_use', 0)/2**30:6.2f} GiB  loss {float(lo):.8f}", flush=True)

for v in os.environ.get("VARIANTS", "preln attnres moda").split():
    try:
        run(v)
    except Exception as e:
        print(f"{v:16s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
    jax.clear_caches()
