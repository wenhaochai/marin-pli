"""H100 attention kernel benchmark at the 300m per-device shape (speed-all, owner 2026-10-25: "可以再考虑底层，比如kernel").

Shape: B sequences (default 32, the per-device microbatch of the 300m runs), T tokens (default 4096), 12 heads, head_dim 64,
bf16, causal attention inside documents (levanter's block_cross_document_attention: segment ids = cumsum of document
starts, so each query sees the keys of its own document up to itself). Two document layouts: 'docs' (lengths drawn
from a lognormal with mean ~1000 tokens, cut to the window) and 'one' (one document per row, plain causal).
Backends:
  jax_flash  levanter dot_product_attention, AttentionBackend.JAX_FLASH, block FLASH (default 1024): today's path
  fa4        levanter dot_product_attention, AttentionBackend.FA4_CUTE (branch speed-all)
  pallas     jax.experimental.pallas.ops.gpu.attention.mha (Triton), causal + segment_ids
  cudnn      jax._src.cudnn.fused_attention_stablehlo.dot_product_attention, packed documents (q/kv offsets, MAXSEG per row)
Per backend and layout: median ms of fwd+bwd (value_and_grad of sum(out * cotangent) w.r.t. q, k, v) and of fwd only,
over ITERS timed calls after warmup; relative L2 error of out, dq, dk, dv against an fp32 dense reference on the first
REF rows. One JSON line per result to stdout. Contains no training; synthetic inputs only.
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import jax
import jax.numpy as jnp
import haliax as hax
from haliax import Axis

from levanter.layers.attention import AttentionBackend, dot_product_attention
from levanter.layers.attention_mask import AttentionMask

B, T, NH, D = int(os.environ.get("B", 32)), int(os.environ.get("T", 4096)), 12, 64
FLASH = int(os.environ.get("FLASH", 1024))
ITERS, REF, MAXSEG = int(os.environ.get("ITERS", 10)), int(os.environ.get("REF", 2)), int(os.environ.get("MAXSEG", 64))
BACKENDS = os.environ.get("BACKENDS", "jax_flash fa4 pallas cudnn").split()
LAYOUTS = os.environ.get("LAYOUTS", "docs one").split()
SCALE = 1.0 / np.sqrt(D)
print(json.dumps({"devices": [str(d) for d in jax.devices()], "jax": jax.__version__, "B": B, "T": T, "flash_block": FLASH}), flush=True)


def segments(layout, rng):
    if layout == "one":
        return np.zeros((B, T), np.int32)
    seg = np.zeros((B, T), np.int32)
    for b in range(B):
        pos, s = int(rng.integers(0, 800)), 0   # the window starts inside a document
        while pos < T:
            seg[b, pos:] = s + 1
            pos += max(1, int(rng.lognormal(np.log(600), 1.0)))
            s += 1
    return seg


def timed(f, *a):
    out = f(*a)
    jax.block_until_ready(out)
    for _ in range(2):
        jax.block_until_ready(f(*a))
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1e3, out


Batch, Pos, KPos, Heads, Key = Axis("batch", B), Axis("position", T), Axis("key_position", T), Axis("heads", NH), Axis("head_size", D)


def levanter_attn(backend):
    be = {"jax_flash": AttentionBackend.JAX_FLASH, "fa4": AttentionBackend.FA4_CUTE}[backend]

    def f(q, k, v, seg):
        qn = hax.named(q, (Batch, Pos, Heads, Key))
        kn = hax.named(k, (Batch, KPos, Heads, Key))
        vn = hax.named(v, (Batch, KPos, Heads, Key))
        mask = AttentionMask.causal().with_segment_ids(hax.named(seg, (Batch, Pos)))
        out = dot_product_attention(Pos, KPos, Key, qn, kn, vn, mask=mask, attn_backend=be,
                                    flash_block_size=FLASH if backend == "jax_flash" else None, scaling_factor=SCALE)
        return out.rearrange((Batch, Pos, Heads, Key)).array
    return f


def pallas_attn(q, k, v, seg):
    from jax.experimental.pallas.ops.gpu import attention as pa
    return pa.mha(q, k, v, seg, sm_scale=SCALE, causal=True)


def cudnn_attn(q, k, v, seg):
    from jax._src.cudnn import fused_attention_stablehlo as fa
    start = jnp.concatenate([jnp.ones((B, 1), bool), seg[:, 1:] != seg[:, :-1]], axis=1)

    def row(st):
        idx = jnp.nonzero(st, size=MAXSEG, fill_value=-1)[0]
        nxt = jnp.concatenate([idx[1:], jnp.array([-1])])
        n = jnp.sum(idx >= 0)
        nxt = jnp.where(jnp.arange(MAXSEG) == n - 1, T, nxt)
        ln = jnp.where(idx >= 0, nxt - idx, -1)
        return ln.astype(jnp.int32), jnp.concatenate([idx, jnp.array([-1])]).astype(jnp.int32)
    ln, off = jax.vmap(row)(start)
    out = fa.dot_product_attention(q, k, v, q_seqlen=ln, kv_seqlen=ln, q_offsets=off, kv_offsets=off, scale=SCALE,
                                   mask_type=fa.MaskType.CAUSAL, qkv_layout="BTNH")
    return out[0] if isinstance(out, tuple) else out


def reference(q, k, v, seg):
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k) * SCALE
    i = jnp.arange(T)
    m = (i[:, None] >= i[None, :])[None, None] & (seg[:, None, :, None] == seg[:, None, None, :])
    p = jax.nn.softmax(jnp.where(m, s, -1e30), axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", p, v)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


rng = np.random.default_rng(0)
key = jax.random.PRNGKey(0)
q, k, v, ct = (jax.random.normal(kk, (B, T, NH, D), jnp.float32).astype(jnp.bfloat16) for kk in jax.random.split(key, 4))
for layout in LAYOUTS:
    seg_np = segments(layout, rng)
    nseg = [len(np.unique(r)) for r in seg_np]
    print(json.dumps({"layout": layout, "segments_per_row_mean": float(np.mean(nseg)), "max": int(np.max(nseg))}), flush=True)
    seg = jnp.asarray(seg_np)
    rf = jax.jit(jax.value_and_grad(lambda q, k, v: jnp.sum(reference(q, k, v, seg[:REF]) * ct[:REF].astype(jnp.float32)), argnums=(0, 1, 2)))
    (_, rg), ro = rf(q[:REF], k[:REF], v[:REF]), jax.jit(lambda q, k, v: reference(q, k, v, seg[:REF]))(q[:REF], k[:REF], v[:REF])
    for be in BACKENDS:
        try:
            f = levanter_attn(be) if be in ("jax_flash", "fa4") else {"pallas": pallas_attn, "cudnn": cudnn_attn}[be]
            fwd = jax.jit(lambda q, k, v, s: f(q, k, v, s))
            vg = jax.jit(jax.value_and_grad(lambda q, k, v, s: jnp.sum(f(q, k, v, s).astype(jnp.float32) * ct.astype(jnp.float32)), argnums=(0, 1, 2)))
            t_f, out = timed(fwd, q, k, v, seg)
            t_fb, (_, g) = timed(vg, q, k, v, seg)
            # errors on the first REF rows: the backend run on those rows alone, against the fp32 reference
            o_s = jax.jit(lambda q, k, v, s: f(q, k, v, s))
            errs = {"out": rel(out[:REF], ro)}
            # gradient of the REF-row loss = the first REF rows of the full gradient (rows are independent)
            for name, a, r in zip(("dq", "dk", "dv"), g, rg):
                errs[name] = rel(a[:REF], r)
            print(json.dumps({"layout": layout, "backend": be, "fwd_ms": round(t_f, 2), "fwd_bwd_ms": round(t_fb, 2),
                              "rel_err": {k2: float(f"{v2:.2e}") for k2, v2 in errs.items()}}), flush=True)
        except Exception as e:
            print(json.dumps({"layout": layout, "backend": be, "error": f"{type(e).__name__}: {str(e)[:300]}"}), flush=True)
            traceback.print_exc(file=sys.stderr)
