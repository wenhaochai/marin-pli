"""Does a float32 hax.dot on an Ampere-class GPU round its operands to TF32 (10 mantissa bits)? Measures the sum error of one row-stochastic 4 x 4 mixing
with |dif| / |sum| = R, the shape of hc's mixing: res (batch, position, stream, stream_out) dot H (stream, batch, position, embed), contraction over stream.
Compares the default precision, Precision.HIGHEST, and a broadcast multiply plus hax.sum (no dot_general). Same input on every path."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp, numpy as np
import haliax as hax
from haliax import Axis
print("devices", jax.devices(), flush=True)
B, T, D, M = 4, 256, 128, 4
Batch, Pos, Embed, S, So = Axis("batch", B), Axis("position", T), Axis("embed", D), Axis("stream", M), Axis("stream_out", M)
rng = np.random.default_rng(0)
A = rng.random((B, T, M, M)); A /= A.sum(3, keepdims=True)
res = hax.named(jnp.asarray(A, jnp.float32), (Batch, Pos, S, So))
@jax.jit
def d_default(r, h): return hax.dot(r, h, axis="stream")
@jax.jit
def d_high(r, h): return hax.dot(r, h, axis="stream", precision=jax.lax.Precision.HIGHEST)
@jax.jit
def d_elem(r, h): return hax.sum(r.broadcast_axis(Embed) * h.broadcast_axis(So), axis="stream")
for R in (1.0, 300.0, 1e4, 1e5):
    s = rng.standard_normal((B, T, D)); d = rng.standard_normal((M, B, T, D)); d -= d.mean(0, keepdims=True)
    Hnp = (s[None] / M + R * d * np.linalg.norm(s) / np.linalg.norm(d)).astype(np.float32)
    H = hax.named(jnp.asarray(Hnp), (S, Batch, Pos, Embed))
    true = Hnp.astype(np.float64).sum(0)
    out = {}
    for name, f in (("default", d_default), ("highest", d_high), ("elementwise", d_elem)):
        o = f(res, H).rearrange(("stream_out", "batch", "position", "embed"))
        out[name] = np.linalg.norm(np.asarray(o.array, np.float64).sum(0) - true) / np.linalg.norm(true)
    print(f"|dif|/|sum| {R:>8g}: rel error of the sum after one mixing: default {out['default']:.1e}, HIGHEST {out['highest']:.1e}, elementwise {out['elementwise']:.1e}", flush=True)
for name, f in (("default", d_default), ("highest", d_high), ("elementwise", d_elem)):
    txt = f.lower(res, H).compile().as_text()
    lines = [l.strip() for l in txt.splitlines() if ("custom-call" in l and ("gemm" in l.lower() or "cublas" in l.lower())) or (" dot(" in l)]
    print(f"--- {name}: {len(lines)} gemm/dot lines", flush=True)
    for l in lines[:2]:
        print("   ", l[:420], flush=True)
