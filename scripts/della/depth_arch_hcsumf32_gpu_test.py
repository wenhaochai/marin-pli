"""GPU check for depth_hc_state_f32 (audit 2026-11-03): one hyper_step of the real model with differences R times the sum, on an Ampere or later GPU, with the
default jax_default_matmul_precision as in training. A float32 dot at default precision rounds both operands to TF32 there, so this is the test the CPU file cannot do.
    ssh della-vis2 'cd REPO && CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. nice -n 19 timeout 300 .venv/bin/python -u scripts/della/depth_arch_hcsumf32_gpu_test.py'
Expected at R = 1e4: the relative error of the stream sum after one layer (two read-and-mix sublayers) matches the CPU's float32 value, 3.5e-3 (the random mixing grows the
differences about 6 times per layer), so under 1e-2. A40, 2026-11-03: this code 3.9e-7 at R = 1, 1.1e-4 at R = 300, 3.8e-3 at R = 1e4; default-precision dots (commit 52942aeb4e) 1.3e-3, 3.7e-1, 12."""
import os

os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import dataclasses  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import jmp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

print("devices", jax.devices(), "matmul precision", jax.config.jax_default_matmul_precision, flush=True)
assert jax.default_backend() == "gpu"
rng = np.random.default_rng(0)
B, T, V, L, D, M = 4, 64, 300, 4, 96, 4
Batch, Pos, Vocab, Stream, Embed = Axis("batch", B), Axis("position", T), Axis("vocab", V), Axis("stream", M), Axis("embed", D)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=128, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True, hybrid_norm=True, depth_arch="hc", scan_layers=False)
mask = AttentionMask.causal()
mp = jmp.get_policy("p=f32,c=bfloat16")
m_f = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, depth_hc_res="sum", depth_hc_state_f32=True), key=jrandom.PRNGKey(0))


def rand_like(a, scale, k):
    return hax.named(jnp.asarray(np.random.default_rng(k).standard_normal(a.array.shape) * scale, a.array.dtype), a.axes)


hy = [dataclasses.replace(p, read=rand_like(p.read, 0.3, 100 + j), res=rand_like(p.res, 0.3, 110 + j), w_read=rand_like(p.w_read, 0.3, 120 + j), w_res=rand_like(p.w_res, 0.3, 130 + j), scale=rand_like(p.scale, 1.0, 140 + j),
                          write=hax.named(jnp.full(p.write.array.shape, -1.0, p.write.array.dtype), p.write.axes), w_write=hax.named(jnp.zeros_like(p.w_write.array), p.w_write.axes)) for j, p in enumerate(m_f.transformer.layers.stacked.hyper)]
mc = mp.cast_to_compute(eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_f, tuple(hy)))
lay = mc.transformer.layer(0)
step = eqx.filter_jit(lambda H: lay.hyper_step(H, mask, key=None, pos_ids=None, layer=0))
for R in (1.0, 300.0, 1e4, 1e5):
    s = rng.standard_normal((B, T, D))
    d = rng.standard_normal((M, B, T, D)); d -= d.mean(0, keepdims=True)
    Hnp = (s[None] / M + R * d * np.linalg.norm(s) / np.linalg.norm(d)).astype(np.float32)
    true = Hnp.astype(np.float64).sum(0)
    o = step(hax.named(jnp.asarray(Hnp), (Stream, Batch, Pos, Embed))).rearrange((Stream, Batch, Pos, Embed))
    assert o.array.dtype == jnp.float32
    e = np.linalg.norm(np.asarray(o.array, np.float64).sum(0) - true) / np.linalg.norm(true)
    print(f"|dif|/|sum| {R:>8g}: relative error of the stream sum after one layer {e:.1e}", flush=True)
    if R == 1e4:
        assert e < 1e-2, e
print("ALL OK")
