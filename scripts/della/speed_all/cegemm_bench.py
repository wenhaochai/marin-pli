"""The CE logits GEMM on one H100 (speed work 2026-10-26, lse kernel v3 with the pls session): x [M, K] @ w [K, N] in
bf16 with f32 accumulation at the b-tiled CE block shapes (M 8192; K 768 for the 300m vocab runs, 512 for pls; N 128256),
timed as: XLA's dot (cuBLAS) to bf16 and to f32, the current Triton lse kernel (lse_logits_triton.py, the production
config), and JAX's Pallas Mosaic GPU Hopper matmul (jax.experimental.pallas.ops.gpu.hopper_matmul_mgpu: TMA, wgmma, two
compute warpgroups and a memory warpgroup, persistent) over a few tile configs. Prints one JSON line per form: median ms
per block over 20 timed calls and TFLOP/s."""
import itertools
import json
import statistics
import time

import jax
import jax.numpy as jnp

from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hm
from levanter.kernels.pallas.fused_cross_entropy_loss.lse_logits_triton import lse_and_logits


def bench(fn, *args, reps=20):
    out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


for K in (768, 512):
    M, N = 8192, 128256
    flop = 2 * M * N * K
    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), jnp.float32).astype(jnp.bfloat16)
    w = (jax.random.normal(jax.random.PRNGKey(1), (K, N), jnp.float32) * 0.03).astype(jnp.bfloat16)
    labels = jax.random.randint(jax.random.PRNGKey(2), (M,), 0, N)
    forms = {
        "xla_dot_bf16": jax.jit(lambda x, w: jnp.dot(x, w)),
        "xla_dot_f32": jax.jit(lambda x, w: jnp.dot(x, w, preferred_element_type=jnp.float32)),
        "triton_lse_prod": jax.jit(lambda x, w: lse_and_logits(
            x, w, labels, out_dtype=jnp.bfloat16, block_m=128, block_n=256, block_k=64, num_warps=8, num_stages=3,
            accurate_exp=False)),
    }
    for tm, tn, steps, minor, width, wgd in itertools.product((128,), (128, 64), (4, 3), (hm.MatmulDimension.M, hm.MatmulDimension.N),
                                                         (4, 8, 16), (hm.MatmulDimension.N, hm.MatmulDimension.M)):
        cfg = hm.TuningConfig(tile_m=tm, tile_n=tn, tile_k=64, max_concurrent_steps=steps, epi_tile_n=64, epi_tile_m=64,
                              grid_minor_dim=minor, grid_tile_width=width, wg_dimension=wgd)
        forms[f"pallas_mgpu_m{tm}_n{tn}_s{steps}_{minor}{width}_wg{wgd}"] = jax.jit(
            lambda x, w, cfg=cfg: hm.matmul(x, w, None, cfg))
    ref = None
    for name, fn in forms.items():
        try:
            ms = bench(fn, x, w)
        except Exception as e:  # a config that does not fit (shared memory, divisibility)
            print(json.dumps({"K": K, "form": name, "error": str(e)[:160]}), flush=True)
            continue
        print(json.dumps({"K": K, "form": name, "ms": round(ms, 3), "tflops": round(flop / ms / 1e9, 1)}), flush=True)
