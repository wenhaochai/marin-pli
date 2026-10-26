"""lse kernel v3 on one H100 (speed work 2026-10-26, with the pls session): lse_logits_mgpu.lse_and_logits_mgpu (Pallas
Mosaic GPU GEMM with logsumexp partials in its epilogue) against the production Triton kernel (lse_logits_triton) and
cuBLAS, at the b-tiled CE block (8192 rows, 128256 columns; K 768 for the 300m vocab runs, 512 for pls). Correctness
against an f32 reference: the share of bf16 logits equal to the reference rounded to bf16, the largest |lse - reference|
and |label logit - reference|. Prints one JSON line per form: median ms over 20 calls, TFLOP/s and the errors."""
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from levanter.kernels.pallas.fused_cross_entropy_loss.lse_logits_mgpu import lse_and_logits_mgpu
from levanter.kernels.pallas.fused_cross_entropy_loss.lse_logits_triton import lse_and_logits


def bench(fn, *args, reps=20):
    jax.block_until_ready(fn(*args))
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
    w = (jax.random.normal(jax.random.PRNGKey(1), (K, N), jnp.float32) * 0.05).astype(jnp.bfloat16)
    labels = jax.random.randint(jax.random.PRNGKey(2), (M,), 0, N)
    ref = jnp.dot(x, w, preferred_element_type=jnp.float32)
    ref_lse = np.asarray(jax.nn.logsumexp(ref, axis=1))
    ref_lab = np.asarray(ref[jnp.arange(M), labels])
    ref_bf16 = np.asarray(ref.astype(jnp.bfloat16).astype(jnp.float32))
    del ref
    forms = {
        "cublas_bf16": jax.jit(lambda x, w: jnp.dot(x, w)),
        "triton_prod": jax.jit(lambda x, w, labels=labels: lse_and_logits(
            x, w, labels, out_dtype=jnp.bfloat16, block_m=128, block_n=256, block_k=64, num_warps=8, num_stages=3,
            accurate_exp=False)),
        "mgpu_v3": jax.jit(lambda x, w, labels=labels: lse_and_logits_mgpu(x, w, labels)),
        "mgpu_v3_nolabels": jax.jit(lambda x, w: lse_and_logits_mgpu(x, w)),
        "label_gather_dot": jax.jit(lambda x, w, labels=labels: jnp.sum(
            x.astype(jnp.float32) * jnp.take(w, labels, axis=1).T.astype(jnp.float32), axis=1)),
    }
    for name, fn in forms.items():
        try:
            ms = bench(fn, x, w)
            row = {"K": K, "form": name, "ms": round(ms, 3), "tflops": round(flop / ms / 1e9, 1)}
            if name in ("triton_prod", "mgpu_v3"):
                logits, lse, lab = fn(x, w)
                lg = np.asarray(logits.astype(jnp.float32))
                row["logits_bf16_equal"] = float(np.mean(lg == ref_bf16))
                row["lse_max_abs_err"] = float(np.max(np.abs(np.asarray(lse) - ref_lse)))
                row["label_max_abs_err"] = float(np.max(np.abs(np.asarray(lab) - ref_lab)))
            print(json.dumps(row), flush=True)
        except Exception as e:
            print(json.dumps({"K": K, "form": name, "error": f"{type(e).__name__}: {str(e)[:400]}"}), flush=True)
