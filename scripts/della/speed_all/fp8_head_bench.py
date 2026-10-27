"""Upper bound of an FP8 output head on one H100 (speed work 2026-10-26 night). The September FP8 runs quantized every
hax.nn.Linear and left the lm_head in bf16 (it lives in the fused cross-entropy, logs/kaiyue_fp8_qwen3_h100x4.yaml), and the
Linear layers at hidden 512-1024 gained nothing; at 128K the head's three GEMMs are about a quarter of the 300m step.
Times the head's GEMMs per b-tiled CE block (8192 rows, hidden 768) in bf16 and with float8 operands (e4m3 activations
and weights, e5m2 for the probabilities), bf16 outputs with f32 accumulation, for each vocabulary in VOCABS:
  fwd  z  = x @ w.T          [8192, 768] x [768, V]
  dx   dx = p @ w            [8192, V] x [V, 768]
  dw   dw = p.T @ x          [V, 8192] x [8192, 768]   (fp8 on Hopper needs both operands K-major: p transposed first)
and the transpose of p that the fp8 dw needs. No scales: this measures the GEMMs, not a training recipe. Median of ITERS
calls each. Usage: python fp8_head_bench.py [iters]"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ROWS, HID = 8192, 768
VOCABS = (128256, 64000, 32000)
E4, E5, BF = jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.bfloat16
DN = lambda lc, rc: (((lc,), (rc,)), ((), ()))  # noqa: E731


def timed(f, *args):
    c = jax.jit(f).lower(*args).compile()
    for _ in range(3):
        out = c(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        jax.block_until_ready(c(*args))
        ts.append(time.perf_counter() - t0)
    n_fp8 = c.as_text().count("f8e4m3fn") + c.as_text().count("f8e5m2")
    return np.median(ts) * 1e3, n_fp8


def main():
    print(f"devices {jax.devices()}", flush=True)
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    for v in VOCABS:
        x = jax.random.normal(k[0], (ROWS, HID), jnp.float32).astype(BF)
        w = (jax.random.normal(k[1], (v, HID), jnp.float32) * HID ** -0.5).astype(BF)
        p = jax.random.uniform(k[2], (ROWS, v), jnp.float32).astype(BF)
        x8, w8, p8 = x.astype(E4), w.astype(E4), p.astype(E5)
        flops = 2 * ROWS * HID * v
        rows = {
            "fwd bf16": timed(lambda a, b: jax.lax.dot_general(a, b, DN(1, 1), preferred_element_type=BF), x, w),
            "fwd fp8": timed(lambda a, b: jax.lax.dot_general(a, b, DN(1, 1), preferred_element_type=BF), x8, w8),
            "dx bf16": timed(lambda a, b: jax.lax.dot_general(a, b, DN(1, 0), preferred_element_type=BF), p, w),
            "dx fp8": timed(lambda a, b: jax.lax.dot_general(a, b, DN(1, 0), preferred_element_type=BF), p8, w8),
            "dw bf16": timed(lambda a, b: jax.lax.dot_general(a, b, DN(0, 0), preferred_element_type=BF), p, x),
            "dw fp8": timed(lambda a, b: jax.lax.dot_general(a, b, DN(0, 0), preferred_element_type=BF), p8, x8),
            "p transpose fp8": timed(lambda a: a.T.copy(), p8),
        }
        for name, (ms, n8) in rows.items():
            tf = flops / ms / 1e9 if "transpose" not in name else float("nan")
            print(f"FP8HEAD V={v} {name:16s} {ms:7.3f} ms  {tf:6.0f} TFLOP/s  fp8_mentions={n8}", flush=True)


if __name__ == "__main__":
    main()
