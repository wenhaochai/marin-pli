"""Copies of saved activations in the layer scan (speed work 2026-10-26 night): can a GEMM write the value the backward
pass reads straight into the scan's stacked buffer? In the 300m train step every saved name costs a full copy per layer
(profile w350: loop_dynamic_update_slice_fusion.2/.3, 5.69 ms each per 128K step for mlp_gate and mlp_up), because the
GEMM writes its own buffer and the scan then copies it into the stack; XLA's dynamic-slice fusion lets a cuBLAS call
write into a dynamic-update-slice in place only when the slice is the call's only user, and here the forward pass also
reads the value. The production A/Bs of that flag gained nothing (w398, w413).

Two versions of one 12-layer SwiGLU MLP stack at the per-device 300m shape (32 x 4096 rows, 768 -> 2688 -> 768, bf16)
saving gate and up per layer:
  A  jax.checkpoint(save_only_these_names) inside lax.scan, as levanter's ScanCheckpointPolicy does;
  B  a custom_vjp over the stack whose forward scan writes gate, up and the layer input into explicit [L, ...] buffers
     (each GEMM output's only user is its dynamic-update-slice; the forward reads the values back from the buffers
     behind an optimization barrier) and whose backward scan reads them, with the same backward GEMMs as A's autodiff.
Prints, per version: median ms per forward + backward over ITERS steps, the optimized HLO's dynamic-update-slice fusions
and dynamic-slice-fusion custom fusions, and B's gradients against A's (max relative difference per weight).
Usage: python scan_save_bench.py <tag> [iters]   (XLA_FLAGS from the caller; the HLO goes to $TMPDIR/hlo_<tag>_<ver>)"""
import functools
import glob
import os
import re
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.ad_checkpoint import checkpoint_name

TAG = sys.argv[1]
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
L, M, D, F = 12, 32 * 4096, 768, 2688
BF = jnp.bfloat16
SAVE = jax.checkpoint_policies.save_only_these_names("mlp_gate", "mlp_up")


def layer(x, wg, wu, wd):
    g = checkpoint_name(x @ wg, "mlp_gate")
    u = checkpoint_name(x @ wu, "mlp_up")
    return x + (jax.nn.silu(g) * u) @ wd


def stack_a(x, wg, wu, wd):
    f = jax.checkpoint(layer, policy=SAVE)
    return lax.scan(lambda c, w: (f(c, *w), None), x, (wg, wu, wd))[0]


@jax.custom_vjp
def stack_b(x, wg, wu, wd):
    return lax.scan(lambda c, w: (layer(c, *w), None), x, (wg, wu, wd))[0]


def _fwd(x, wg, wu, wd):
    bufs = (jnp.zeros((L, M, F), BF), jnp.zeros((L, M, F), BF), jnp.zeros((L, M, D), BF))

    def body(carry, xs):
        x, (G, U, X) = carry
        wg_i, wu_i, wd_i, i = xs
        X = lax.dynamic_update_index_in_dim(X, x, i, 0)
        G = lax.dynamic_update_index_in_dim(G, x @ wg_i, i, 0)
        U = lax.dynamic_update_index_in_dim(U, x @ wu_i, i, 0)
        G, U = lax.optimization_barrier((G, U))
        g = lax.dynamic_index_in_dim(G, i, 0, keepdims=False)
        u = lax.dynamic_index_in_dim(U, i, 0, keepdims=False)
        return (x + (jax.nn.silu(g) * u) @ wd_i, (G, U, X)), None

    (y, bufs), _ = lax.scan(body, (x, bufs), (wg, wu, wd, jnp.arange(L)))
    return y, (bufs, wg, wu, wd)


def _bwd(res, dy):
    (G, U, X), wg, wu, wd = res

    def body(dx, xs):
        wg_i, wu_i, wd_i, i = xs
        g = lax.dynamic_index_in_dim(G, i, 0, keepdims=False)
        u = lax.dynamic_index_in_dim(U, i, 0, keepdims=False)
        x = lax.dynamic_index_in_dim(X, i, 0, keepdims=False)
        h, act_vjp = jax.vjp(lambda g_, u_: jax.nn.silu(g_) * u_, g, u)
        dh = dx @ wd_i.T
        dg, du = act_vjp(dh)
        return dx + dg @ wg_i.T + du @ wu_i.T, (x.T @ dg, x.T @ du, h.T @ dx)

    dx, (dwg, dwu, dwd) = lax.scan(body, dy, (wg, wu, wd, jnp.arange(L)), reverse=True)
    return dx, dwg, dwu, dwd


stack_b.defvjp(_fwd, _bwd)


def loss(stack, ws, x):
    return jnp.sum(jnp.square(stack(x, *ws).astype(jnp.float32))) * 1e-6


def hlo_counts(text):
    dus = len(re.findall(r"^\s*\S*dynamic_update_slice_fusion\S* = ", text, re.M))
    dsf = len(re.findall(r"dynamic_slice_fusion|address_computation", text))
    copies = len(re.findall(r"^\s*copy\S* = ", text, re.M))
    gemms = len(re.findall(r"custom_call_target=\"__cublas", text))
    return f"dus_fusions={dus} dsf_mentions={dsf} copies={copies} cublas_calls={gemms}"


def main():
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    x = jax.random.normal(k[0], (M, D), jnp.float32).astype(BF)
    ws = (jax.random.normal(k[1], (L, D, F), jnp.float32) * D ** -0.5).astype(BF), \
         (jax.random.normal(k[2], (L, D, F), jnp.float32) * D ** -0.5).astype(BF), \
         (jax.random.normal(k[3], (L, F, D), jnp.float32) * F ** -0.5 * 0.5).astype(BF)
    grads = {}
    for ver, stack in (("A", stack_a), ("B", stack_b)):
        step = jax.jit(jax.value_and_grad(functools.partial(loss, stack)))
        t0 = time.perf_counter()
        compiled = step.lower(ws, x).compile()
        t_compile = time.perf_counter() - t0
        text = compiled.as_text()
        os.makedirs(f"{os.environ.get('HLO_DIR', os.environ.get('TMPDIR', '/tmp'))}/hlo_{TAG}_{ver}", exist_ok=True)
        with open(f"{os.environ.get('HLO_DIR', os.environ.get('TMPDIR', '/tmp'))}/hlo_{TAG}_{ver}/step.txt", "w") as fh:
            fh.write(text)
        for _ in range(3):
            v, g = compiled(ws, x)
        jax.block_until_ready(g)
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            v, g = compiled(ws, x)
            jax.block_until_ready(g)
            ts.append(time.perf_counter() - t0)
        grads[ver] = [np.asarray(jax.device_get(a), np.float32) for a in g]
        print(f"BENCH {TAG} {ver}: median {np.median(ts) * 1e3:.2f} ms min {np.min(ts) * 1e3:.2f} ms over {ITERS} "
              f"(compile {t_compile:.1f}s) loss {float(v):.6f} {hlo_counts(text)}", flush=True)
        del step, compiled, g
    for name, a, b in zip(("dwg", "dwu", "dwd"), grads["A"], grads["B"]):
        rel = np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-30)
        print(f"GRAD {TAG} {name}: max|B-A|/max|A| = {rel:.3e}", flush=True)


if __name__ == "__main__":
    print(f"devices {jax.devices()} XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}", flush=True)
    main()
