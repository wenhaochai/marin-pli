"""The fused SwiGLU kernels of levanter.kernels.pallas.swiglu_mgpu on one H100 at the 300m shape per device (M 131072, E 768,
F 2688; speed work 2026-10-27).
  bwd <config>      the backward kernel (d_h GEMM with d_gate / d_up / h in its epilogue) against XLA's d_h GEMM plus the
                    elementwise pass, outputs checked against that reference
  layer <fwd> <bwd> one MLP's forward + backward (value_and_grad of a fixed linear loss) three ways: the production path
                    (two GEMMs, XLA product, SWIGLU_VJP backward), the fused forward with XLA's backward (LEVANTER_SWIGLU_FUSED
                    today), and both fused kernels; gradients checked against the production path
Usage: python swiglu_bwd_bench.py bwd m128wgm3 | python swiglu_bwd_bench.py layer m128wgm3 m128wgm3 [iters]"""
import functools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from levanter.kernels.pallas import swiglu_mgpu as sw

M, E, F = 32 * 4096, 768, 2688
BF = jnp.bfloat16


def timed(f, *args, iters=20):
    for _ in range(3):
        out = f(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts) * 1e3)


def rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


def act_mul(g, u):
    return jax.nn.silu(g) * u


def bwd_bench(name, iters):
    cfg = sw.CONFIGS[name]
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    d_out = (jax.random.normal(k[0], (M, E), jnp.float32) * 1e-2).astype(BF)
    w_down = (jax.random.normal(k[1], (F, E), jnp.float32) * F ** -0.5).astype(BF)
    gu = jax.random.normal(k[2], (M, 2 * F), jnp.float32).astype(BF)
    nb = F // sw.BLOCK

    @jax.jit
    def ref(d_out, w_down, gu):
        d_h = (d_out @ w_down.T).reshape(M, nb, sw.BLOCK)
        gu4 = gu.reshape(M, nb, 2, sw.BLOCK)
        h, vjp = jax.vjp(act_mul, gu4[:, :, 0], gu4[:, :, 1])
        d_g, d_u = vjp(d_h)
        return jnp.stack([d_g, d_u], axis=2).reshape(M, 2 * F), h.reshape(M, F)

    fused = jax.jit(functools.partial(sw.swiglu_bwd_mgpu, config=cfg))
    (dgu0, h0), t_ref = timed(ref, d_out, w_down, gu, iters=iters)
    (dgu1, h1), t_fused = timed(fused, d_out, w_down.T, gu, iters=iters)
    print(f"SWIGLU_BWD {name}: XLA d_h GEMM + pass {t_ref:.3f} ms, fused {t_fused:.3f} ms, saving per layer "
          f"{t_ref - t_fused:+.3f} ms; max rel err d_gu {rel(dgu1, dgu0):.2e} h {rel(h1, h0):.2e}", flush=True)


def layer_bench(fwd_name, bwd_name, iters):
    fcfg, bcfg = sw.CONFIGS[fwd_name], sw.CONFIGS[bwd_name]
    nb = F // sw.BLOCK
    k = jax.random.split(jax.random.PRNGKey(0), 5)
    x = jax.random.normal(k[0], (M, E), jnp.float32).astype(BF)
    wg = (jax.random.normal(k[1], (E, F), jnp.float32) * E ** -0.5).astype(BF)
    wu = (jax.random.normal(k[2], (E, F), jnp.float32) * E ** -0.5).astype(BF)
    wd = (jax.random.normal(k[3], (F, E), jnp.float32) * F ** -0.5).astype(BF)
    r = (jax.random.normal(k[4], (M, E), jnp.float32) * 1e-2).astype(BF)

    @jax.custom_vjp
    def prod(x, wg, wu, wd):
        return act_mul(x @ wg, x @ wu) @ wd

    def prod_fwd(x, wg, wu, wd):
        g, u = x @ wg, x @ wu
        return act_mul(g, u) @ wd, (x, g, u, wg, wu, wd)

    def prod_bwd(res, d_out):
        x, g, u, wg, wu, wd = res
        h, vjp = jax.vjp(act_mul, g, u)
        d_g, d_u = vjp(d_out @ wd.T)
        return d_g @ wg.T + d_u @ wu.T, x.T @ d_g, x.T @ d_u, h.T @ d_out

    prod.defvjp(prod_fwd, prod_bwd)

    def make_fused(fused_bwd):
        @jax.custom_vjp
        def mlp(x, w_cat, wd):
            return sw.swiglu_fwd_mgpu(x, w_cat, fcfg)[1] @ wd

        def mlp_fwd(x, w_cat, wd):
            gu, h = sw.swiglu_fwd_mgpu(x, w_cat, fcfg)
            return h @ wd, (x, gu, w_cat, wd)

        def mlp_bwd(res, d_out):
            x, gu, w_cat, wd = res
            if fused_bwd:
                d_gu, h = sw.swiglu_bwd_mgpu(d_out, wd.T, gu, bcfg)
            else:
                d_h = (d_out @ wd.T).reshape(M, nb, sw.BLOCK)
                gu4 = gu.reshape(M, nb, 2, sw.BLOCK)
                h, vjp = jax.vjp(act_mul, gu4[:, :, 0], gu4[:, :, 1])
                d_g, d_u = vjp(d_h)
                d_gu = jnp.stack([d_g, d_u], axis=2).reshape(M, 2 * F)
                h = h.reshape(M, F)
            return d_gu @ w_cat.T, x.T @ d_gu, h.T @ d_out

        mlp.defvjp(mlp_fwd, mlp_bwd)
        return mlp

    def loss_prod(x, wg, wu, wd):
        return jnp.sum((prod(x, wg, wu, wd) * r).astype(jnp.float32))

    def loss_fused(mlp, x, wg, wu, wd):
        return jnp.sum((mlp(x, sw.interleave(wg, wu), wd) * r).astype(jnp.float32))

    variants = {
        "production": jax.jit(jax.value_and_grad(loss_prod, argnums=(0, 1, 2, 3))),
        "fused fwd": jax.jit(jax.value_and_grad(functools.partial(loss_fused, make_fused(False)), argnums=(0, 1, 2, 3))),
        "fused fwd+bwd": jax.jit(jax.value_and_grad(functools.partial(loss_fused, make_fused(True)), argnums=(0, 1, 2, 3))),
    }
    ref_grads, t_ref = None, None
    for vname, f in variants.items():
        (v, grads), t = timed(f, x, wg, wu, wd, iters=iters)
        if ref_grads is None:
            ref_grads, t_ref = grads, t
        errs = " ".join(f"{nm} {rel(a, b):.2e}" for nm, a, b in zip(("dx", "dwg", "dwu", "dwd"), grads, ref_grads))
        print(f"SWIGLU_LAYER fwd {fwd_name} bwd {bwd_name} {vname:14s}: {t:.3f} ms per layer ({t - t_ref:+.3f} vs "
              f"production; x12 layers {12 * (t - t_ref):+.1f} ms per step); max rel grad diff {errs}", flush=True)


if __name__ == "__main__":
    print(f"devices {jax.devices()}", flush=True)
    mode = sys.argv[1]
    if mode == "bwd":
        bwd_bench(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    else:
        layer_bench(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 20)
