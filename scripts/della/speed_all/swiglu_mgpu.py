"""SwiGLU forward in one H100 GEMM (speed work 2026-10-26 night; prototype): the MLP's gate and up projections as one Pallas
Mosaic GPU GEMM whose epilogue also forms h = silu(gate) * up, so the separate elementwise pass of the 300m step (8.2 ms
per 128K step, profile w458: it reads gate and up back, 1.4 GB per layer) goes away. The weights are concatenated with
their columns interleaved in blocks of half a warpgroup tile (tile_n // 2 gate columns, then the same features' up
columns), so every warpgroup tile holds matching gate and up columns. Outputs: gu [M, 2F] bf16 (the GEMM in that
interleaved layout, saved for the backward pass) and h [M, F] bf16, computed from the bf16-rounded gate and up as the XLA
path does. Built on jax.experimental.pallas.ops.gpu.hopper_matmul_mgpu like levanter's lse_logits_mgpu.

Bench (python swiglu_mgpu.py <config name> [iters]): the 300m shape per device (M 131072, K 768, F 2688), this kernel
against two cuBLAS GEMMs plus XLA's product (the production forward), with the outputs checked against that reference."""
import dataclasses
import functools
import sys
import time

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hm

CONFIGS = {
    "m128wgm4": hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4, epi_tile_n=64, epi_tile_m=64,
                                grid_minor_dim=hm.MatmulDimension.N, grid_tile_width=16, wg_dimension=hm.MatmulDimension.M),
    "m128wgm3": hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=3, epi_tile_n=64, epi_tile_m=64,
                                grid_minor_dim=hm.MatmulDimension.N, grid_tile_width=16, wg_dimension=hm.MatmulDimension.M),
    "m128wgn4": hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4, epi_tile_n=64, epi_tile_m=64,
                                grid_minor_dim=hm.MatmulDimension.M, grid_tile_width=16, wg_dimension=hm.MatmulDimension.N),
    "m128wgn3": hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=3, epi_tile_n=64, epi_tile_m=64,
                                grid_minor_dim=hm.MatmulDimension.M, grid_tile_width=16, wg_dimension=hm.MatmulDimension.N),
}


def interleave(wg: jax.Array, wu: jax.Array, block: int) -> jax.Array:
    """[K, F] gate and up weights -> [K, 2F] with columns [g_0 | u_0 | g_1 | u_1 ...] in blocks of ``block``."""
    k, f = wg.shape
    return jnp.stack([wg.reshape(k, f // block, block), wu.reshape(k, f // block, block)], axis=2).reshape(k, 2 * f)


def deinterleave(gu: jax.Array, block: int) -> tuple[jax.Array, jax.Array]:
    """Inverse of ``interleave`` along the last axis."""
    *lead, two_f = gu.shape
    x = gu.reshape(*lead, two_f // (2 * block), 2, block)
    return x[..., 0, :].reshape(*lead, two_f // 2), x[..., 1, :].reshape(*lead, two_f // 2)


def _kernel(a_gmem, b_gmem, gu_gmem, h_gmem, *, config: hm.TuningConfig):
    dtype = a_gmem.dtype
    m, k = a_gmem.shape
    _, n = b_gmem.shape
    tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
    half = tile_n // 2
    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))
    wg_m = config.wg_dimension == hm.MatmulDimension.M
    cta_tile_m = tile_m * (2 if wg_m else 1)
    cta_tile_n = tile_n * (1 if wg_m else 2)
    m_iters, n_iters, k_iters = m // cta_tile_m, n // cta_tile_n, k // tile_k
    epi_tile_m, epi_tile_n = config.epi_tile_m, config.epi_tile_n
    assert half % epi_tile_n == 0, (half, epi_tile_n)
    out_swizzle = plgpu.find_swizzle(epi_tile_n * jnp.dtype(dtype).itemsize * 8)
    out_swizzle_elems = out_swizzle // jnp.dtype(dtype).itemsize
    out_transforms = (plgpu.TilingTransform((8, out_swizzle_elems)), plgpu.SwizzleTransform(out_swizzle))

    def get_pipeline(pipeline_body, compute_context):
        return plgpu.emit_pipeline_warp_specialized(
            pipeline_body,
            grid=(k_iters,),
            memory_registers=40,
            in_specs=[
                plgpu.BlockSpec((cta_tile_m, tile_k), lambda kk: (0, kk), transforms=transforms, memory_space=plgpu.SMEM),
                plgpu.BlockSpec((tile_k, cta_tile_n), lambda kk: (kk, 0), transforms=transforms, memory_space=plgpu.SMEM),
            ],
            wg_axis="wg",
            num_compute_wgs=2,
            max_concurrent_steps=config.max_concurrent_steps,
            compute_context=compute_context,
        )

    ignore = lambda *_, **__: None  # noqa: E731

    @functools.partial(
        pl.run_scoped,
        pipeline_allocs=get_pipeline(ignore, ignore).get_allocations(a_gmem, b_gmem),
        out_smem=plgpu.SMEM((2, 2, epi_tile_m, epi_tile_n), dtype, transforms=out_transforms),
        collective_axes="wg",
    )
    def _pipeline_scope(pipeline_allocs, out_smem):
        wg_idx = lax.axis_index("wg")

        @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
        def _mn_loop(loop_info: plgpu.NDLoopInfo):
            (lin_idx,) = loop_info.index
            m_idx, n_idx = plgpu.planar_snake(lin_idx, (m_iters, n_iters), config.grid_minor_dim, config.grid_tile_width)
            cta_m_slice = pl.ds(m_idx * cta_tile_m, cta_tile_m)
            cta_n_slice = pl.ds(n_idx * cta_tile_n, cta_tile_n)
            if wg_m:
                wg_m_slice, wg_n_slice = pl.ds(wg_idx * tile_m, tile_m), slice(None)
                n_tile = n_idx
            else:
                wg_m_slice, wg_n_slice = slice(None), pl.ds(wg_idx * tile_n, tile_n)
                n_tile = n_idx * 2 + wg_idx
            h_tile = h_gmem.at[cta_m_slice, pl.ds(n_tile * half, half)].at[wg_m_slice, slice(None)]
            gu_tile = gu_gmem.at[cta_m_slice, cta_n_slice].at[wg_m_slice, wg_n_slice]

            def compute_context(eval_pipeline):
                @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((tile_m, tile_n), jnp.float32))
                def _acc_scope(acc_ref):
                    eval_pipeline(acc_ref)
                    acc = acc_ref[...].astype(dtype)
                    i = 0

                    def store(val, dst, i):
                        slot = i % 2
                        plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                        out_smem[wg_idx, slot] = val
                        plgpu.commit_smem()
                        plgpu.copy_smem_to_gmem(out_smem.at[wg_idx, slot], dst)

                    for epi_mi in range(tile_m // epi_tile_m):
                        ms = slice(epi_mi * epi_tile_m, (epi_mi + 1) * epi_tile_m)
                        for epi_ni in range(tile_n // epi_tile_n):
                            ns = slice(epi_ni * epi_tile_n, (epi_ni + 1) * epi_tile_n)
                            store(acc[ms, ns], gu_tile.at[ms, ns], i)
                            i += 1
                        for epi_ni in range(half // epi_tile_n):
                            ns = slice(epi_ni * epi_tile_n, (epi_ni + 1) * epi_tile_n)
                            ns_u = slice(half + epi_ni * epi_tile_n, half + (epi_ni + 1) * epi_tile_n)
                            g = acc[ms, ns].astype(jnp.float32)
                            u = acc[ms, ns_u].astype(jnp.float32)
                            store((g / (1.0 + jnp.exp(-g)) * u).astype(dtype), h_tile.at[ms, ns], i)
                            i += 1

            def mma_body(idxs, a_smem, b_smem, acc_ref):
                plgpu.wgmma(acc_ref, a_smem.at[wg_m_slice], b_smem.at[:, wg_n_slice])
                plgpu.wgmma_wait(0)
                return acc_ref

            get_pipeline(mma_body, compute_context)(
                a_gmem.at[cta_m_slice, :], b_gmem.at[:, cta_n_slice], allocations=pipeline_allocs
            )

    plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def swiglu_fwd_mgpu(x: jax.Array, w_cat: jax.Array, config: hm.TuningConfig):
    """gu = x @ w_cat [M, 2F] (interleaved in blocks of tile_n // 2) and h = silu(g) * u [M, F], both bf16."""
    m_dim, k_dim = x.shape
    _, n_dim = w_cat.shape
    return plgpu.kernel(
        functools.partial(_kernel, config=config),
        out_type=(jax.ShapeDtypeStruct((m_dim, n_dim), jnp.bfloat16), jax.ShapeDtypeStruct((m_dim, n_dim // 2), jnp.bfloat16)),
        grid=(hm.get_num_sms(),),
        grid_names=("cluster_grid",),
        cluster=(1,),
        cluster_names=("cluster",),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True, lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )(x, w_cat)


def _time(f, *args, iters):
    for _ in range(3):
        out = f(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts) * 1e3)


def main():
    name = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    cfg = CONFIGS[name]
    M, K, F = 32 * 4096, 768, 2688
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(k[0], (M, K), jnp.float32).astype(jnp.bfloat16)
    wg = (jax.random.normal(k[1], (K, F), jnp.float32) * K ** -0.5).astype(jnp.bfloat16)
    wu = (jax.random.normal(k[2], (K, F), jnp.float32) * K ** -0.5).astype(jnp.bfloat16)
    half = cfg.tile_n // 2

    @jax.jit
    def ref(x, wg, wu):
        g, u = x @ wg, x @ wu
        return g, u, jax.nn.silu(g) * u

    @jax.jit
    def ref_gemms(x, wg, wu):
        return x @ wg, x @ wu

    w_cat = interleave(wg, wu, half)
    fused = jax.jit(functools.partial(swiglu_fwd_mgpu, config=cfg))
    (g0, u0, h0), t_ref = _time(ref, x, wg, wu, iters=iters)
    _, t_gemms = _time(ref_gemms, x, wg, wu, iters=iters)
    (gu1, h1), t_fused = _time(fused, x, w_cat, iters=iters)
    g1, u1 = deinterleave(gu1, half)
    err = lambda a, b: float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))) /
                             jnp.max(jnp.abs(b.astype(jnp.float32))))  # noqa: E731
    flops = 2 * M * K * 2 * F
    print(f"SWIGLU {name}: cuBLAS gate+up {t_gemms:.3f} ms ({flops / t_gemms / 1e9:.0f} TFLOP/s), + XLA product "
          f"{t_ref:.3f} ms; fused {t_fused:.3f} ms ({flops / t_fused / 1e9:.0f} TFLOP/s); saving per layer "
          f"{t_ref - t_fused:+.3f} ms; max rel err gate {err(g1, g0):.2e} up {err(u1, u0):.2e} h {err(h1, h0):.2e}", flush=True)


if __name__ == "__main__":
    print(f"devices {jax.devices()}", flush=True)
    main()
