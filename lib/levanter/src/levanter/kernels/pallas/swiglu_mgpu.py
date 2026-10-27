# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU's gate and up projections and act(gate) * up as one H100 GEMM, in JAX's Pallas Mosaic GPU (speed work
2026-10-27; LlamaMlp uses it under LEVANTER_SWIGLU_FUSED=1). The gate and up weights are concatenated with their columns
interleaved in blocks of half a warpgroup tile (``block`` gate columns, then the same features' up columns), so every
warpgroup tile of the persistent, warp-specialized TMA + wgmma GEMM (jax.experimental.pallas.ops.gpu.hopper_matmul_mgpu,
as lse_logits_mgpu) holds matching gate and up columns, and its epilogue forms h = silu(gate) * up from the bf16-rounded
gate and up, as the XLA path does. Outputs: gu [M, 2F] bf16, the GEMM in the interleaved layout (what the backward pass
needs), and h [M, F] bf16. The separate elementwise pass of the XLA path, which reads gate and up back (1.4 GB per layer
at 300m, 32 x 4096 per device; 8.2 ms per 128K step, profile w458), goes away."""

from __future__ import annotations

import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hm

# Tile configs by name (wq w469 benches them at M 131072, K 768, F 2688); LEVANTER_SWIGLU_CFG / _BWD_CFG pick them.
CONFIGS = {
    "m128wgm4": hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4, epi_tile_n=64, epi_tile_m=64,
                                grid_minor_dim=hm.MatmulDimension.N, grid_tile_width=16, wg_dimension=hm.MatmulDimension.M),
    "m128wgm3": hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=3, epi_tile_n=64, epi_tile_m=64,
                                grid_minor_dim=hm.MatmulDimension.N, grid_tile_width=16, wg_dimension=hm.MatmulDimension.M),
    "m128wgn4": hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4, epi_tile_n=64, epi_tile_m=64,
                                grid_minor_dim=hm.MatmulDimension.M, grid_tile_width=16, wg_dimension=hm.MatmulDimension.N),
    "m128wgn3": hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=3, epi_tile_n=64, epi_tile_m=64,
                                grid_minor_dim=hm.MatmulDimension.M, grid_tile_width=16, wg_dimension=hm.MatmulDimension.N),
    # wgmma's native m64n256 per warpgroup (a 128 x 256 CTA, as cuBLAS's fastest Hopper GEMMs); interleave block 128
    "m64n256wgm4": hm.TuningConfig(tile_m=64, tile_n=256, tile_k=64, max_concurrent_steps=4, epi_tile_n=64, epi_tile_m=64,
                                   grid_minor_dim=hm.MatmulDimension.N, grid_tile_width=16, wg_dimension=hm.MatmulDimension.M),
    "m64n256wgm3": hm.TuningConfig(tile_m=64, tile_n=256, tile_k=64, max_concurrent_steps=3, epi_tile_n=64, epi_tile_m=64,
                                   grid_minor_dim=hm.MatmulDimension.N, grid_tile_width=16, wg_dimension=hm.MatmulDimension.M),
}
# forward: m128wgn4 was fastest (1.942 ms, w469); backward: m128wgm3, the one that fits (4 stages need 262 KB of SMEM, w472)
CONFIG = CONFIGS[os.environ.get("LEVANTER_SWIGLU_CFG", "m128wgn4")]
BWD_CONFIG = CONFIGS[os.environ.get("LEVANTER_SWIGLU_BWD_CFG", "m128wgm3")]
BLOCK = CONFIG.tile_n // 2
assert BWD_CONFIG.tile_n % BLOCK == 0, (CONFIG, BWD_CONFIG)


def interleave(wg: jax.Array, wu: jax.Array, block: int = BLOCK) -> jax.Array:
    """[K, F] gate and up weights -> [K, 2F] with columns [g_0 | u_0 | g_1 | u_1 ...] in blocks of ``block``."""
    k, f = wg.shape
    return jnp.stack([wg.reshape(k, f // block, block), wu.reshape(k, f // block, block)], axis=2).reshape(k, 2 * f)


def deinterleave(gu: jax.Array, block: int = BLOCK) -> tuple[jax.Array, jax.Array]:
    """Inverse of ``interleave`` along the last axis."""
    *lead, two_f = gu.shape
    x = gu.reshape(*lead, two_f // (2 * block), 2, block)
    return x[..., 0, :].reshape(*lead, two_f // 2), x[..., 1, :].reshape(*lead, two_f // 2)


def supported(m: int, k: int, two_f: int, config: hm.TuningConfig = CONFIG) -> bool:
    """Whether ``swiglu_fwd_mgpu`` takes these shapes: M, 2F divisible by the CTA tile, K by tile_k."""
    wg_m = config.wg_dimension == hm.MatmulDimension.M
    cta_m = config.tile_m * (2 if wg_m else 1)
    cta_n = config.tile_n * (1 if wg_m else 2)
    return m % cta_m == 0 and two_f % cta_n == 0 and k % config.tile_k == 0


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

                    def store(val, dst, i):
                        # two SMEM slots per warpgroup: wait until the copy that last used this slot has read it
                        plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                        out_smem[wg_idx, i % 2] = val
                        plgpu.commit_smem()
                        plgpu.copy_smem_to_gmem(out_smem.at[wg_idx, i % 2], dst)

                    i = 0
                    for epi_mi in range(tile_m // epi_tile_m):
                        ms = slice(epi_mi * epi_tile_m, (epi_mi + 1) * epi_tile_m)
                        for epi_ni in range(tile_n // epi_tile_n):
                            ns = slice(epi_ni * epi_tile_n, (epi_ni + 1) * epi_tile_n)
                            store(acc[ms, ns], gu_tile.at[ms, ns], i)
                            i += 1
                        for epi_ni in range(half // epi_tile_n):
                            ns = slice(epi_ni * epi_tile_n, (epi_ni + 1) * epi_tile_n)
                            g = acc[ms, ns].astype(jnp.float32)
                            u = acc[ms, slice(half + ns.start, half + ns.stop)].astype(jnp.float32)
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


def swiglu_fwd_mgpu(x: jax.Array, w_cat: jax.Array, config: hm.TuningConfig = CONFIG) -> tuple[jax.Array, jax.Array]:
    """gu = x @ w_cat [M, 2F] (columns interleaved in blocks of config.tile_n // 2) and h = silu(gate) * up [M, F]; bf16
    ``x`` [M, K] and ``w_cat`` [K, 2F]; shapes must pass ``supported``."""
    m_dim, k_dim = x.shape
    _, n_dim = w_cat.shape
    if x.dtype != jnp.bfloat16 or w_cat.dtype != jnp.bfloat16:
        raise ValueError(f"swiglu_fwd_mgpu takes bf16 x and w_cat, got {x.dtype} and {w_cat.dtype}")
    if not supported(m_dim, k_dim, n_dim, config):
        raise ValueError(f"swiglu_fwd_mgpu: no config for x {x.shape}, w_cat {w_cat.shape}")
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


def _bwd_kernel(a_gmem, b_gmem, gu_gmem, dgu_gmem, h_gmem, *, config: hm.TuningConfig):
    """d_h = d_out @ w_down.T as a GEMM whose epilogue reads the saved gate and up (interleaved gu) for its tile and
    writes the interleaved d_gate / d_up and h = silu(gate) * up (what down_proj's weight gradient needs), so d_h never
    reaches global memory. One warpgroup tile covers tile_n features = tile_n // BLOCK interleave blocks; the epilogue
    walks it in [epi_tile_m, BLOCK] chunks: two TMA loads (gate, up), then three stores (d_gate, d_up, h)."""
    dtype = a_gmem.dtype
    m, k = a_gmem.shape
    _, n = b_gmem.shape
    tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
    block = BLOCK
    assert tile_n % block == 0 and config.wg_dimension == hm.MatmulDimension.M, (tile_n, block, config.wg_dimension)
    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))
    cta_tile_m, cta_tile_n = tile_m * 2, tile_n
    m_iters, n_iters, k_iters = m // cta_tile_m, n // cta_tile_n, k // tile_k
    epi_tile_m = config.epi_tile_m
    chunk_swizzle = plgpu.find_swizzle(block * jnp.dtype(dtype).itemsize * 8)
    chunk_transforms = (plgpu.TilingTransform((8, chunk_swizzle // jnp.dtype(dtype).itemsize)), plgpu.SwizzleTransform(chunk_swizzle))

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
        in_smem=plgpu.SMEM((2, 2, epi_tile_m, block), dtype, transforms=chunk_transforms),
        out_smem=plgpu.SMEM((2, 2, epi_tile_m, block), dtype, transforms=chunk_transforms),
        load_barriers=plgpu.Barrier(num_barriers=4),
        collective_axes="wg",
    )
    def _pipeline_scope(pipeline_allocs, in_smem, out_smem, load_barriers):
        wg_idx = lax.axis_index("wg")

        @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
        def _mn_loop(loop_info: plgpu.NDLoopInfo):
            (lin_idx,) = loop_info.index
            m_idx, n_idx = plgpu.planar_snake(lin_idx, (m_iters, n_iters), config.grid_minor_dim, config.grid_tile_width)
            cta_m_slice = pl.ds(m_idx * cta_tile_m, cta_tile_m)
            cta_n_slice = pl.ds(n_idx * cta_tile_n, cta_tile_n)
            wg_m_slice = pl.ds(wg_idx * tile_m, tile_m)

            def compute_context(eval_pipeline):
                @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((tile_m, tile_n), jnp.float32))
                def _acc_scope(acc_ref):
                    eval_pipeline(acc_ref)
                    d_h_all = acc_ref[...].astype(dtype)   # the bf16 rounding of the XLA path's d_h
                    i = 0
                    for epi_mi in range(tile_m // epi_tile_m):
                        ms = slice(epi_mi * epi_tile_m, (epi_mi + 1) * epi_tile_m)
                        rows = pl.ds(m_idx * cta_tile_m + wg_idx * tile_m + epi_mi * epi_tile_m, epi_tile_m)
                        for blk in range(tile_n // block):
                            b_idx = n_idx * (tile_n // block) + blk   # interleave block: gu columns [2 b block, 2 (b + 1) block)
                            g_cols = pl.ds(b_idx * 2 * block, block)
                            u_cols = pl.ds(b_idx * 2 * block + block, block)
                            # every thread of this warpgroup is done reading the previous chunk before TMA overwrites it
                            plgpu.commit_smem()
                            plgpu.copy_gmem_to_smem(gu_gmem.at[rows, g_cols], in_smem.at[wg_idx, 0], load_barriers.at[2 * wg_idx])
                            plgpu.copy_gmem_to_smem(gu_gmem.at[rows, u_cols], in_smem.at[wg_idx, 1], load_barriers.at[2 * wg_idx + 1])
                            plgpu.barrier_wait(load_barriers.at[2 * wg_idx])
                            plgpu.barrier_wait(load_barriers.at[2 * wg_idx + 1])
                            g = plgpu.load(in_smem.at[wg_idx, 0], (), layout=plgpu.Layout.WGMMA).astype(jnp.float32)
                            u = plgpu.load(in_smem.at[wg_idx, 1], (), layout=plgpu.Layout.WGMMA).astype(jnp.float32)
                            d_h = d_h_all[ms, slice(blk * block, (blk + 1) * block)].astype(jnp.float32)
                            sig = 1.0 / (1.0 + jnp.exp(-g))
                            silu = g * sig
                            outs = (
                                ((d_h * u * (sig + silu * (1.0 - sig))).astype(dtype), dgu_gmem.at[rows, g_cols]),
                                ((d_h * silu).astype(dtype), dgu_gmem.at[rows, u_cols]),
                                ((silu * u).astype(dtype), h_gmem.at[rows, pl.ds(n_idx * tile_n + blk * block, block)]),
                            )
                            for val, dst in outs:
                                plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                                out_smem[wg_idx, i % 2] = val
                                plgpu.commit_smem()
                                plgpu.copy_smem_to_gmem(out_smem.at[wg_idx, i % 2], dst)
                                i += 1

            def mma_body(idxs, a_smem, b_smem, acc_ref):
                plgpu.wgmma(acc_ref, a_smem.at[wg_m_slice], b_smem)
                plgpu.wgmma_wait(0)
                return acc_ref

            get_pipeline(mma_body, compute_context)(
                a_gmem.at[cta_m_slice, :], b_gmem.at[:, cta_n_slice], allocations=pipeline_allocs
            )

    plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def bwd_supported(m: int, e: int, f: int, config: hm.TuningConfig = CONFIG) -> bool:
    """Whether ``swiglu_bwd_mgpu`` takes these shapes (a warpgroup-along-M config; M by 2 tile_m, F by tile_n, E by tile_k)."""
    return (config.wg_dimension == hm.MatmulDimension.M and m % (2 * config.tile_m) == 0
            and f % config.tile_n == 0 and e % config.tile_k == 0 and config.tile_n % BLOCK == 0)


def swiglu_bwd_mgpu(d_out: jax.Array, w_down_t: jax.Array, gu: jax.Array, config: hm.TuningConfig = CONFIG):
    """The interleaved d_gu [M, 2F] and h [M, F] from d_out [M, E], w_down.T [E, F] and the saved interleaved gu [M, 2F]
    (the gradients of silu(gate) * up through down_proj, h for down_proj's weight gradient); all bf16."""
    m_dim, e_dim = d_out.shape
    _, f_dim = w_down_t.shape
    if not bwd_supported(m_dim, e_dim, f_dim, config):
        raise ValueError(f"swiglu_bwd_mgpu: no config for d_out {d_out.shape}, w_down_t {w_down_t.shape}")
    return plgpu.kernel(
        functools.partial(_bwd_kernel, config=config),
        out_type=(jax.ShapeDtypeStruct((m_dim, 2 * f_dim), jnp.bfloat16), jax.ShapeDtypeStruct((m_dim, f_dim), jnp.bfloat16)),
        grid=(hm.get_num_sms(),),
        grid_names=("cluster_grid",),
        cluster=(1,),
        cluster_names=("cluster",),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True, lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )(d_out, w_down_t, gu)
