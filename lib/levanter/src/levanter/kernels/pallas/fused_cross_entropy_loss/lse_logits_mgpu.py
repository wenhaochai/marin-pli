"""Logits GEMM with logsumexp partials in its epilogue on H100, in JAX's Pallas Mosaic GPU (lse kernel v3, speed work
2026-10-26 with the pls session): the same outputs as ``lse_logits_triton.lse_and_logits`` (bf16 logits, f32 logsumexp,
f32 label logit) from a persistent, warp-specialized TMA + wgmma GEMM (jax.experimental.pallas.ops.gpu.hopper_matmul_mgpu:
one memory warpgroup and two compute warpgroups). On one H100 at the b-tiled CE block (8192 rows, 128256 columns) the plain
GEMM runs at 2.44 ms for K 768 and 1.83 ms for K 512 with 3 pipeline stages, against 3.09 and 2.46 ms for the Triton kernel and 2.15 and 1.61 ms
for cuBLAS (wq w356).

The epilogue of each warpgroup's [tile_m, tile_n] tile takes the f32 accumulator before the bf16 store and writes the row
max ``m`` and ``s = sum(exp(z - m))`` of the tile to [N // tile_n, M] partials, which ``merge_lse_partials`` combines as in
the Triton kernel. The f32 logit at each row's label is an XLA gather-dot outside the kernel: the same products as the
GEMM, summed in another order (~1e-7 relative), and it enters only the loss value, not the gradients. Shapes: M divisible
by the CTA's rows, N by the CTA's columns (128256 and 64000 by 128 and 256), K by 64.
"""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hm


def merge_lse_partials(m_part: jax.Array, s_part: jax.Array) -> jax.Array:
    """lse per row from the per-tile partials [num_n_tiles, M]: M + log(sum_t s_t exp(m_t - M)) (as the Triton module's;
    copied, not imported, because importing jax_triton resets TRITON_CACHE_DIR)."""
    m_max = jnp.max(m_part, axis=0)
    return m_max + jnp.log(jnp.sum(s_part * jnp.exp(m_part - m_max[None, :]), axis=0))

# Fastest plain-GEMM configs of wq w356 (one H100, M 8192, N 128256) with 3 pipeline stages, by K: with 4 stages the
# stats buffer of the epilogue passes the 227 KB of shared memory by 1 KB (w366). The 4-stage plain GEMM was 1.3% (K 768)
# and 4% (K 512) faster.
CONFIGS = {
    768: hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=3, epi_tile_n=64, epi_tile_m=64,
                         grid_minor_dim=hm.MatmulDimension.N, grid_tile_width=16, wg_dimension=hm.MatmulDimension.M),
    512: hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=3, epi_tile_n=64, epi_tile_m=64,
                         grid_minor_dim=hm.MatmulDimension.M, grid_tile_width=16, wg_dimension=hm.MatmulDimension.N),
}


def _kernel(a_gmem, b_gmem, *refs, config: hm.TuningConfig, exp_out: bool = False):
    """hopper_matmul_mgpu.kernel without the C operand, clusters or callbacks, plus the logsumexp partials. With exp_out
    the kernel also reads a row offset c [M] and stores exp(z - c) instead of z (see probs_and_sum_mgpu)."""
    if exp_out:
        c_gmem, out_gmem, m_gmem, s_gmem = refs
    else:
        out_gmem, m_gmem, s_gmem = refs
    dtype = a_gmem.dtype
    out_dtype = out_gmem.dtype
    m, k = a_gmem.shape
    _, n = b_gmem.shape
    tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
    swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
    transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))
    wg_m = config.wg_dimension == hm.MatmulDimension.M
    cta_tile_m = tile_m * (2 if wg_m else 1)
    cta_tile_n = tile_n * (1 if wg_m else 2)
    m_iters, n_iters, k_iters = m // cta_tile_m, n // cta_tile_n, k // tile_k
    epi_tile_m, epi_tile_n = config.epi_tile_m, config.epi_tile_n
    num_out_slots = min(2, (tile_m * tile_n) // (epi_tile_m * epi_tile_n))
    out_swizzle = plgpu.find_swizzle(epi_tile_n * jnp.dtype(out_dtype).itemsize * 8)
    out_swizzle_elems = out_swizzle // jnp.dtype(out_dtype).itemsize
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
        out_smem=plgpu.SMEM((2, num_out_slots, epi_tile_m, epi_tile_n), out_dtype, transforms=out_transforms),
        stats_smem=plgpu.SMEM((2, 2, tile_m), jnp.float32),
        collective_axes="wg",
    )
    def _pipeline_scope(pipeline_allocs, out_smem, stats_smem):
        wg_idx = lax.axis_index("wg")

        @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
        def _mn_loop(loop_info: plgpu.NDLoopInfo):
            (lin_idx,) = loop_info.index
            m_idx, n_idx = plgpu.planar_snake(lin_idx, (m_iters, n_iters), config.grid_minor_dim, config.grid_tile_width)
            cta_m_slice = pl.ds(m_idx * cta_tile_m, cta_tile_m)
            cta_n_slice = pl.ds(n_idx * cta_tile_n, cta_tile_n)
            if wg_m:
                wg_m_slice, wg_n_slice = pl.ds(wg_idx * tile_m, tile_m), slice(None)
                rows = pl.ds(m_idx * cta_tile_m + wg_idx * tile_m, tile_m)
                n_tile = n_idx
            else:
                wg_m_slice, wg_n_slice = slice(None), pl.ds(wg_idx * tile_n, tile_n)
                rows = pl.ds(m_idx * cta_tile_m, tile_m)
                n_tile = n_idx * 2 + wg_idx

            def compute_context(eval_pipeline):
                @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((tile_m, tile_n), jnp.float32))
                def _acc_scope(acc_ref):
                    eval_pipeline(acc_ref)
                    acc32 = acc_ref[...]
                    if exp_out:
                        # exp(z - c) clamped at 60, so neither a stored value nor a GEMM sum over a row overflows f32;
                        # with c the label logit only a token whose loss passes 60 nats is changed
                        c = plgpu.load(c_gmem.at[rows], layout=plgpu.Layout.WGMMA.reduce(1))
                        shifted = jnp.minimum(acc32 - lax.broadcast_in_dim(c, acc32.shape, [0]), 60.0)
                        row_max = shifted.max(axis=1)
                        e = jnp.exp(shifted)
                        row_sum = e.sum(axis=1)
                        acc = e.astype(out_dtype)
                    else:
                        row_max = acc32.max(axis=1)
                        row_sum = jnp.exp(acc32 - lax.broadcast_in_dim(row_max, acc32.shape, [0])).sum(axis=1)
                        acc = acc32.astype(out_dtype)
                    # every earlier copy out of this warpgroup's SMEM (last tile's stats and logits) is done
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    stats_smem[wg_idx, 0] = row_max
                    stats_smem[wg_idx, 1] = row_sum
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(stats_smem.at[wg_idx, 0], m_gmem.at[n_tile, rows])
                    plgpu.copy_smem_to_gmem(stats_smem.at[wg_idx, 1], s_gmem.at[n_tile, rows])
                    for epi_mi in range(tile_m // epi_tile_m):
                        for epi_ni in range(tile_n // epi_tile_n):
                            epi_m_slice = slice(epi_mi * epi_tile_m, (epi_mi + 1) * epi_tile_m)
                            epi_n_slice = slice(epi_ni * epi_tile_n, (epi_ni + 1) * epi_tile_n)
                            slot = (epi_mi * (tile_n // epi_tile_n) + epi_ni) % 2
                            plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                            out_smem[wg_idx, slot] = acc[epi_m_slice, epi_n_slice]
                            plgpu.commit_smem()
                            plgpu.copy_smem_to_gmem(
                                out_smem.at[wg_idx, slot],
                                out_gmem.at[cta_m_slice, cta_n_slice].at[wg_m_slice, wg_n_slice].at[epi_m_slice, epi_n_slice],
                            )

            def mma_body(idxs, a_smem, b_smem, acc_ref):
                plgpu.wgmma(acc_ref, a_smem.at[wg_m_slice], b_smem.at[:, wg_n_slice])
                plgpu.wgmma_wait(0)
                return acc_ref

            get_pipeline(mma_body, compute_context)(
                a_gmem.at[cta_m_slice, :], b_gmem.at[:, cta_n_slice], allocations=pipeline_allocs
            )

    plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def supported(m: int, k: int, n: int) -> bool:
    """Whether lse_and_logits_mgpu takes these shapes (a config for K, and M, N divisible by its CTA tile)."""
    cfg = CONFIGS.get(k)
    if cfg is None:
        return False
    wg_m = cfg.wg_dimension == hm.MatmulDimension.M
    return m % (cfg.tile_m * (2 if wg_m else 1)) == 0 and n % (cfg.tile_n * (1 if wg_m else 2)) == 0


def lse_and_logits_mgpu(x: jax.Array, w: jax.Array, labels: jax.Array | None = None):
    """bf16 logits ``x @ w`` [M, N], f32 logsumexp per row [M] and, with ``labels`` [M] in [0, N), the f32 label logit;
    bf16 ``x`` [M, K] and ``w`` [K, N]; shapes must pass ``supported``."""
    m_dim, k_dim = x.shape
    _, n_dim = w.shape
    if x.dtype != jnp.bfloat16 or w.dtype != jnp.bfloat16:
        raise ValueError(f"lse_and_logits_mgpu takes bf16 x and w, got {x.dtype} and {w.dtype}")
    if not supported(m_dim, k_dim, n_dim):
        raise ValueError(f"lse_and_logits_mgpu: no config for x {x.shape}, w {w.shape}")
    cfg = CONFIGS[k_dim]
    n_tiles = n_dim // cfg.tile_n
    logits, m_part, s_part = plgpu.kernel(
        functools.partial(_kernel, config=cfg),
        out_type=(
            jax.ShapeDtypeStruct((m_dim, n_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((n_tiles, m_dim), jnp.float32),
            jax.ShapeDtypeStruct((n_tiles, m_dim), jnp.float32),
        ),
        grid=(hm.get_num_sms(),),
        grid_names=("cluster_grid",),
        cluster=(1,),
        cluster_names=("cluster",),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True, lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )(x, w)
    lse = merge_lse_partials(m_part, s_part)
    if labels is None:
        return logits, lse
    # each row's label row of w.T [N, K]: in training w is the [V, H] lm_head transposed (loss.py rearranges it to
    # (Embed, Vocab)), so XLA can cancel the two transposes and gather contiguous K-element rows; the column gather of
    # [K, N] took 0.27 ms per 8192-row block on its own (wq w379)
    w_rows = jnp.take(w.T, labels, axis=0)   # [M, K]
    label_logits = jnp.sum(x.astype(jnp.float32) * w_rows.astype(jnp.float32), axis=1)
    return logits, lse, label_logits


def probs_and_sum_mgpu(x: jax.Array, w: jax.Array, offset: jax.Array):
    """bf16 P = exp(x @ w - offset[:, None]) [M, N] (the shifted logits clamped at 60), the f32 sum of each row's
    exp(z - offset) [M] (from the f32 accumulators, before the bf16 rounding) and each row's largest shifted logit [M];
    bf16 ``x`` [M, K] and ``w`` [K, N], f32 ``offset`` [M]; shapes must pass ``supported``. With the label logit as the
    offset, P holds the probabilities times the row's sum (P at the label is 1), so the cross-entropy gradients are
    GEMMs on P with the row factor g / sum outside them, and no pass over the logits is needed between the GEMMs."""
    m_dim, k_dim = x.shape
    _, n_dim = w.shape
    if x.dtype != jnp.bfloat16 or w.dtype != jnp.bfloat16:
        raise ValueError(f"probs_and_sum_mgpu takes bf16 x and w, got {x.dtype} and {w.dtype}")
    if not supported(m_dim, k_dim, n_dim):
        raise ValueError(f"probs_and_sum_mgpu: no config for x {x.shape}, w {w.shape}")
    cfg = CONFIGS[k_dim]
    n_tiles = n_dim // cfg.tile_n
    probs, m_part, s_part = plgpu.kernel(
        functools.partial(_kernel, config=cfg, exp_out=True),
        out_type=(
            jax.ShapeDtypeStruct((m_dim, n_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((n_tiles, m_dim), jnp.float32),
            jax.ShapeDtypeStruct((n_tiles, m_dim), jnp.float32),
        ),
        grid=(hm.get_num_sms(),),
        grid_names=("cluster_grid",),
        cluster=(1,),
        cluster_names=("cluster",),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True, lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )(x, w, offset.astype(jnp.float32))
    return probs, jnp.sum(s_part, axis=0), jnp.max(m_part, axis=0)
