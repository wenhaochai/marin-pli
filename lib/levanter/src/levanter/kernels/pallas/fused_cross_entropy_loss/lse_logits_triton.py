"""Logits GEMM with logsumexp partials in its epilogue, for the b-tiled fused cross-entropy on H100 (Triton through
jax_triton; speed work 2026-10-26, for the pls session to review and integrate).

``lse_and_logits(x, w)`` returns ``logits = x @ w`` in f32 (bf16 inputs, f32 accumulation, as the XLA GEMM with
``preferred_element_type=jnp.float32``) and ``lse = logsumexp(logits, axis=-1)``. The epilogue of each
[BLOCK_M, BLOCK_N] output tile writes the tile and, from the same f32 accumulators, the row max ``m`` of the tile and the
row sum ``s = sum(exp(z - m))``. A small XLA op merges the partials: ``lse = M + log(sum_t s_t exp(m_t - M))``, with
``M`` the max over the tiles. This replaces the two reductions over the f32 logits (the max, then the sum of exp) that
``_cross_entropy_logsumexp`` runs after the GEMM in ``_full_vocab_b_tiled_forward_with_grads``.

With ``labels`` and ``out_dtype=jnp.bfloat16`` (v2, the target): the tile is written in bf16 (round to nearest even, the
value ``reduce_precision(logits_f32, bf16)`` gives in the GIF path) and the f32 logit at each row's label comes back as a
third output (the tile that holds the label writes it), so the forward loss keeps f32 semantics while the logits write
and the GIF dlogits read are halved. Labels must lie in [0, N).

Partials are stored [num_n_tiles, M] so each tile writes contiguous rows. Within a tile the sum is the exact two-pass
one; ``accurate_exp`` uses libdevice's expf (the default; ``tl.exp`` is ex2.approx after a multiply by log2(e)).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _epilogue(acc, pid_m, pid_n, labels_ptr, out_ptr, m_ptr, s_ptr, lab_ptr,
              M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, ACCURATE_EXP: tl.constexpr,
              OUT_BF16: tl.constexpr, HAS_LABELS: tl.constexpr):
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EVEN_N:
        z = acc
    else:
        z = tl.where((offs_n < N)[None, :], acc, float("-inf"))
    m = tl.max(z, axis=1)
    if ACCURATE_EXP:
        p = libdevice.exp(z - m[:, None])
    else:
        p = tl.exp(z - m[:, None])
    s = tl.sum(p, axis=1)
    if OUT_BF16:
        val = acc.to(tl.bfloat16)
    else:
        val = acc
    out_ptrs = out_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :]
    row_ok = offs_m < M
    if EVEN_M:
        if EVEN_N:
            tl.store(out_ptrs, val)
        else:
            tl.store(out_ptrs, val, mask=(offs_n < N)[None, :])
        tl.store(m_ptr + pid_n * M + offs_m, m)
        tl.store(s_ptr + pid_n * M + offs_m, s)
    else:
        tl.store(out_ptrs, val, mask=row_ok[:, None] & (offs_n < N)[None, :])
        tl.store(m_ptr + pid_n * M + offs_m, m, mask=row_ok)
        tl.store(s_ptr + pid_n * M + offs_m, s, mask=row_ok)
    if HAS_LABELS:
        # the f32 logit at the label: exactly one tile per row holds it; x + 0.0 is exact, so the sum is that logit
        lab = tl.load(labels_ptr + offs_m, mask=row_ok, other=-1)
        z_lab = tl.sum(tl.where(offs_n[None, :] == lab[:, None], acc, 0.0), axis=1)
        in_tile = (lab >= pid_n * BLOCK_N) & (lab < pid_n * BLOCK_N + BLOCK_N) & row_ok
        tl.store(lab_ptr + offs_m, z_lab, mask=in_tile)


@triton.jit
def _tile_offsets(pid_m, pid_n, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    # rows and columns past the end read row or column 0 (in bounds); the epilogue masks them out
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if not EVEN_M:
        offs_am = tl.where(offs_am < M, offs_am, 0)
    if not EVEN_N:
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    return offs_am, offs_bn


@triton.jit
def _lse_logits_kernel(x_ptr, w_ptr, labels_ptr, out_ptr, m_ptr, s_ptr, lab_ptr,
                       M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, ACCURATE_EXP: tl.constexpr,
                       OUT_BF16: tl.constexpr, HAS_LABELS: tl.constexpr):
    """One program per output tile. Programs run down a column of tiles first: consecutive programs read the same
    [K, BLOCK_N] slice of w, so w comes from HBM about once, and x ([M, K], 12.6 MB at 8192 x 768) stays in L2."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am, offs_bn = _tile_offsets(pid_m, pid_n, M, N, BLOCK_M, BLOCK_N, EVEN_M, EVEN_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_am[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_k[:, None] * N + offs_bn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_K):
        acc = tl.dot(tl.load(x_ptrs), tl.load(w_ptrs), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * N
    _epilogue(acc, pid_m, pid_n, labels_ptr, out_ptr, m_ptr, s_ptr, lab_ptr, M, N, BLOCK_M, BLOCK_N, EVEN_M, EVEN_N,
              ACCURATE_EXP, OUT_BF16, HAS_LABELS)


@triton.jit
def _lse_logits_persistent_kernel(x_ptr, w_ptr, labels_ptr, out_ptr, m_ptr, s_ptr, lab_ptr,
                                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                  EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, ACCURATE_EXP: tl.constexpr,
                                  OUT_BF16: tl.constexpr, HAS_LABELS: tl.constexpr, NUM_PROGRAMS: tl.constexpr):
    """NUM_PROGRAMS programs loop over the tiles in the same order; the flattened loop lets the compiler prefetch the
    next tile's first K blocks while this tile's epilogue runs (Triton's persistent matmul tutorial)."""
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_tiles = num_pid_m * tl.cdiv(N, BLOCK_N)
    for tile_id in tl.range(start_pid, num_tiles, NUM_PROGRAMS, flatten=True):
        pid_m = tile_id % num_pid_m
        pid_n = tile_id // num_pid_m
        offs_am, offs_bn = _tile_offsets(pid_m, pid_n, M, N, BLOCK_M, BLOCK_N, EVEN_M, EVEN_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(0, K // BLOCK_K):
            offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(x_ptr + offs_am[:, None] * K + offs_k[None, :])
            b = tl.load(w_ptr + offs_k[:, None] * N + offs_bn[None, :])
            acc = tl.dot(a, b, acc)
        _epilogue(acc, pid_m, pid_n, labels_ptr, out_ptr, m_ptr, s_ptr, lab_ptr, M, N, BLOCK_M, BLOCK_N, EVEN_M,
                  EVEN_N, ACCURATE_EXP, OUT_BF16, HAS_LABELS)


@functools.cache
def _num_sms() -> int:
    return int(getattr(jax.devices()[0], "core_count", 0) or 132)


def merge_lse_partials(m_part: jax.Array, s_part: jax.Array) -> jax.Array:
    """lse per row from the per-tile partials [num_n_tiles, M]: M + log(sum_t s_t exp(m_t - M))."""
    m_max = jnp.max(m_part, axis=0)
    return m_max + jnp.log(jnp.sum(s_part * jnp.exp(m_part - m_max[None, :]), axis=0))


def lse_and_logits(
    x: jax.Array,
    w: jax.Array,
    labels: jax.Array | None = None,
    *,
    out_dtype: jnp.dtype = jnp.float32,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int = 8,
    num_stages: int = 4,
    persistent: bool = False,
    ctas_per_sm: int = 1,
    accurate_exp: bool = True,
) -> tuple[jax.Array, ...]:
    """Logits ``x @ w`` [M, N] in ``out_dtype`` (f32 accumulation) and their logsumexp per row [M] (f32), for bf16 ``x``
    [M, K] and ``w`` [K, N] on an H100; with ``labels`` [M] (int, in [0, N)) also the f32 logit at each label [M]."""
    m_dim, k_dim = x.shape
    k_w, n_dim = w.shape
    if x.dtype != jnp.bfloat16 or w.dtype != jnp.bfloat16:
        raise ValueError(f"lse_and_logits takes bf16 x and w, got {x.dtype} and {w.dtype}")
    if k_w != k_dim or k_dim % block_k:
        raise ValueError(f"x {x.shape} and w {w.shape}: K must match and be a multiple of block_k={block_k}")
    out_dtype = jnp.dtype(out_dtype)
    if out_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16)):
        raise ValueError(f"out_dtype {out_dtype}: float32 or bfloat16")
    has_labels = labels is not None
    labels_in = labels.astype(jnp.int32) if has_labels else jnp.zeros((1,), jnp.int32)
    n_tiles = triton.cdiv(n_dim, block_n)
    out_shape = (
        jax.ShapeDtypeStruct((m_dim, n_dim), out_dtype),
        jax.ShapeDtypeStruct((n_tiles, m_dim), jnp.float32),
        jax.ShapeDtypeStruct((n_tiles, m_dim), jnp.float32),
        jax.ShapeDtypeStruct((m_dim if has_labels else 1,), jnp.float32),
    )
    tiles = triton.cdiv(m_dim, block_m) * n_tiles
    meta = dict(M=m_dim, N=n_dim, K=k_dim, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                EVEN_M=m_dim % block_m == 0, EVEN_N=n_dim % block_n == 0, ACCURATE_EXP=accurate_exp,
                OUT_BF16=bool(out_dtype == jnp.dtype(jnp.bfloat16)), HAS_LABELS=has_labels)
    if persistent:
        programs = min(tiles, _num_sms() * ctas_per_sm)
        logits, m_part, s_part, label_logits = jt.triton_call(
            x, w, labels_in, kernel=_lse_logits_persistent_kernel, out_shape=out_shape, grid=(programs,),
            num_warps=num_warps, num_stages=num_stages, NUM_PROGRAMS=programs, **meta)
    else:
        logits, m_part, s_part, label_logits = jt.triton_call(
            x, w, labels_in, kernel=_lse_logits_kernel, out_shape=out_shape, grid=(tiles,),
            num_warps=num_warps, num_stages=num_stages, **meta)
    lse = merge_lse_partials(m_part, s_part)
    return (logits, lse, label_logits) if has_labels else (logits, lse)
