# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-head RMS norm and rotary embedding of q or k in one pass, forward and backward, as Pallas Triton kernels (speed work
2026-10-27; Attention._compute_qkv uses it under LEVANTER_QK_ROPE_FUSED=1). On the XLA path the norm is one fusion and the
rotary another (rotate_half is a concatenate, which the norm's reduction fusion does not take), and the rotary's cos and
sin are broadcast to the full [batch, position, heads, 64] shape outside the layer loop, so every pass reads two
activation-sized tables: 300m at 32 x 4096 per device spends ~15 ms per step on these passes forward and in the remat
recompute, ~11 ms backward (profile w476). Here one kernel reads x once, reads cos and sin as [positions, 32] tables, and
writes the result once; the backward recomputes the forward from x and writes dx and per-program partial sums of the norm
weight's gradient.

x is [rows, heads, 2, 32]: each 64-wide head split into the halves rotate_half swaps, rows = tokens in (batch, position)
order, so row r has position r % S. The grid covers rows in blocks of ROWS (dividing S) and heads in groups of HEADS, so
every block dimension is a power of two and divides its array. Math in f32 from the bf16 inputs, with the norm's output
and the weighted norm rounded to bf16 as haliax.nn.RmsNorm does (in_dtype, then a bf16 weight multiply); the rotary is
computed in f32 and rounded once (the XLA path rounds after each of its three ops)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

ROWS, HEADS = 16, 4


def _norm_rope(x1, x2, w1, w2, c, s, eps):
    """f32 halves x1, x2 [R, H, 32] -> (n1, n2, z1, z2, out1, out2, inv); w1, w2 [32]; c, s [R, 1, 32]."""
    var = (jnp.sum(x1 * x1, axis=-1, keepdims=True) + jnp.sum(x2 * x2, axis=-1, keepdims=True)) / 64.0
    inv = jax.lax.rsqrt(var + eps)
    n1 = (x1 * inv).astype(jnp.bfloat16).astype(jnp.float32)
    n2 = (x2 * inv).astype(jnp.bfloat16).astype(jnp.float32)
    z1 = (w1 * n1).astype(jnp.bfloat16).astype(jnp.float32)
    z2 = (w2 * n2).astype(jnp.bfloat16).astype(jnp.float32)
    return n1, n2, z1, z2, z1 * c - z2 * s, z2 * c + z1 * s, inv


def _fwd_kernel(x_ref, w_ref, cos_ref, sin_ref, o_ref, *, eps):
    x1 = x_ref[:, :, 0, :].astype(jnp.float32)
    x2 = x_ref[:, :, 1, :].astype(jnp.float32)
    w1 = w_ref[0, :].astype(jnp.float32)
    w2 = w_ref[1, :].astype(jnp.float32)
    c = cos_ref[...].astype(jnp.float32)[:, None, :]
    s = sin_ref[...].astype(jnp.float32)[:, None, :]
    *_, out1, out2, _ = _norm_rope(x1, x2, w1, w2, c, s, eps)
    o_ref[:, :, 0, :] = out1.astype(o_ref.dtype)
    o_ref[:, :, 1, :] = out2.astype(o_ref.dtype)


def _bwd_kernel(x_ref, w_ref, cos_ref, sin_ref, dy_ref, dx_ref, dw_ref, *, eps):
    x1 = x_ref[:, :, 0, :].astype(jnp.float32)
    x2 = x_ref[:, :, 1, :].astype(jnp.float32)
    w1 = w_ref[0, :].astype(jnp.float32)
    w2 = w_ref[1, :].astype(jnp.float32)
    c = cos_ref[...].astype(jnp.float32)[:, None, :]
    s = sin_ref[...].astype(jnp.float32)[:, None, :]
    n1, n2, _, _, _, _, inv = _norm_rope(x1, x2, w1, w2, c, s, eps)
    dy1 = dy_ref[:, :, 0, :].astype(jnp.float32)
    dy2 = dy_ref[:, :, 1, :].astype(jnp.float32)
    # the rotary's transpose, then the weight multiply's, then the norm's
    dz1 = dy1 * c + dy2 * s
    dz2 = dy2 * c - dy1 * s
    dw_ref[0, 0, 0, :] = jnp.sum(jnp.sum(dz1 * n1, axis=0), axis=0)
    dw_ref[0, 0, 1, :] = jnp.sum(jnp.sum(dz2 * n2, axis=0), axis=0)
    dn1, dn2 = dz1 * w1, dz2 * w2
    m1, m2 = x1 * inv, x2 * inv
    mean_dn_n = (jnp.sum(dn1 * m1, axis=-1, keepdims=True) + jnp.sum(dn2 * m2, axis=-1, keepdims=True)) / 64.0
    dx_ref[:, :, 0, :] = (inv * (dn1 - m1 * mean_dn_n)).astype(dx_ref.dtype)
    dx_ref[:, :, 1, :] = (inv * (dn2 - m2 * mean_dn_n)).astype(dx_ref.dtype)


def _specs(n_rows, heads, seq_len):
    x_spec = pl.BlockSpec((ROWS, HEADS, 2, 32), lambda i, j: (i, j, 0, 0))
    w_spec = pl.BlockSpec((2, 32), lambda i, j: (0, 0))
    t_spec = pl.BlockSpec((ROWS, 32), lambda i, j: (i % (seq_len // ROWS), 0))
    return x_spec, w_spec, t_spec, (n_rows // ROWS, heads // HEADS)


def supported(n_rows: int, heads: int, head_dim: int, seq_len: int) -> bool:
    return head_dim == 64 and heads % HEADS == 0 and seq_len % ROWS == 0 and n_rows % seq_len == 0


def rope_tables(seq_len: int, theta: float, factor: float = 1.0, dtype=jnp.bfloat16):
    """cos and sin [seq_len, 32] of positions 0 .. seq_len - 1, as levanter's DefaultRotaryEmbeddings forms them."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, 64, 2, dtype=jnp.float32) / 64.0)) / factor
    freqs = jnp.arange(seq_len, dtype=jnp.float32)[:, None] * inv_freq[None, :]
    return jnp.cos(freqs).astype(dtype), jnp.sin(freqs).astype(dtype)


def reference(x, w, cos, sin, eps):
    """The same computation in jnp on [rows, heads, 2, 32] (tests, and the path off the GPU)."""
    seq_len = cos.shape[0]
    pos = jnp.arange(x.shape[0]) % seq_len
    c = cos[pos].astype(jnp.float32)[:, None, :]
    s = sin[pos].astype(jnp.float32)[:, None, :]
    w2 = w.reshape(2, 32).astype(jnp.float32)
    *_, out1, out2, _ = _norm_rope(x[:, :, 0, :].astype(jnp.float32), x[:, :, 1, :].astype(jnp.float32), w2[0], w2[1], c, s, eps)
    return jnp.stack([out1, out2], axis=2).astype(x.dtype)


def _fwd_call(x, w, cos, sin, eps, interpret):
    n_rows, heads = x.shape[0], x.shape[1]
    x_spec, w_spec, t_spec, grid = _specs(n_rows, heads, cos.shape[0])
    return pl.pallas_call(
        functools.partial(_fwd_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=[x_spec, w_spec, t_spec, t_spec],
        out_specs=x_spec,
        compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=2),
        interpret=interpret,
        name="qk_norm_rope_fwd",
    )(x, w.reshape(2, 32), cos, sin)


def _bwd_call(x, w, cos, sin, dy, eps, interpret):
    n_rows, heads = x.shape[0], x.shape[1]
    x_spec, w_spec, t_spec, grid = _specs(n_rows, heads, cos.shape[0])
    dw_spec = pl.BlockSpec((1, 1, 2, 32), lambda i, j: (i, j, 0, 0))
    dx, dw = pl.pallas_call(
        functools.partial(_bwd_kernel, eps=eps),
        out_shape=(jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct((grid[0], grid[1], 2, 32), jnp.float32)),
        grid=grid,
        in_specs=[x_spec, w_spec, t_spec, t_spec, x_spec],
        out_specs=(x_spec, dw_spec),
        compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=2),
        interpret=interpret,
        name="qk_norm_rope_bwd",
    )(x, w.reshape(2, 32), cos, sin, dy)
    return dx, jnp.sum(dw, axis=(0, 1)).reshape(64)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def qk_norm_rope(x, w, cos, sin, eps: float, interpret: bool = False):
    """RmsNorm over each 64-wide head (weight w [64]) then the rotary embedding, x [rows, heads, 2, 32] bf16 with row r at
    position r % len(cos); cos, sin [positions, 32] from rope_tables. Gradients for x and w (cos and sin are constants)."""
    return _fwd_call(x, w, cos, sin, eps, interpret)


def _qk_norm_rope_fwd(x, w, cos, sin, eps, interpret):
    return _fwd_call(x, w, cos, sin, eps, interpret), (x, w, cos, sin)


def _qk_norm_rope_bwd(eps, interpret, residuals, dy):
    x, w, cos, sin = residuals
    dx, dw = _bwd_call(x, w, cos, sin, dy, eps, interpret)
    return dx, dw.astype(w.dtype), jnp.zeros_like(cos), jnp.zeros_like(sin)


qk_norm_rope.defvjp(_qk_norm_rope_fwd, _qk_norm_rope_bwd)
