"""CPU check of levanter.kernels.pallas.qk_norm_rope (2026-10-27): the Pallas kernels in interpret mode against the jnp
reference (forward, and dx / dw against the reference's autodiff), and the reference against levanter's own path
(haliax RmsNorm over the head, then DefaultRotaryEmbeddings) on q shaped [batch, position, heads, head]."""
import jax
import jax.numpy as jnp
import numpy as np

import haliax as hax
from levanter.kernels.pallas import qk_norm_rope as K
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig

B, S, H, D, EPS, THETA = 2, 64, 8, 64, 1e-5, 10000.0
key = jax.random.split(jax.random.PRNGKey(0), 3)
x = (jax.random.normal(key[0], (B * S, H, 2, 32), jnp.float32) * 3.0).astype(jnp.bfloat16)
w = (1.0 + 0.1 * jax.random.normal(key[1], (D,), jnp.float32)).astype(jnp.bfloat16)
dy = jax.random.normal(key[2], x.shape, jnp.float32).astype(jnp.bfloat16)
cos, sin = K.rope_tables(S, THETA)


def rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


out_k = K.qk_norm_rope(x, w, cos, sin, EPS, True)
out_r = K.reference(x, w, cos, sin, EPS)
print(f"forward: kernel vs reference max rel {rel(out_k, out_r):.2e}")
_, vjp_k = jax.vjp(lambda a, b: K.qk_norm_rope(a, b, cos, sin, EPS, True), x, w)
_, vjp_r = jax.vjp(lambda a, b: K.reference(a, b, cos, sin, EPS), x, w)
(dx_k, dw_k), (dx_r, dw_r) = vjp_k(dy), vjp_r(dy)
print(f"backward: dx max rel {rel(dx_k, dx_r):.2e}, dw max rel {rel(dw_k, dw_r):.2e}")

# levanter's path on the same values: [batch, position, heads, head], halves = head[:32], head[32:]
Batch, Pos, Heads, Head = hax.Axis("batch", B), hax.Axis("position", S), hax.Axis("heads", H), hax.Axis("head_size", D)
q = hax.named(x.reshape(B, S, H, D), (Batch, Pos, Heads, Head))
norm = hax.nn.RmsNorm.init(Head, eps=EPS, use_weight=True, use_bias=False)
norm = hax.tree_util.tree_map(lambda a: a, norm)
norm = type(norm)(axis=norm.axis, weight=hax.named(w, (Head,)), bias=None, eps=EPS, dtype=jnp.float32)
rope = DefaultRotaryEmbeddingsConfig(theta=THETA).build(Head)
lev = rope(norm(q), hax.arange(Pos)).astype(q.dtype).array.reshape(B * S, H, 2, 32)
print(f"reference vs levanter path: max rel {rel(out_r, lev):.2e}, kernel vs levanter path {rel(out_k, lev):.2e}")


def f32_math(xf, wf):
    """The kernel's math with no bf16 rounding anywhere (to check the hand-written backward)."""
    pos = jnp.arange(xf.shape[0]) % S
    c, s = cos[pos].astype(jnp.float32)[:, None, :], sin[pos].astype(jnp.float32)[:, None, :]
    x1, x2 = xf[:, :, 0, :], xf[:, :, 1, :]
    inv = jax.lax.rsqrt((jnp.sum(x1 * x1, -1, keepdims=True) + jnp.sum(x2 * x2, -1, keepdims=True)) / 64.0 + EPS)
    z1, z2 = wf[:32] * x1 * inv, wf[32:] * x2 * inv
    return jnp.stack([z1 * c - z2 * s, z2 * c + z1 * s], axis=2)


_, vjp_f = jax.vjp(f32_math, x.astype(jnp.float32), w.astype(jnp.float32))
dx_f, dw_f = vjp_f(dy.astype(jnp.float32))
print(f"backward against the f32 math: kernel dx {rel(dx_k, dx_f):.2e} dw {rel(dw_k, dw_f):.2e}; "
      f"reference autodiff dx {rel(dx_r, dx_f):.2e} dw {rel(dw_r, dw_f):.2e}")
