"""CPU check of Attention._compute_qkv with LEVANTER_QK_ROPE_FUSED=1 (2026-10-27) against the XLA path: q, k and the
gradients of a fixed linear loss of q and k with respect to the input and every attention parameter (the norm weights
included), bf16 module as in training. Off the GPU the fused path runs the jnp reference of the kernel, so this checks
the plumbing (axes, rows, positions, the custom_vjp and the norm-weight gradient)."""
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import haliax as hax
from levanter.layers import RmsNormConfig
from levanter.layers.attention import Attention, AttentionConfig
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig

Batch, Pos, Embed = hax.Axis("batch", 2), hax.Axis("position", 64), hax.Axis("embed", 128)
cfg = AttentionConfig(Embed=Embed, num_heads=8, num_kv_heads=8, head_dim=64, qk_norm=RmsNormConfig(eps=1e-5),
                      rope=DefaultRotaryEmbeddingsConfig(theta=10000.0))
attn = Attention.init(cfg, key=jax.random.PRNGKey(0))
attn = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if eqx.is_inexact_array(a) else a, attn)
# norm weights away from 1 so their gradient path is exercised
attn = eqx.tree_at(lambda m: m.q_norm.weight, attn, attn.q_norm.weight * 1.1)
attn = eqx.tree_at(lambda m: m.k_norm.weight, attn, attn.k_norm.weight * 0.9)
x = hax.random.normal(jax.random.PRNGKey(1), (Batch, Pos, Embed)).astype(jnp.bfloat16)
params, static = eqx.partition(attn, eqx.is_inexact_array)


def run(flag):
    os.environ["LEVANTER_QK_ROPE_FUSED"] = flag

    def f(p, xa):   # fresh closure per mode: JAX caches traces by function
        q, k, _ = eqx.combine(p, static)._compute_qkv(hax.named(xa, x.axes), key=None)
        r_q = jnp.sin(jnp.arange(q.array.size, dtype=jnp.float32)).reshape(q.array.shape)
        r_k = jnp.cos(jnp.arange(k.array.size, dtype=jnp.float32)).reshape(k.array.shape)
        return jnp.sum(q.array.astype(jnp.float32) * r_q) + jnp.sum(k.array.astype(jnp.float32) * r_k), (q.array, k.array)

    (v, (q, k)), (gp, gx) = jax.jit(jax.value_and_grad(f, argnums=(0, 1), has_aux=True))(params, x.array)
    leaves = {jax.tree_util.keystr(kp): np.asarray(a, np.float32) for kp, a in jax.tree_util.tree_leaves_with_path(gp)}
    return float(v), np.asarray(q, np.float32), np.asarray(k, np.float32), leaves, np.asarray(gx, np.float32)


def rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


v0, q0, k0, g0, gx0 = run("0")
v1, q1, k1, g1, gx1 = run("1")
print(f"loss {v1:.4f} vs {v0:.4f}; q max rel {rel(q1, q0):.2e}, k {rel(k1, k0):.2e}; d_x {rel(gx1, gx0):.2e}")
for name in g0:
    print(f"  grad {name}: max rel {rel(g1[name], g0[name]):.2e}")

# the XLA path in f32 (module and input) as the reference for both bf16 paths
params_bf16 = params
params = jax.tree.map(lambda a: a.astype(jnp.float32), params_bf16)
x = x.astype(jnp.float32)
v32, q32, k32, g32, gx32 = run("0")
print("against f32 (XLA path in f32): max rel error of the bf16 XLA path / the bf16 fused path")
print(f"  q {rel(q0, q32):.2e} / {rel(q1, q32):.2e}; k {rel(k0, k32):.2e} / {rel(k1, k32):.2e}; d_x {rel(gx0, gx32):.2e} / {rel(gx1, gx32):.2e}")
for name in g0:
    print(f"  grad {name}: {rel(g0[name], g32[name]):.2e} / {rel(g1[name], g32[name]):.2e}")
