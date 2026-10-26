"""H100 debug of the FA4 THD path (branch fa4-sm90, 2026-10-26): w276 aborted in C++ (std::bad_variant_access) on the
first fa4thd call. One step per process (argv[1]), so an abort names its step:
  fwd64     fa4_thd_attention_forward, head_dim 64, 12 query heads = 12 KV heads (the 300m shape, 2 x 4096 tokens)
  bwd64     fa4_thd_attention_backward on the same (the new one-query-head-per-KV-head SM90 launcher)
  fwd128    levanter's original path: head_dim 128, 16 query heads over 4 KV heads (GQA), forward
  bwd128    the same, backward
  local64   fa4_thd_local_attention (cu_seqlens from segment ids, padded), forward + grad
Prints STEP <name> OK with a checksum, or the exception; an abort shows as the step's missing OK line."""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug.attention import _fa4_thd as t

step = sys.argv[1]
B, S = 2, 4096
heads, kv, d = (12, 12, 64) if step.endswith("64") else (16, 4, 128)
total = B * S
ks = jax.random.split(jax.random.key(0), 4)
q = jax.random.normal(ks[0], (total, heads, d), jnp.float32).astype(jnp.bfloat16)
k = jax.random.normal(ks[1], (total, kv, d), jnp.float32).astype(jnp.bfloat16)
v = jax.random.normal(ks[2], (total, kv, d), jnp.float32).astype(jnp.bfloat16)
do = jax.random.normal(ks[3], (total, heads, d), jnp.float32).astype(jnp.bfloat16)
cu = jnp.asarray([0, 1000, 2500, 4096, 5000, 8192], jnp.int32)
scale = 1.0 / float(np.sqrt(d))
print("STEP", step, "START", jax.devices()[0], flush=True)
cfg = t._thd_kernel_config(d)
print("config", cfg, flush=True)
if step.startswith("fwd"):
    out, lse = jax.jit(lambda q, k, v, c: t.fa4_thd_attention_forward(q, k, v, c, softmax_scale=scale, kernel_config=cfg, sliding_window=None))(q, k, v, cu)
    print("STEP", step, "OK", float(jnp.sum(out.astype(jnp.float32))), float(jnp.sum(lse)), flush=True)
elif step.startswith("bwd"):
    out, lse = jax.jit(lambda q, k, v, c: t.fa4_thd_attention_forward(q, k, v, c, softmax_scale=scale, kernel_config=cfg, sliding_window=None))(q, k, v, cu)
    print("forward done", flush=True)
    dq, dk, dv = jax.jit(lambda q, k, v, o, g, l, c: t.fa4_thd_attention_backward(q, k, v, o, g, l, c, softmax_scale=scale, kernel_config=cfg, sliding_window=None))(q, k, v, out, do, lse, cu)
    print("STEP", step, "OK", *(float(jnp.sum(x.astype(jnp.float32))) for x in (dq, dk, dv)), flush=True)
elif step == "local64":
    seg = jnp.asarray(np.repeat(np.array([[0, 1, 2], [3, 4, 5]]), [1000, 1500, 1596], axis=1), jnp.int32)
    qb, kb, vb = (x.reshape(B, S, x.shape[1], d) for x in (q, k, v))
    f = lambda q, k, v: jnp.sum(t.fa4_thd_local_attention(q, k, v, seg, softmax_scale=scale, max_documents=20).astype(jnp.float32))
    val, g = jax.jit(jax.value_and_grad(f, argnums=(0, 1, 2)))(qb, kb, vb)
    print("STEP", step, "OK", float(val), *(float(jnp.sum(x.astype(jnp.float32))) for x in g), flush=True)
