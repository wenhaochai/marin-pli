"""CPU check of LlamaMlp's LEVANTER_SWIGLU_FUSED path (the XLA fallback of the kernel) against the plain and SWIGLU_VJP
paths (2026-10-27): loss and gradients directly and under a checkpoint policy that saves mlp_gu / mlp_gate / mlp_up and
mlp_out, the gradients against an f32 reference, and how many forward GEMM equations ([N, 2F] or [B, S, F] outputs) the
gradient's jaxpr holds (the policy keeps the saved values when the remat'd gradient holds as many as the plain one).
Every mode traces fresh closures: JAX caches traces by function, and the mode is a module global it does not see."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name

import haliax as hax
from levanter.models import llama as L
from levanter.utils.activation import ActivationFunctionEnum

Batch, Pos, Embed, Mlp = hax.Axis("batch", 2), hax.Axis("position", 64), hax.Axis("embed", 32), hax.Axis("mlp", 128)
mlp = L.LlamaMlp.init(Embed, Mlp, ActivationFunctionEnum.silu, key=jax.random.PRNGKey(0))
mlp = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if eqx.is_inexact_array(a) else a, mlp)
x = hax.random.normal(jax.random.PRNGKey(1), (Batch, Pos, Embed)).astype(jnp.bfloat16)
POLICY = jax.checkpoint_policies.save_only_these_names("mlp_gu", "mlp_gate", "mlp_up", "mlp_out")
params, static = eqx.partition(mlp, eqx.is_inexact_array)


def run(mode: str, remat: bool, model_params):
    L._SWIGLU_VJP = mode == "vjp"
    L._SWIGLU_FUSED = mode == "fused"
    dtype = jax.tree.leaves(model_params)[0].dtype

    def loss(p, xa):  # a fresh closure per call, so no trace of another mode is reused
        out = eqx.combine(p, static)(hax.named(xa, x.axes)).array
        # named as LlamaDecoderLayer names it: fmlpo saves it, so the backward pass need not rebuild the MLP output
        return jnp.sum(jnp.square(checkpoint_name(out, "mlp_out").astype(jnp.float32)))

    f = jax.checkpoint(loss, policy=POLICY) if remat else loss
    vg = jax.value_and_grad(f, argnums=(0, 1))
    text = str(jax.make_jaxpr(vg)(model_params, x.array.astype(dtype)))
    v, (gp, gx) = jax.jit(vg)(model_params, x.array.astype(dtype))
    grads = [np.asarray(a, np.float32) for a in (gp.gate_proj.weight.array, gp.up_proj.weight.array,
                                                  gp.down_proj.weight.array, gx)]
    n = Batch.size * Pos.size
    dt = "bf16" if dtype == jnp.bfloat16 else "f32"
    fwd_gemms = (text.count(f"{dt}[{n},{2 * Mlp.size}] = dot_general")
                 + text.count(f"{dt}[{Batch.size},{Pos.size},{Mlp.size}] = dot_general"))
    return float(v), grads, fwd_gemms, text.count("mlp_gu")


res = {(m, r): run(m, r, params) for m in ("plain", "vjp", "fused") for r in (False, True)}
ref_v, ref_g, _, _ = res[("plain", False)]
for k, (v, gs, n_fwd, n_gu) in res.items():
    rel = [float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30)) for a, b in zip(gs, ref_g)]
    print(f"{k}: loss {v:.6f} (plain {ref_v:.6f}); max rel grad diff gate/up/down/x "
          + " ".join(f"{r:.2e}" for r in rel) + f"; forward-GEMM eqns {n_fwd}; mlp_gu names {n_gu}", flush=True)

params32 = jax.tree.map(lambda a: a.astype(jnp.float32), params)
_, g32, _, _ = run("plain", False, params32)
for mode in ("plain", "vjp", "fused"):
    gs = res[(mode, False)][1]
    msg = [f"{nm} {np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)):.2e}"
           for nm, a, b in zip(("gate", "up", "down", "x"), gs, g32)]
    print(f"rms rel error against f32, {mode}: " + " ".join(msg), flush=True)
