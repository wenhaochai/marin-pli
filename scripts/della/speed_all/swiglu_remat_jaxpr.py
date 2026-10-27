"""Where the [N, 2F] forward GEMM of LlamaMlp's fused path sits in the remat'd gradient's jaxpr (CPU): the jaxpr goes
to tmp/speed_wt/profiles/swiglu_remat_fused.jaxpr.txt and the enclosing primitives of each GEMM are printed."""
import os

import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
from levanter.models import llama as L
from levanter.utils.activation import ActivationFunctionEnum

Batch, Pos, Embed, Mlp = hax.Axis("batch", 2), hax.Axis("position", 64), hax.Axis("embed", 32), hax.Axis("mlp", 128)
mlp = L.LlamaMlp.init(Embed, Mlp, ActivationFunctionEnum.silu, key=jax.random.PRNGKey(0))
mlp = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if eqx.is_inexact_array(a) else a, mlp)
x = hax.random.normal(jax.random.PRNGKey(1), (Batch, Pos, Embed)).astype(jnp.bfloat16)
POLICY = jax.checkpoint_policies.save_only_these_names("mlp_gu", "mlp_gate", "mlp_up", "mlp_out")
params, static = eqx.partition(mlp, eqx.is_inexact_array)
os.environ["LEVANTER_SWIGLU_FUSED"], L._SWIGLU_VJP = "1", False


def loss(p, xa):
    return jnp.sum(jnp.square(eqx.combine(p, static)(hax.named(xa, x.axes)).array.astype(jnp.float32)))


closed = jax.make_jaxpr(jax.grad(jax.checkpoint(loss, policy=POLICY)))(params, x.array)
open("/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/swiglu_remat_fused.jaxpr.txt", "w").write(str(closed))


def walk(jaxpr, path):
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "dot_general" and tuple(eqn.outvars[0].aval.shape) == (Batch.size * Pos.size, 2 * Mlp.size):
            print("GEMM [N,2F] in", " > ".join(path) or "top")
        for p in eqn.params.values():
            subs = p if isinstance(p, (list, tuple)) else [p]
            for sub in subs:
                inner = getattr(sub, "jaxpr", None)
                if inner is not None and hasattr(inner, "eqns"):
                    walk(inner, path + [name])
                elif hasattr(sub, "eqns"):
                    walk(sub, path + [name])


walk(closed.jaxpr, [])
