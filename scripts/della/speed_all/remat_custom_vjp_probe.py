"""Does a remat policy keep a residual named inside a custom_vjp's fwd rule (2026-10-27)? Counts the forward GEMM
[64, 128] in the remat'd gradient's jaxpr: once (top) when the named residual is kept, twice when the backward pass runs
the fwd rule again."""
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

POLICY = jax.checkpoint_policies.save_only_these_names("res_y", "res_z")


def make(variant):
    @jax.custom_vjp
    def f(x, w):
        return jnp.tanh(x @ w)

    def f_fwd(x, w):
        y = x @ w
        if variant == "name_residual":
            y = checkpoint_name(y, "res_y")
            return jnp.tanh(y), (x, w, y)
        if variant == "name_output":
            z = checkpoint_name(jnp.tanh(y), "res_z")
            return z, (x, w, y, z)
        if variant == "name_residual_late":
            out = jnp.tanh(y)
            return out, (x, w, checkpoint_name(y, "res_y"))
        if variant == "barrier":
            out = jnp.tanh(y)
            y_n = checkpoint_name(y, "res_y")
            _, out = jax.lax.optimization_barrier((y_n, out))
            return out, (x, w, y_n)
        if variant == "name_both":
            y = checkpoint_name(y, "res_y")
            z = checkpoint_name(jnp.tanh(y), "res_z")
            return z, (x, w, y, z)
        return jnp.tanh(y), (x, w, y)

    def f_bwd(res, g):
        x, w, y = res[:3]
        dy = g * (1 - jnp.tanh(y) ** 2)
        return dy @ w.T, x.T @ dy

    f.defvjp(f_fwd, f_bwd)
    return f


x = jnp.ones((64, 32))
w = jnp.ones((32, 128))
for variant in ("unnamed", "name_residual", "name_residual_late", "barrier", "name_output", "name_both"):
    f = make(variant)
    loss = lambda w_, x_: jnp.sum(f(x_ * 2.0, w_) ** 2)  # noqa: E731
    text = str(jax.make_jaxpr(jax.grad(jax.checkpoint(loss, policy=POLICY)))(w, x))
    print(f"{variant:14s}: forward GEMMs {text.count('f32[64,128] = dot_general')}", flush=True)

# the fused MLP's custom_vjp directly, and through LlamaMlp (named_call, NamedArrays)
import equinox as eqx  # noqa: E402
import haliax as hax  # noqa: E402
from haliax.jax_utils import named_call  # noqa: E402
from levanter.models import llama as L  # noqa: E402
from levanter.utils.activation import ActivationFunctionEnum  # noqa: E402

POL2 = jax.checkpoint_policies.save_only_these_names("mlp_gu")
x2 = jnp.ones((128, 32), jnp.bfloat16)
wc = jnp.ones((32, 256), jnp.bfloat16)
wd = jnp.ones((128, 32), jnp.bfloat16)
for label, fn in (
    ("direct", lambda wc_, x_: jnp.sum(L._fused_swiglu_mlp(x_, wc_, wd, None).astype(jnp.float32) ** 2)),
    ("named_call", lambda wc_, x_: jnp.sum(named_call(lambda a, b: L._fused_swiglu_mlp(a, b, wd, None))(x_, wc_).astype(jnp.float32) ** 2)),
):
    text = str(jax.make_jaxpr(jax.grad(jax.checkpoint(fn, policy=POL2)))(wc, x2))
    print(f"{label:14s}: forward GEMMs {text.count('bf16[128,256] = dot_general')}", flush=True)
