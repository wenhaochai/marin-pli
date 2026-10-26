"""CPU checks of FloodQwen3Config against the baseline (tiny model), from the spec in flood_qwen3.py:
1. same init parameters as the baseline;
2. loss equals the baseline's at evaluation (no step), before fl_from_step, and with no level file;
3. from fl_from_step on, with the level above the loss: loss = 2b - L and the gradient is minus the baseline's;
4. with the level below the loss: loss and gradient equal the baseline's;
5. the level is read at the traced step (two steps with different levels); under jit the switch still works; metrics."""
import os, tempfile
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx
import haliax as hax
from haliax import Axis
from levanter.models.qwen import Qwen3Config
from levanter.models.lm_model import LmExample
from levanter.layers.attention import AttentionMask
import levanter.trainer as T
from experiments.references.flood_qwen3 import FloodQwen3Config, flood_levels

jax.config.update("jax_default_matmul_precision", "highest")
kw = dict(max_seq_len=16, hidden_dim=32, intermediate_dim=64, num_layers=2, num_heads=4, num_kv_heads=2, hybrid_norm=True, gradient_checkpointing=False)
V = Axis("vocab", 53); B = Axis("batch", 3); P = Axis("position", 16)
FROM = 5
key = jr.PRNGKey(0)
ids = hax.named(jr.randint(jr.PRNGKey(1), (3, 16), 0, 53), (B, P))
lw = hax.named(jnp.ones((3, 16)).at[2, 10:].set(0.0), (B, P))
ex = LmExample(tokens=ids, loss_weight=lw, attn_mask=AttentionMask.causal())
base_cfg = Qwen3Config(**kw)
mb = base_cfg.build(V, key=key)


def with_step(step, f):
    tok = T._TRACED_TRAIN_STEP.set(step)
    try:
        return f()
    finally:
        T._TRACED_TRAIN_STEP.reset(tok)


def out_of(m, step, k=jr.PRNGKey(2)):
    return with_step(step, lambda: m.compute_next_token_loss(ex, key=k))


def loss_of(m, step, k=jr.PRNGKey(2)):
    out = out_of(m, step, k)
    out = out[0] if isinstance(out, tuple) else out
    return out.array if isinstance(out, hax.NamedArray) else out


L0 = float(loss_of(mb, jnp.asarray(7)))          # the untrained tiny model's loss, ~ln 53 = 3.97
print("baseline loss", L0)
d = tempfile.mkdtemp()
hi, lo, mixed = (os.path.join(d, n) for n in ("hi.npy", "lo.npy", "mixed.npy"))
np.save(hi, np.full(20, L0 + 0.5, np.float32))     # level above the loss: ascent
np.save(lo, np.full(20, L0 - 0.5, np.float32))     # level below the loss: plain
lv = np.full(20, L0 - 0.5, np.float32); lv[9] = L0 + 0.25
np.save(mixed, lv)                                 # only step 9 floods
assert flood_levels(hi).shape == (20,)

m_hi = FloodQwen3Config(**kw, fl_level=hi, fl_from_step=FROM).build(V, key=key)
lb, lu = jax.tree_util.tree_leaves(eqx.filter(mb, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(m_hi, eqx.is_array))
assert len(lb) == len(lu) and all(np.array_equal(a, b) for a, b in zip(lb, lu)), "init differs"; print("1 init identical")

base_eval = loss_of(mb, None, None)
assert np.allclose(loss_of(m_hi, None, None), base_eval, atol=1e-6); print("2 eval identical")
assert np.allclose(loss_of(m_hi, jnp.asarray(3)), loss_of(mb, jnp.asarray(3)), atol=1e-6); print("2 before from_step identical")
m_none = FloodQwen3Config(**kw, fl_level="", fl_from_step=0).build(V, key=key)
assert np.allclose(loss_of(m_none, jnp.asarray(7)), L0, atol=1e-6); print("2 no level file identical")

got = float(loss_of(m_hi, jnp.asarray(7)))
assert np.isclose(got, 2 * (L0 + 0.5) - L0, atol=1e-5), (got, L0); print("3 loss = 2b - L:", got)
gb = eqx.filter_grad(lambda m: loss_of(m, jnp.asarray(7)))(mb)
gh = eqx.filter_grad(lambda m: loss_of(m, jnp.asarray(7)))(m_hi)
d_neg = max(float(jnp.abs(a + b).max()) for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(gb, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(gh, eqx.is_array))))
assert d_neg < 1e-6, d_neg; print("3 gradient = minus the baseline's, max |g_b + g_f|", d_neg)

m_lo = FloodQwen3Config(**kw, fl_level=lo, fl_from_step=FROM).build(V, key=key)
assert np.isclose(float(loss_of(m_lo, jnp.asarray(7))), L0, atol=1e-6)
gl = eqx.filter_grad(lambda m: loss_of(m, jnp.asarray(7)))(m_lo)
d_eq = max(float(jnp.abs(a - b).max()) for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(gb, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(gl, eqx.is_array))))
assert d_eq < 1e-7, d_eq; print("4 level below: loss and gradient equal the baseline's")

m_mx = FloodQwen3Config(**kw, fl_level=mixed, fl_from_step=FROM).build(V, key=key)
jl = jax.jit(lambda s: loss_of(m_mx, s))
assert np.isclose(float(jl(jnp.asarray(8))), L0, atol=1e-5)
assert np.isclose(float(jl(jnp.asarray(9))), 2 * (L0 + 0.25) - L0, atol=1e-5)
assert np.isclose(float(jl(jnp.asarray(3))), L0, atol=1e-5)
assert np.isclose(float(jl(jnp.asarray(500))), L0, atol=1e-5)   # past the curve: the last level (below the loss)
print("5 level read at the traced step under jit")
_, met = out_of(m_hi, jnp.asarray(7))
assert set(met) == {"ce_loss", "flood_level", "flood_ascent"}
print("5 metrics", {k: float(v.value() if hasattr(v, "value") else v) for k, v in met.items()} if hasattr(next(iter(met.values())), "value") else sorted(met))
print("ALL OK")
