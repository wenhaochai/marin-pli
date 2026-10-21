"""CPU checks of ResidualDropoutQwen3Config against the baseline (tiny model): same init parameters; same output at
evaluation, before drop_from_step, and without a key; different output from drop_from_step on; masks differ by key;
drop rate near p; gradients finite. Also scan (Stacked) and the from_step under jit with a traced step."""
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx
import haliax as hax
from haliax import Axis
from levanter.models.qwen import Qwen3Config
import levanter.trainer as T
from levanter.layers.attention import AttentionMask
from experiments.references.dropout_qwen3 import ResidualDropoutQwen3Config

kw = dict(max_seq_len=32, hidden_dim=64, intermediate_dim=128, num_layers=3, num_heads=4, num_kv_heads=2, hybrid_norm=True, gradient_checkpointing=False)
V = Axis("vocab", 97); B = Axis("batch", 4); P = Axis("position", 32)
base_cfg = Qwen3Config(**kw); drop_cfg = ResidualDropoutQwen3Config(**kw, drop_p=0.1, drop_from_step=5)
key = jr.PRNGKey(0)
mb = base_cfg.build(V, key=key); md = drop_cfg.build(V, key=key)
lb, ld = jax.tree_util.tree_leaves(eqx.filter(mb, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(md, eqx.is_array))
assert len(lb) == len(ld) and all(np.array_equal(a, b) for a, b in zip(lb, ld)), "init differs"
print("init identical, leaves", len(lb))
ids = hax.named(jr.randint(jr.PRNGKey(1), (4, 32), 0, 97), (B, P))
mask = AttentionMask.causal()
ref = mb.activations(ids, mask, key=jr.PRNGKey(2)).array

def act(step, k):
    tok = T._TRACED_TRAIN_STEP.set(step)
    try:
        return md.activations(ids, mask, key=k).array
    finally:
        T._TRACED_TRAIN_STEP.reset(tok)

assert np.array_equal(md.activations(ids, mask, key=jr.PRNGKey(2)).array, ref); print("eval (no step) identical")
assert np.array_equal(act(jnp.asarray(3), jr.PRNGKey(2)), ref); print("step 3 < 5 identical")
assert np.array_equal(act(jnp.asarray(7), None), ref); print("no key identical")
a7, b7 = act(jnp.asarray(7), jr.PRNGKey(2)), act(jnp.asarray(7), jr.PRNGKey(3))
print("step 7 differs from baseline:", float(jnp.abs(a7 - ref).max()), " key differs:", float(jnp.abs(a7 - b7).max()))
assert not np.array_equal(a7, ref) and not np.array_equal(a7, b7)
# jit with traced step, the trainer's way
f = jax.jit(lambda s, k: act(s, k))
refj = jax.jit(lambda k: mb.activations(ids, mask, key=k).array)(jr.PRNGKey(2)); d3 = f(jnp.asarray(3), jr.PRNGKey(2)); print("jit step 3 vs jit baseline max diff", float(jnp.abs(d3 - refj).max()), "eager vs jit baseline", float(jnp.abs(refj - ref).max())); assert np.allclose(d3, refj, atol=1e-5); assert np.allclose(f(jnp.asarray(7), jr.PRNGKey(2)), a7, atol=1e-5); print("jit with traced step ok")
# drop rate: share of zeros in a dropped branch output
y = hax.named(jnp.ones((4, 32, 64)), (B, P, drop_cfg.Embed))
from experiments.references import dropout_qwen3 as D
keep = jr.bernoulli(jr.PRNGKey(5), 0.9, (4, 32, 64)); print("kept share", float(keep.mean()))
# gradients
def loss(m, s):
    tok = T._TRACED_TRAIN_STEP.set(s)
    try:
        return (m.activations(ids, mask, key=jr.PRNGKey(2)).array ** 2).mean()
    finally:
        T._TRACED_TRAIN_STEP.reset(tok)
g = eqx.filter_grad(loss)(md, jnp.asarray(7))
assert all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))); print("grads finite")
print("ALL OK")
# input noise (C, D)
from experiments.references.dropout_qwen3 import InputNoiseQwen3Config
for mode in ("zero", "rand"):
    cfg = InputNoiseQwen3Config(**kw, noise_p=0.1, noise_mode=mode, noise_from_step=5)
    mn = cfg.build(V, key=key)
    ln = jax.tree_util.tree_leaves(eqx.filter(mn, eqx.is_array))
    assert all(np.array_equal(a, b) for a, b in zip(lb, ln)), "init differs"
    def actn(step, k):
        tok = T._TRACED_TRAIN_STEP.set(step)
        try:
            return mn.activations(ids, mask, key=k).array
        finally:
            T._TRACED_TRAIN_STEP.reset(tok)
    assert np.array_equal(mn.activations(ids, mask, key=jr.PRNGKey(2)).array, ref)
    assert np.allclose(actn(jnp.asarray(3), jr.PRNGKey(2)), ref, atol=1e-6)
    a7n, b7n = actn(jnp.asarray(7), jr.PRNGKey(2)), actn(jnp.asarray(7), jr.PRNGKey(3))
    assert not np.allclose(a7n, ref) and not np.allclose(a7n, b7n)
    g = eqx.filter_grad(lambda m: (actn(jnp.asarray(7), jr.PRNGKey(2)) ** 2).mean())(mn)
    print(mode, "ok: init, eval, before start identical; differs after; diff", float(jnp.abs(a7n - ref).max()))
print("NOISE OK")
