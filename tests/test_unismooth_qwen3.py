"""CPU checks of UniSmoothQwen3Config against the baseline (tiny model), from the spec in unismooth_qwen3.py:
1. same init parameters as the baseline;
2. loss equals the baseline's at evaluation (no step), before us_from_step, and with eps 0;
3. from us_from_step on, the loss equals an explicit full-softmax -sum_v q_v log p_v with q = (1-eps) onehot + eps u,
   weighted-mean over the loss weight, and the gradients equal that reference's gradients;
4. the gradient with respect to the logits is p - q (one position, explicit);
5. padding ids beyond the count vector get u = 0; under jit with a traced step the switch still works; grads finite."""
import os, tempfile
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx
import haliax as hax
from haliax import Axis
from levanter.models.qwen import Qwen3Config
from levanter.models.lm_model import LmExample
import levanter.trainer as T
from experiments.references.unismooth_qwen3 import UniSmoothQwen3Config, unigram

jax.config.update("jax_default_matmul_precision", "highest")
kw = dict(max_seq_len=16, hidden_dim=32, intermediate_dim=64, num_layers=2, num_heads=4, num_kv_heads=2, hybrid_norm=True, gradient_checkpointing=False)
V = Axis("vocab", 53); B = Axis("batch", 3); P = Axis("position", 16)
counts = np.random.RandomState(0).randint(1, 50, size=50)          # 50 < 53: ids 50-52 are padding (u = 0)
path = os.path.join(tempfile.mkdtemp(), "counts.npy"); np.save(path, counts)
u = unigram(path, 53); assert abs(u.sum() - 1) < 1e-6 and (u[50:] == 0).all(); print("unigram ok, padding 0")
EPS, FROM = 0.1, 5
base_cfg = Qwen3Config(**kw); us_cfg = UniSmoothQwen3Config(**kw, us_eps=EPS, us_from_step=FROM, us_counts=path)
key = jr.PRNGKey(0)
mb = base_cfg.build(V, key=key); mu = us_cfg.build(V, key=key)
lb, lu = jax.tree_util.tree_leaves(eqx.filter(mb, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(mu, eqx.is_array))
assert len(lb) == len(lu) and all(np.array_equal(a, b) for a, b in zip(lb, lu)), "init differs"; print("1 init identical")
ids = hax.named(jr.randint(jr.PRNGKey(1), (3, 16), 0, 50), (B, P))
lw = hax.named(jnp.ones((3, 16)).at[2, 10:].set(0.0), (B, P))     # a padded tail in one row
from levanter.layers.attention import AttentionMask
ex = LmExample(tokens=ids, loss_weight=lw, attn_mask=AttentionMask.causal())

def with_step(step, f):
    tok = T._TRACED_TRAIN_STEP.set(step)
    try:
        return f()
    finally:
        T._TRACED_TRAIN_STEP.reset(tok)

def loss_of(m, step, k=jr.PRNGKey(2)):
    out = with_step(step, lambda: m.compute_next_token_loss(ex, key=k))
    out = out[0] if isinstance(out, tuple) else out
    return out.array if isinstance(out, hax.NamedArray) else out

base = loss_of(mb, None, None)
assert np.allclose(loss_of(mu, None, None), base, atol=1e-6); print("2 eval identical")
assert np.allclose(loss_of(mu, jnp.asarray(3)), loss_of(mb, jnp.asarray(3)), atol=1e-6); print("2 before from_step identical")
mu0 = UniSmoothQwen3Config(**kw, us_eps=0.0, us_from_step=0, us_counts=path).build(V, key=key)
assert np.allclose(loss_of(mu0, jnp.asarray(7)), loss_of(mb, jnp.asarray(7)), atol=1e-6); print("2 eps 0 identical")

def reference(m):
    """explicit full softmax: weighted mean over positions with weight (loss_weight, last position 0) of -sum q log p"""
    h = m.activations(ids, ex.attn_mask, key=jr.PRNGKey(2)).array.astype(jnp.float32)
    W = m.get_lm_head().array.astype(jnp.float32)
    W = W if W.shape[0] == 53 else W.T                                   # [vocab, embed]
    logp = jax.nn.log_softmax(h @ W.T, axis=-1)                          # [B, P, V]
    tgt = jnp.roll(ids.array, -1, axis=1)
    q = (1 - EPS) * jax.nn.one_hot(tgt, 53) + EPS * jnp.asarray(u)
    per = -(q * logp).sum(-1)
    w = lw.array.at[:, -1].set(0.0)
    return (per * w).sum() / w.sum()

got = loss_of(mu, jnp.asarray(7)); want = with_step(jnp.asarray(7), lambda: reference(mu))
print("3 loss", float(got), "reference", float(want)); assert np.allclose(got, want, rtol=1e-4, atol=1e-5)
gg = eqx.filter_grad(lambda m: loss_of(m, jnp.asarray(7)))(mu)
gr = eqx.filter_grad(lambda m: with_step(jnp.asarray(7), lambda: reference(m)))(mu)
d = max(float(jnp.abs(a - b).max()) for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(gg, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(gr, eqx.is_array))))
print("3 grad max abs diff vs reference", d); assert d < 1e-4
assert all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(eqx.filter(gg, eqx.is_array))); print("5 grads finite")
# 4: logit gradient of one position = p - q
z = jr.normal(jr.PRNGKey(9), (53,)); t = 7
f = lambda z: -(((1 - EPS) * jax.nn.one_hot(t, 53) + EPS * jnp.asarray(u)) * jax.nn.log_softmax(z)).sum()
f2 = lambda z: -jax.nn.log_softmax(z)[t] + EPS * (z[t] - jnp.asarray(u) @ z)     # the form the module uses
assert np.allclose(jax.grad(f)(z), jax.nn.softmax(z) - ((1 - EPS) * jax.nn.one_hot(t, 53) + EPS * jnp.asarray(u)), atol=1e-6)
assert np.allclose(f(z), f2(z), atol=1e-5) and np.allclose(jax.grad(f)(z), jax.grad(f2)(z), atol=1e-6); print("4 logit gradient p - q; module form equal")
# 5: jit with a traced step switches at from_step
jl = jax.jit(lambda s: loss_of(mu, s))
assert np.allclose(jl(jnp.asarray(3)), loss_of(mb, jnp.asarray(3)), atol=1e-5) and np.allclose(jl(jnp.asarray(7)), got, atol=1e-5); print("5 jit traced step ok")
print("ALL OK")
