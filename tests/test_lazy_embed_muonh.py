"""CPU check of LazyEmbedMuonHConfig against MuonH on a tiny Qwen3: before lazy_from_step every update equals the
baseline's; from it on, input-embedding rows of tokens absent from the batch get a zero update, present rows and every
other parameter get exactly the baseline's update, and the optimizer states stay equal."""
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx
import haliax as hax
from haliax import Axis
from levanter.models.qwen import Qwen3Config
from levanter.layers.attention import AttentionMask
from levanter.optim.muonh import MuonHConfig
from experiments.references.lazy_embed_muonh import LazyEmbedMuonHConfig

kw = dict(max_seq_len=16, hidden_dim=32, intermediate_dim=64, num_layers=2, num_heads=4, num_kv_heads=2, hybrid_norm=True, gradient_checkpointing=False)
V = Axis("vocab", 50); B = Axis("batch", 2); P = Axis("position", 16)
model = Qwen3Config(**kw).build(V, key=jr.PRNGKey(0))
ids = hax.named(jr.randint(jr.PRNGKey(1), (2, 16), 0, 30), (B, P))   # ids 30..49 never appear
params = eqx.filter(model, eqx.is_array)

def loss(m):
    return (m.activations(ids, AttentionMask.causal()).array ** 2).mean() + (m.embeddings.token_embeddings.weight.array ** 2).sum() * 0
grads = eqx.filter_grad(loss)(model)
okw = dict(learning_rate=0.01, adam_lr=0.01, beta1=0.9, beta2=0.98, epsilon=1e-15, weight_decay=0.1, warmup=0, min_lr_ratio=0.0)
base = MuonHConfig(**okw).build(10); lazy = LazyEmbedMuonHConfig(**okw, lazy_from_step=2).build(10)
sb, sl = base.init(params), lazy.init(params)
# a first step with every token present moves all moments, so absent rows have nonzero Adam updates later
all_ids_grad = jax.tree_util.tree_map(lambda g: g + 0.01 if eqx.is_array(g) else g, grads)
for step in range(4):
    g = all_ids_grad if step == 0 else grads
    ub, sb = base.update(g, sb, params); ul, sl = lazy.update(g, sl, params)
    eb, el = ub.embeddings.token_embeddings.weight.array, ul.embeddings.token_embeddings.weight.array
    rest_b = [x for x in jax.tree_util.tree_leaves(eqx.filter(eqx.tree_at(lambda t: t.embeddings.token_embeddings.weight, ub, None), eqx.is_array))]
    rest_l = [x for x in jax.tree_util.tree_leaves(eqx.filter(eqx.tree_at(lambda t: t.embeddings.token_embeddings.weight, ul, None), eqx.is_array))]
    assert all(np.array_equal(a, b) for a, b in zip(rest_b, rest_l)), "non-embedding update differs"
    if step < 2:
        assert np.array_equal(eb, el), f"step {step}: embedding differs before lazy start"
    else:
        pres = np.zeros(V.size, bool); pres[np.unique(np.asarray(ids.array))] = True
        assert np.array_equal(el[pres], eb[pres]), "present rows differ"
        assert np.all(el[~pres] == 0) and np.abs(eb[~pres]).max() > 0, "absent rows not lazy (or baseline did not move them)"
    print("step", step, "ok")
sbl, sll = jax.tree_util.tree_leaves(sb), jax.tree_util.tree_leaves(sl)
assert all(np.array_equal(a, b) for a, b in zip(sbl, sll)), "optimizer state differs"
print("LAZY OK")
