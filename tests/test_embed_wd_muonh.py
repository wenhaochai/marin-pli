"""CPU check of EmbedWDMuonHConfig against MuonH on a tiny Qwen3: before ewd_from_step every update equals the baseline's;
from it on the input-embedding update is the baseline's minus adam_lr * ewd * w, every other update and the optimizer
state stay exactly the baseline's."""
import jax, jax.random as jr, numpy as np, equinox as eqx
import haliax as hax
from haliax import Axis
from levanter.models.qwen import Qwen3Config
from levanter.layers.attention import AttentionMask
from levanter.optim.muonh import MuonHConfig
from experiments.references.embed_wd_muonh import EmbedWDMuonHConfig

kw = dict(max_seq_len=16, hidden_dim=32, intermediate_dim=64, num_layers=2, num_heads=4, num_kv_heads=2, hybrid_norm=True, gradient_checkpointing=False)
V = Axis("vocab", 50); B = Axis("batch", 2); P = Axis("position", 16)
model = Qwen3Config(**kw).build(V, key=jr.PRNGKey(0))
ids = hax.named(jr.randint(jr.PRNGKey(1), (2, 16), 0, 50), (B, P))
params = eqx.filter(model, eqx.is_array)
grads = eqx.filter_grad(lambda m: (m.activations(ids, AttentionMask.causal()).array ** 2).mean())(model)
okw = dict(learning_rate=0.01, adam_lr=0.002, beta1=0.9, beta2=0.98, epsilon=1e-15, weight_decay=0.1, warmup=0, min_lr_ratio=0.0)
base = MuonHConfig(**okw).build(10); ewd = EmbedWDMuonHConfig(**okw, ewd=0.35, ewd_from_step=2).build(10)
sb, se = base.init(params), ewd.init(params)
emb = lambda t: t.embeddings.token_embeddings.weight
for step in range(4):
    lr = float(sb.hyperparams["adam_lr"])
    ub, sb = base.update(grads, sb, params); ue, se = ewd.update(grads, se, params)
    rb = jax.tree_util.tree_leaves(eqx.filter(eqx.tree_at(emb, ub, None), eqx.is_array))
    re_ = jax.tree_util.tree_leaves(eqx.filter(eqx.tree_at(emb, ue, None), eqx.is_array))
    assert all(np.array_equal(a, b) for a, b in zip(rb, re_)), "non-embedding update differs"
    eb, ee, w = emb(ub).array, emb(ue).array, emb(params).array
    if step < 2:
        assert np.array_equal(eb, ee), f"step {step}: differs before the decay starts"
    else:
        assert np.allclose(ee, eb - lr * 0.35 * w, atol=1e-7), f"step {step}: not baseline minus lr*wd*w"
    print("step", step, "ok, lr", lr)
assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(sb), jax.tree_util.tree_leaves(se))), "state differs"
print("EWD OK")
