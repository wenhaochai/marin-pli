"""CPU tests for the launcher's LENS_FROM path (owner popups 2026-10-17).
1. FrozenMuonHConfig: updates are exactly zero for every leaf, over several steps, and its schedules stay injected.
2. The lens model (pls_weight 0, no separate heads) reads every layer through the final norm and lm_head: readout_losses
   of layer k equals a dense CE of final_norm(h_k) @ lm_head.
3. A lens model takes a separate-heads model's shared parameters (embeddings, blocks, final norm, lm_head) and nothing
   else: the tree of the lens model is the separate-heads tree with the readout heads removed.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, numpy as np  # noqa: E401,E402
import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from experiments.references.per_layer_qwen3 import FrozenMuonHConfig, PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

B, T, V, L = 2, 16, 64, 4
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
FAIL = []


def check(c, m):
    print(("ok   " if c else "FAIL ") + m)
    if not c:
        FAIL.append(m)


def cfg(**kw):
    return PerLayerQwen3Config(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True, use_qk_norm=False, attn_backend=AttentionBackend.VANILLA, **kw)


lens = PerLayerQwen3LMHeadModel.init(Vocab, cfg(pls_weight=0.0, pls_eval=True, pls_monitor_stride=1), key=jrandom.PRNGKey(0))
sep = PerLayerQwen3LMHeadModel.init(Vocab, cfg(pls_weight=1.0, pls_separate_heads=True), key=jrandom.PRNGKey(0))
p = eqx.filter(lens, eqx.is_array)
oc = FrozenMuonHConfig(learning_rate=0.02, adam_lr=1e-3, warmup=0, min_lr_ratio=1.0, lr_schedule="constant")
opt = oc.build(10); st = opt.init(p)
zero = True
for i in range(3):
    g = jax.tree.map(lambda x: jrandom.normal(jrandom.PRNGKey(i), x.shape, x.dtype), p)
    u, st = opt.update(g, st, p)
    zero = zero and all(float(jnp.max(jnp.abs(x))) == 0.0 for x in jax.tree.leaves(u))
check(zero, "1. FrozenMuonHConfig: every update is exactly zero over 3 steps")
check("learning_rate" in st.hyperparams and "adam_lr" in st.hyperparams, "1. the schedules stay injected (learning_rate, adam_lr)")

r = np.random.default_rng(0)
tok = r.integers(0, V, (B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32), (Batch, Pos)), attn_mask=AttentionMask.causal())
per = lens.readout_losses(ex, tuple(range(L)))
outs = lens.layer_outputs(ex.tokens, ex.attn_mask)
W = lens.get_lm_head().rearrange((lens.Vocab, lens.Embed)).array
y = np.roll(tok, -1, 1)
for k in range(L):
    h = lens.transformer.norm(outs[lens.transformer.layers.Block.name, k]).rearrange((Batch, Pos, lens.Embed)).array
    ref = -np.take_along_axis(np.asarray(jax.nn.log_softmax(jnp.einsum("bte,ve->btv", h, W), -1)), y[..., None], -1)[..., 0]
    got = np.asarray(per["readout", k].rearrange((Batch, Pos)).array)
    check(np.max(np.abs(got[:, : T - 1] - ref[:, : T - 1])) < 1e-4, f"2. layer {k}: the lens model's readout = CE of final_norm(h_k) @ lm_head")
sh = lambda m: [x.shape for x in jax.tree.leaves(eqx.filter((m.embeddings, m.transformer, m.lm_head), eqx.is_array))]  # noqa: E731
check(sh(lens) == sh(sep) and lens.aux_lm_heads is None and sep.aux_lm_heads is not None, "3. the lens model's tree is the separate-heads model's without its readout heads")
print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
