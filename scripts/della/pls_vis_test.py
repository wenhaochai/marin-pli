"""CPU tests for pls_visible_frac (owner popup 2026-10-23): with separate heads, every intermediate readout sees only the
first round(f * hidden) coordinates of its layer output; the final readout sees all. Run on a vis node:
    nice -n 19 $PY -u scripts/della/pls_vis_test.py
1. f = 1 gives the same loss and gradients as the default config.
2. f = 0.5: readout k's CE does not change when the hidden coordinates of layer k's output change, and its gradient with
   respect to layer k's output is exactly 0 on them, nonzero on the visible ones; the final readout's gradient reaches
   every coordinate.
3. A separate-heads training step with f = 0.5 (Pre-LN, 4 layers) is finite and differs from f = 1; validation rejects
   f outside (0, 1] and f < 1 without separate heads.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
B, T, V, L, D = 4, 16, 300, 4, 48
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
rng = np.random.default_rng(0)
tok = rng.integers(2, V, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=AttentionMask.causal())


def model(**kw):
    cfg = PerLayerQwen3Config(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA,
                              pls_weight=1.0, pls_separate_heads=True, hybrid_norm=True, depth_arch="preln", scan_layers=False, **kw)
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))


def lf(m):
    out = m.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
    loss = out[0] if isinstance(out, tuple) else out
    return loss.array if isinstance(loss, hax.NamedArray) else loss


# 1. f = 1 is the default
m0, m1 = model(), model(pls_visible_frac=1.0)
l0, g0 = eqx.filter_value_and_grad(lf)(m0)
l1, g1 = eqx.filter_value_and_grad(lf)(m1)
assert float(l0) == float(l1), (l0, l1)
for a, b in zip(jax.tree.leaves(g0), jax.tree.leaves(g1)):
    assert np.array_equal(np.asarray(a), np.asarray(b))
print(f"ok 1 f = 1 equals the default (loss {float(l0):.4f})")

# 2. readout k ignores the hidden coordinates
mh = model(pls_visible_frac=0.5)
n = D // 2
emb = mh.embeddings.embed(ex.tokens)
outs = mh.transformer.outputs(emb, ex.attn_mask) if hasattr(mh.transformer, "outputs") else None
assert outs is not None
Blk = mh.transformer.layers.Block.name
noise = hax.named(jnp.asarray(rng.standard_normal(outs.array.shape), outs.dtype), outs.axes) * hax.named((jnp.arange(D) >= n).astype(outs.dtype), mh.Embed)
for k in range(L):
    ce = lambda o, k=k: jnp.asarray(mh._readout_ce(o, k, ex).array if isinstance(mh._readout_ce(o, k, ex), hax.NamedArray) else mh._readout_ce(o, k, ex), jnp.float32)  # noqa: E731
    c0, c1 = float(ce(outs)), float(ce(outs + noise * 3.0))
    gr = jax.grad(ce)(outs)
    gk = np.asarray(gr[Blk, k].rearrange((Batch, Pos, mh.Embed)).array)
    if k < L - 1:
        assert abs(c0 - c1) < 1e-6 * max(1.0, abs(c0)), (k, c0, c1)
        assert np.abs(gk[..., n:]).max() == 0.0 and np.abs(gk[..., :n]).max() > 0, k
        print(f"ok 2 readout {k}: CE unchanged by the hidden coordinates ({c0:.4f}), gradient 0 on them")
    else:
        assert abs(c0 - c1) > 1e-3 and np.abs(gk[..., n:]).max() > 0, (k, c0, c1)
        print(f"ok 2 final readout {k}: sees every coordinate (CE {c0:.4f} -> {c1:.4f})")

# 3. a training step, validation
lh, gh = eqx.filter_value_and_grad(lf)(mh)
assert np.isfinite(float(lh)) and all(np.all(np.isfinite(np.asarray(a))) for a in jax.tree.leaves(gh))
d = sum(float(jnp.abs(a - b).max()) for a, b in zip(jax.tree.leaves(gh), jax.tree.leaves(g0)))
assert float(lh) != float(l0) and d > 0
print(f"ok 3 f = 0.5 step finite, loss {float(lh):.4f} vs {float(l0):.4f}")
for kw in (dict(pls_visible_frac=0.0), dict(pls_visible_frac=1.5)):
    try:
        model(**kw)
    except ValueError as e:
        print(f"ok 3 rejects {kw}: {str(e)[:60]}")
    else:
        raise AssertionError(kw)
try:
    PerLayerQwen3Config(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, pls_weight=1.0, pls_separate_heads=False, pls_visible_frac=0.5)
except ValueError as e:
    print(f"ok 3 rejects f < 1 without separate heads: {str(e)[:60]}")
else:
    raise AssertionError("no separate heads")
print("ALL OK")
