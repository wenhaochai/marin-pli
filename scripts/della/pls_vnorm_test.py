"""CPU tests for depth_attnres_vnorm (owner popup 2026-10-24): Full AttnRes whose sources enter every mix at unit RMS.
    nice -n 19 $PY -u scripts/della/pls_vnorm_test.py
1. vnorm off is the default (same loss and gradients).
2. _mix_buf with vnorm equals sum_i softmax_i(logits) v_i / RMS(v_i) computed in float64 from the same sources, and the scan
   path (outputs) equals the checkpointed Python loop (_loop_outputs) with vnorm on.
3. With vnorm, scaling one source by 1000 (the embedding: x -> 1000 x) leaves every read unchanged up to the RMS eps
   (without vnorm it does not), i.e. source magnitudes are out of the mix.
4. A separate-heads and a probes step are finite with non-zero query gradients and differ from vnorm off; validation
   rejects vnorm with preln, with static routing, and with attnres_block.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402

from experiments.references.depth_arch_qwen3 import _mix_buf, _put  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
rng = np.random.default_rng(0)
B, T, V, L, D = 4, 16, 300, 4, 48
Batch, Pos, Vocab, Embed = Axis("batch", B), Axis("position", T), Axis("vocab", V), Axis("embed", D)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True)
mask = AttentionMask.causal()
x = hax.named(jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32), (Batch, Pos, Embed))
tok = rng.integers(2, V, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=mask)


def model(arch="attnres", **kw):
    cfg = PerLayerQwen3Config(**{**common, **kw}, hybrid_norm=True, depth_arch=arch, scan_layers=False)
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))


def randomize(m, scale=0.5):
    """random pseudo-queries and key gains, so that the reads are not uniform"""
    tr = m.transformer
    st = tr.layers.stacked
    k = iter(jrandom.split(jrandom.PRNGKey(7), 4))
    rnd = lambda a: hax.named(jax.random.normal(next(k), a.array.shape, a.dtype) * scale, a.axes)  # noqa: E731
    return eqx.tree_at(lambda m: (m.transformer.layers.stacked.query, m.transformer.final_query, m.transformer.layers.stacked.key_gain),
                       m, (rnd(st.query), rnd(tr.final_query), hax.named(st.key_gain.array + 0.3 * jax.random.normal(next(k), st.key_gain.array.shape), st.key_gain.axes)))


def outs(m, xx=x, loop=False):
    tr = m.transformer
    if loop:
        return np.stack([np.asarray(o.rearrange((Batch, Pos, tr.config.Embed)).array) for o in tr._loop_outputs(xx, mask)])
    return np.asarray(tr.outputs(xx, mask).rearrange((tr.config.Layers, Batch, Pos, tr.config.Embed)).array)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def lf(m):
    out = m.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
    loss = out[0] if isinstance(out, tuple) else out
    return loss.array if isinstance(loss, hax.NamedArray) else loss


# 1
m0, m1 = randomize(model()), randomize(model(depth_attnres_vnorm=False))
l0, g0 = eqx.filter_value_and_grad(lf)(m0)
l1, g1 = eqx.filter_value_and_grad(lf)(m1)
assert float(l0) == float(l1) and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree.leaves(g0), jax.tree.leaves(g1)))
print(f"ok 1 vnorm off equals the default (loss {float(l0):.5f})")

# 2
Src = Axis("src", 5)
Vb = hax.zeros((Src, Batch, Pos, Embed), jnp.float32)
Rb = hax.zeros((Src, Batch, Pos), jnp.float32)
vs = [rng.standard_normal((B, T, D)) * 10.0 ** rng.uniform(-2, 4) for _ in range(4)]
for i, v in enumerate(vs):
    Vb, Rb = _put(Vb, Rb, i, hax.named(jnp.asarray(v, jnp.float32), (Batch, Pos, Embed)), Embed)
q = hax.named(jnp.asarray(rng.standard_normal(D), jnp.float32), (Embed,))
got = np.asarray(_mix_buf(q, Vb, Rb, 4, True).rearrange((Batch, Pos, Embed)).array, np.float64)
vv = np.stack(vs).astype(np.float64)
un = vv / np.sqrt((vv ** 2).mean(-1, keepdims=True) + 1e-6)
lg = np.einsum("sbtd,d->sbt", un, np.asarray(q.array, np.float64))
w = np.exp(lg - lg.max(0)); w /= w.sum(0)
want = np.einsum("sbt,sbtd->btd", w, un)
assert rel(got, want) < 1e-5, rel(got, want)
mv = randomize(model(depth_attnres_vnorm=True))
a, b = outs(mv), outs(mv, loop=True)
assert rel(a, b) < 1e-5, rel(a, b)
small = x * 0.02   # an embedding RMS like the trained models' (review 2026-10-24: layer 0's input must be at unit RMS in both paths)
a2, b2 = outs(mv, small), outs(mv, small, loop=True)
assert rel(a2, b2) < 1e-5, rel(a2, b2)
print(f"ok 2 _mix_buf vnorm vs float64 rel {rel(got, want):.1e}; scan vs loop rel {rel(a, b):.1e}, at embedding RMS 0.02 {rel(a2, b2):.1e}")

# 3
big = x * 1000.0
d_on = rel(outs(mv, big), outs(mv))
d_off = rel(outs(m0, big), outs(m0))
assert d_on < 1e-4 and d_off > 1e-2, (d_on, d_off)
print(f"ok 3 embedding x1000 changes the reads by rel {d_on:.1e} with vnorm, {d_off:.2f} without")

# 4
for kw in (dict(), dict(pls_detach_backbone=True)):
    m = randomize(model(depth_attnres_vnorm=True, **kw)); mo = randomize(model(**kw))
    l, g = eqx.filter_value_and_grad(lf)(m); lo, _ = eqx.filter_value_and_grad(lf)(mo)
    leaves = jax.tree.leaves(g)
    assert np.isfinite(float(l)) and all(np.all(np.isfinite(np.asarray(t))) for t in leaves)
    gq = float(np.abs(np.asarray(g.transformer.layers.stacked.query.array)).max())
    assert gq > 0 and float(l) != float(lo), (gq, l, lo)
    print(f"ok 4 {'probes' if kw else 'separate heads'} step finite, loss {float(l):.4f} vs off {float(lo):.4f}, max |grad q| {gq:.2e}")
for arch, kw in (("preln", {}), ("attnres", dict(depth_route="static")), ("attnres_block", {})):
    try:
        cfg = PerLayerQwen3Config(**common, hybrid_norm=True, depth_arch=arch, scan_layers=False, depth_attnres_vnorm=True, **kw)
        PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0)).transformer.outputs(x, mask)
    except ValueError as e:
        print(f"ok 4 rejects vnorm with {arch} {kw}: {str(e)[:70]}")
    else:
        raise AssertionError((arch, kw))
print("ALL OK")
