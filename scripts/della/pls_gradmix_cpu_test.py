"""CPU tests for the intermediate weight shape and the gradient surgery (per_layer_qwen3: pls_weight_shape,
pls_pcgrad; launcher PLS_SHAPE, PLS_PCGRAD).

Run from the worktree on a vis node, the worktree first on PYTHONPATH:
    nice -n 19 .venv/bin/python -u scripts/della/pls_gradmix_cpu_test.py
1. Shape: the loss equals final + sum_k w_k ce_k with w_k = pls_weight (uniform) or pls_weight * 2(k+1)/L (depth),
   ce_k read from the per-layer metrics; the depth shares sum to L-1, and at L=6, pls_weight=0.2 they are k/15.
2. Junction backward on random cotangents: output = trunk + readout', where readout' = readout when their per-token dot
   product is >= 0, else readout minus its projection on trunk; readout' . trunk >= 0 for every token; tokens are
   independent (the batch and position axes are not mixed).
3. pls_pcgrad leaves the forward loss, the metrics and every head gradient (aux_lm_heads, aux_norms, lm_head, final
   norm) bit-identical, and changes the backbone gradient.
4. Against an independent reference: the backward pass written by hand, layer by layer with jax.vjp (readout
   cotangents from the gradient of the readout losses with respect to the stacked layer outputs, surgery at each
   layer output, then the layer's vjp), gives the model's backbone and embedding gradients, for both shapes.
5. Without surgery the same hand-written backward (plain sums at every layer output) gives the model's gradients:
   the default path computes exactly what it did before, for both shapes.
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

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel, _junction  # noqa: E402

jax.config.update("jax_enable_x64", False)
B, T, V, L = 4, 32, 300, 6
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)


def batch(seed):
    r = np.random.default_rng(seed)
    tok = r.integers(2, V, size=(B, T))
    return LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)),
                     loss_weight=hax.named(jnp.asarray((np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)), (Batch, Pos)),
                     attn_mask=AttentionMask.causal())


common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA,
              pls_separate_heads=True, hybrid_norm=True)


def model(**kw):
    return PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, **kw), key=jrandom.PRNGKey(0))


def with_cfg(m, **kw):
    """The same weights under another config (the config lives on the model and its transformer)."""
    import dataclasses
    c = dataclasses.replace(m.config, **kw)
    return dataclasses.replace(m, transformer=dataclasses.replace(m.transformer, config=c))


def trained(m, steps=12):
    """A few Adam steps on random batches so heads and backbone are not at init (gradients then conflict)."""
    import optax
    opt = optax.adam(3e-3)
    st = opt.init(eqx.filter(m, eqx.is_array))

    @eqx.filter_jit
    def step(m, st, ex):
        g = eqx.filter_grad(lambda mm: mm.compute_next_token_loss(ex, key=jrandom.PRNGKey(1))[0].array)(m)
        u, st = opt.update(g, st, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, u), st

    for i in range(steps):
        m, st = step(m, st, batch(100 + i))
    return m


def loss_and_grad(m, ex):
    def f(mm):
        loss, stats = mm.compute_next_token_loss(ex, key=jrandom.PRNGKey(7))
        return loss.array, stats
    (loss, stats), g = eqx.filter_value_and_grad(f, has_aux=True)(m)
    return float(loss), {k: float(v.value()) for k, v in stats.items()}, g


def leaves(t):
    return [np.asarray(x, np.float64) for x in jax.tree.leaves(eqx.filter(t, eqx.is_array))]


def maxdiff(a, b):
    return max(float(np.abs(x - y).max()) for x, y in zip(leaves(a), leaves(b), strict=True))


def rel(a, b):
    num = sum(float(((x - y) ** 2).sum()) for x, y in zip(leaves(a), leaves(b), strict=True))
    den = sum(float((y ** 2).sum()) for y in leaves(b))
    return (num / den) ** 0.5


fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not ok:
        fails.append(name)


# 1. shapes
ex = batch(0)
base = trained(model(pls_weight=0.2))
for shape in ("uniform", "depth"):
    m = with_cfg(base, pls_weight_shape=shape)
    loss, stats, _ = loss_and_grad(m, ex)
    w = [0.2 * (1.0 if shape == "uniform" else 2.0 * (k + 1) / L) for k in range(L - 1)]
    want = stats[f"pls/L{L - 1}"] + sum(w[k] * stats[f"pls/L{k}"] for k in range(L - 1))
    check(f"1 loss = final + sum w_k ce_k ({shape})", abs(loss - want) < 1e-4 * abs(want), f"{loss:.6f} vs {want:.6f}")
dep = [2.0 * (k + 1) / L for k in range(L - 1)]
check("1 depth shares sum to L-1 and are k/15 at pls_weight 0.2", abs(sum(dep) - (L - 1)) < 1e-12 and np.allclose([0.2 * d for d in dep], [k / 15 for k in range(1, L)]))

# 2. junction backward on random cotangents
r = np.random.default_rng(1)
x = jnp.asarray(r.normal(size=(3, 5, 8)), jnp.float32)
t = jnp.asarray(r.normal(size=(3, 5, 8)), jnp.float32)
c = jnp.asarray(r.normal(size=(3, 5, 8)), jnp.float32)
_, vjp = jax.vjp(lambda a: _junction(a, 2), x)
(got,) = vjp((t, c))
tt, cc = np.asarray(t, np.float64), np.asarray(c, np.float64)
dot = (tt * cc).sum(-1, keepdims=True)
cp = np.where(dot < 0, cc - dot / (tt * tt).sum(-1, keepdims=True) * tt, cc)
check("2 junction: trunk + projected readout", np.abs(np.asarray(got) - (tt + cp)).max() < 1e-5, f"max {np.abs(np.asarray(got) - (tt + cp)).max():.2e}")
check("2 junction: projected readout . trunk >= 0 per token", ((cp * tt).sum(-1) >= -1e-9).all() and 0 < (dot < 0).sum() < dot.size, f"{int((dot < 0).sum())} of {dot.size} tokens projected")
(got2,) = jax.vjp(lambda a: _junction(a, 0), x)[1]((t, c))
d0 = (tt * cc).sum(0, keepdims=True)
cp0 = np.where(d0 < 0, cc - d0 / (tt * tt).sum(0, keepdims=True) * tt, cc)
check("2 junction: the reduction runs over the given axis only", np.abs(np.asarray(got2) - (tt + cp0)).max() < 1e-5)

# 3. pcgrad: forward and heads unchanged, backbone changed
for shape in ("uniform", "depth"):
    off = with_cfg(base, pls_weight_shape=shape)
    on = with_cfg(base, pls_weight_shape=shape, pls_pcgrad=True)
    lo, so, go = loss_and_grad(off, ex)
    ln, sn, gn = loss_and_grad(on, ex)
    check(f"3 forward loss and metrics unchanged ({shape})", lo == ln and so == sn)
    heads = lambda g: (g.aux_lm_heads, g.aux_norms, g.lm_head, g.transformer.norm)  # noqa: E731
    check(f"3 head gradients bit-identical ({shape})", maxdiff(heads(go), heads(gn)) == 0.0)
    check(f"3 backbone gradient changes ({shape})", rel(gn.transformer.layers, go.transformer.layers) > 1e-4, f"rel diff {rel(gn.transformer.layers, go.transformer.layers):.3e}")

# 4. independent reference: the backward pass by hand (5: the same without surgery is the default path)
for shape, surgery in (("uniform", True), ("depth", True), ("uniform", False), ("depth", False)):
    m = with_cfg(base, pls_weight_shape=shape, pls_pcgrad=surgery)
    _, _, g = loss_and_grad(m, ex)
    tr = m.transformer
    Blk = tr.layers.Block
    x0, emb_vjp = jax.vjp(lambda e: e.embed(ex.tokens), m.embeddings)
    hs, vjps, h = [], [], x0
    for i in range(L):
        h, f = jax.vjp(lambda lay, c: lay(c, mask=ex.attn_mask, key=None, pos_ids=None), tr.layers.get_layer(i), h)
        hs.append(h); vjps.append(f)
    outs = hax.stack(Blk, hs)

    def readouts(o):
        final = m._readout_ce(o, L - 1, ex)
        tot = final
        for k in range(L - 1):
            tot = tot + m._readout_ce(o, k, ex) * (0.2 * m._shape(k))
        return tot.array

    g_out = jax.grad(readouts)(outs)
    E = outs.axes.index(m.config.Embed) - 1  # axis of Embed within one layer's output
    ct_trunk = jnp.zeros_like(hs[-1].array)
    lay_grads = [None] * L
    for i in reversed(range(L)):
        rd = g_out[Blk.name, i].array
        tt_, rr = ct_trunk.astype(jnp.float32), rd.astype(jnp.float32)
        dt = jnp.sum(tt_ * rr, axis=E, keepdims=True)
        rr = rr - jnp.where((dt < 0) & surgery, dt / (jnp.sum(tt_ * tt_, axis=E, keepdims=True) + 1e-30), 0.0) * tt_
        ct = hax.named(tt_ + rr, hs[i].axes)
        gl, gc = vjps[i](ct)
        lay_grads[i] = gl
        ct_trunk = gc.array
    (g_emb,) = emb_vjp(hax.named(ct_trunk, x0.axes))
    ref = jax.tree.map(lambda *a: jnp.stack(a), *[eqx.filter(lg, eqx.is_array) for lg in lay_grads])
    got_l = eqx.filter(g.transformer.layers.stacked, eqx.is_array)
    rl = rel(got_l, ref)
    n = "4" if surgery else "5"
    check(f"{n} backbone gradient = hand-written backward, surgery {surgery} ({shape})", rl < 1e-5, f"rel diff {rl:.2e}")
    re_ = rel(g.embeddings, g_emb)
    check(f"{n} embedding gradient = hand-written backward, surgery {surgery} ({shape})", re_ < 1e-5, f"rel diff {re_:.2e}")


print("ALL PASS" if not fails else f"FAILED: {fails}")
