"""CPU tests for the two depth_arch_qwen3 variants of the owner popup of 2026-10-22: mhc with identity residual mixing
(depth_mhc_res="identity") and attnres with the pseudo-query Post Clip (DepthArchMuonHConfig.attnres_query_clip). Run on a vis node from this worktree:
    nice -n 19 $PY -u scripts/della/depth_arch_variant3_test.py
1. mhc identity: with every residual-logit bias offset at -1e4 off the diagonal (Sinkhorn then gives I up to its eps), mhc
   == mhc identity; with random phi, gains and biases the two differ; in identity mode the residual columns of phi, the
   residual gain and the residual biases get exactly zero gradient, read and write do not.
2. Post Clip: a step on random large pseudo-queries leaves every s = ||q * g|| sqrt(d) <= tau (to 1e-4 relative), leaves
   queries with s < tau and every other parameter exactly as the unclipped optimizer moves them, and scales a clipped
   query along its own direction.
3. A separate-heads and a probes training step of each variant (3 optimizer steps for the clip) are finite, routing
   gradients are non-zero, and validation rejects the wrong designs.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import dataclasses  # noqa: E402
import math  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402

from experiments.references.depth_arch_qwen3 import DepthArchMuonHConfig  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
rng = np.random.default_rng(0)
B, T, V, L, D = 4, 16, 300, 4, 48
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True)
key0 = jrandom.PRNGKey(0)
mask = AttentionMask.causal()
x = hax.named(jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32), (Batch, Pos, Axis("embed", D)))
tok = rng.integers(2, V, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=mask)


def model(arch, **kw):
    cfg = PerLayerQwen3Config(**{**common, **kw}, hybrid_norm=True, depth_arch=arch, scan_layers=False)
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=key0)


def outs(m, loop=False):
    tr = m.transformer
    if loop:
        return np.stack([np.asarray(o.rearrange((Batch, Pos, tr.config.Embed)).array) for o in tr._loop_outputs(x, mask)])
    return np.asarray(tr.outputs(x, mask).rearrange((tr.config.Layers, Batch, Pos, tr.config.Embed)).array)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def close(a, b, what, tol=2e-5):
    r = rel(a, b)
    assert r <= tol, f"{what}: relative error {r:.2e}"
    print(f"ok {what} (rel {r:.1e})")


def rand_like(a, scale, k):
    return hax.named(jnp.asarray(np.random.default_rng(k).standard_normal(a.array.shape) * scale, a.array.dtype), a.axes)


def lf(m_):
    out = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
    loss = out[0] if isinstance(out, tuple) else out
    return loss.array if isinstance(loss, hax.NamedArray) else loss


m = 4
# 1. mhc identity ---------------------------------------------------------------------------------------------------
m_mh, m_id = model("mhc"), model("mhc", depth_mhc_res="identity")
hy = list(m_mh.transformer.layers.stacked.hyper)
off = np.zeros(m * m + 2 * m, np.float32)
off[2 * m:] = (-1e4 * (1 - np.eye(m))).reshape(-1)
big = tuple(dataclasses.replace(p, bias=hax.named(jnp.broadcast_to(jnp.asarray(off), p.bias.array.shape), p.bias.axes)) for p in hy)
close(outs(eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_mh, big)), outs(m_id), "1 mhc with A = I == mhc identity", tol=5e-5)
rnd = tuple(dataclasses.replace(p, phi=rand_like(p.phi, 0.05, 10 + j), bias=rand_like(p.bias, 1.0, 20 + j), gain=rand_like(p.gain, 0.5, 30 + j)) for j, p in enumerate(hy))
m_mh2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_mh, rnd)
m_id2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_id, rnd)
d = rel(outs(m_id2), outs(m_mh2))
assert d > 1e-3, d
print(f"ok 1 random routing: identity differs from Sinkhorn mixing (rel {d:.1e})")
# identity's outputs must not depend on the residual columns / gain / biases
rnd_res = tuple(dataclasses.replace(p, phi=hax.named(p.phi.array.at[..., 2 * m:].set(np.random.default_rng(40 + j).standard_normal(p.phi.array[..., 2 * m:].shape)), p.phi.axes),
                                    bias=hax.named(p.bias.array.at[..., 2 * m:].add(3.0), p.bias.axes), gain=hax.named(p.gain.array.at[..., 2].set(7.0), p.gain.axes)) for j, p in enumerate(rnd))
close(outs(eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_id, rnd_res)), outs(m_id2), "1 identity ignores the residual logits", tol=1e-6)
_, g = eqx.filter_value_and_grad(lf)(m_id2)
for j, p in enumerate(g.transformer.layers.stacked.hyper):
    res_g = float(jnp.abs(p.phi.array[..., 2 * m:]).max()) + float(jnp.abs(p.bias.array[..., 2 * m:]).max()) + float(jnp.abs(p.gain.array[..., 2]).max())
    rw_g = float(jnp.abs(p.phi.array[..., : 2 * m]).max()) + float(jnp.abs(p.bias.array[..., : 2 * m]).max())
    assert res_g == 0.0 and rw_g > 0, (j, res_g, rw_g)
print("ok 1 identity: residual-logit gradients are 0, read/write gradients are not")

m_ar = model("attnres")

# 2. Post Clip ------------------------------------------------------------------------------------------------------
TAU = 10.0
kw = dict(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5, muon_epsilon=1e-5,
          max_grad_norm=1.0, weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0, coefficient_type="simple")
o_plain, o_clip = DepthArchMuonHConfig(**kw), DepthArchMuonHConfig(**kw, attnres_query_clip=TAU)
# queries with s from ~2 to ~40: some above tau, some below
sc = np.exp(np.random.default_rng(60).uniform(np.log(2.0), np.log(40.0), size=(L, 2, 1))) / math.sqrt(D)
qv = np.random.default_rng(61).standard_normal((L, 2, D))
qv = qv / np.linalg.norm(qv, axis=-1, keepdims=True) * sc
st = m_ar.transformer.layers.stacked
qa = hax.named(jnp.asarray(qv, jnp.float32), st.query.rearrange(("layer", "sub", "embed")).axes).rearrange(st.query.axes)
fv = np.random.default_rng(62).standard_normal(D)
fq = hax.named(jnp.asarray(fv / np.linalg.norm(fv) * 30.0 / math.sqrt(D), jnp.float32), m_ar.transformer.final_query.axes)  # s = 30
m_c = eqx.tree_at(lambda mm: (mm.transformer.layers.stacked.query, mm.transformer.final_query), m_ar, (qa, fq))


def one_step(mm, opt):
    p = eqx.filter(mm, eqx.is_array)
    tx = opt.build(100)
    st_ = tx.init(p)
    _, gr = eqx.filter_value_and_grad(lf)(mm)
    u, _ = tx.update(eqx.filter(gr, eqx.is_array), st_, p)
    return eqx.apply_updates(mm, u)


def svals(mm):
    t = mm.transformer
    w = np.asarray(t.layers.stacked.query.rearrange(("layer", "sub", "embed")).array) * np.asarray(t.layers.stacked.key_gain.rearrange(("layer", "sub", "embed")).array)
    wf = np.asarray(t.final_query.array) * np.asarray(t.final_key_gain.array)
    return np.concatenate([np.linalg.norm(w, axis=-1).ravel(), [np.linalg.norm(wf)]]) * math.sqrt(D)


m_p1, m_c1 = one_step(m_c, o_plain), one_step(m_c, o_clip)
s_p, s_c = svals(m_p1), svals(m_c1)
print(f"   s before clip: {np.round(np.sort(s_p), 1)}")
assert (s_p > TAU).sum() >= 2 and (s_p < TAU).sum() >= 2, s_p
assert np.all(s_c <= TAU * (1 + 1e-4)), s_c
under = s_p <= TAU
assert np.allclose(s_c[under], s_p[under], rtol=1e-6), (s_c[under], s_p[under])
assert np.allclose(s_c[~under], TAU, rtol=1e-4), s_c[~under]
print(f"ok 2 after the step: every s <= {TAU}; {int((~under).sum())} clipped to tau, {int(under.sum())} untouched")
qp = np.asarray(m_p1.transformer.layers.stacked.query.rearrange(("layer", "sub", "embed")).array)
qc = np.asarray(m_c1.transformer.layers.stacked.query.rearrange(("layer", "sub", "embed")).array)
cosq = (qp * qc).sum(-1) / (np.linalg.norm(qp, axis=-1) * np.linalg.norm(qc, axis=-1))
assert np.all(cosq > 1 - 1e-6), cosq
print("ok 2 a clipped query keeps its direction")
leaves_p = jax.tree_util.tree_leaves_with_path(eqx.filter(m_p1, eqx.is_array))
leaves_c = jax.tree_util.tree_leaves(eqx.filter(m_c1, eqx.is_array))
n_other = 0
for (kp, a), b in zip(leaves_p, leaves_c):
    k = jax.tree_util.keystr(kp)
    if ".stacked.query" in k or ".final_query" in k:
        continue
    assert np.array_equal(np.asarray(a), np.asarray(b)), k
    n_other += 1
print(f"ok 2 the other {n_other} parameters are identical with and without the clip")

# 3. training steps and validation ----------------------------------------------------------------------------------
for name, arch, mkw, opt in (("mhc identity", "mhc", dict(depth_mhc_res="identity"), o_plain), ("attnres qclip", "attnres", {}, o_clip)):
    for bb in (False, True):
        mm = model(arch, pls_detach_backbone=bb, **mkw)
        p = eqx.filter(mm, eqx.is_array)
        tx = opt.build(100)
        st_ = tx.init(p)
        losses = []
        for _ in range(3):
            loss, gr = eqx.filter_value_and_grad(lf)(mm)
            assert np.isfinite(float(loss)), (name, bb)
            for leaf in jax.tree.leaves(gr):
                assert np.all(np.isfinite(np.asarray(leaf))), name
            u, st_ = tx.update(eqx.filter(gr, eqx.is_array), st_, eqx.filter(mm, eqx.is_array))
            mm = eqx.apply_updates(mm, u)
            losses.append(float(loss))
        s_ = gr.transformer.layers.stacked
        gn = sum(float(jnp.linalg.norm(p_.phi.array)) for p_ in s_.hyper) if arch == "mhc" else float(jnp.linalg.norm(s_.query.array))
        assert gn > 0, (name, bb)
        print(f"ok 3 {name} {'probes' if bb else 'separate heads'}: losses {np.round(losses, 3)}, routing grad {gn:.2e}")

for arch, kw_ in (("hc", dict(depth_mhc_res="identity")), ("mhc", dict(depth_mhc_res="eye"))):
    try:
        model(arch, **kw_)
    except ValueError as e:
        print(f"ok 3 rejects {arch} {kw_}: {str(e)[:70]}")
    else:
        raise AssertionError((arch, kw_))
print("ALL OK")
