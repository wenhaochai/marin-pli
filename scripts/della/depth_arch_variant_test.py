"""CPU tests for the two depth_arch_qwen3 variants (depth_route="static", depth_gain_init="lns"), the Lean session's
L1 / L2 follow-ups (owner popup 2026-10-21). Run on a vis node from this worktree:
    nice -n 19 $PY -u scripts/della/depth_arch_variant_test.py
1. preln with depth_gain_init="lns" at init == lns at init (same key): every layer's output; its pre-norm scales of
   layer l are 1/sqrt(l).
2. hc static at init == hc at init; with random static offsets and random dynamic weights, hc static == an einsum
   reference without the dynamic terms, and its outputs do not depend on the dynamic weights.
3. attnres static at init == attnres at init (both a uniform mean); with random theta, the two-level scan == the Python
   loop, and == an independent reference softmax(sqrt(d) theta) over the sources so far.
4. Optimizer labels: src_logit -> adam_new; hc's read / res / write offsets stay adam_new.
5. A separate-heads and a probes training step of each variant are finite, the routing parameters get non-zero
   gradients, and the run-id tags and validation reject the wrong designs.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
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

from experiments.references.depth_arch_qwen3 import DepthArchMuonHConfig, _hc_constants, _rms  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

jax.config.update("jax_threefry_partitionable", True)
rng = np.random.default_rng(0)
B, T, V, L, D = 4, 16, 300, 4, 48
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True)
key0 = jrandom.PRNGKey(0)
mask = AttentionMask.causal()


def model(arch, **kw):
    cfg = PerLayerQwen3Config(**{**common, **kw}, hybrid_norm=True, depth_arch=arch, scan_layers=False)
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=key0)


x = hax.named(jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32), (Batch, Pos, Axis("embed", D)))


def outs(m, loop=False):
    tr = m.transformer
    if loop:
        return np.stack([np.asarray(o.rearrange((Batch, Pos, tr.config.Embed)).array) for o in tr._loop_outputs(x, mask)])
    o = tr.outputs(x, mask)
    return np.asarray(o.rearrange((tr.config.Layers, Batch, Pos, tr.config.Embed)).array)


def close(a, b, what, tol=2e-5):
    r = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
    assert r <= tol, f"{what}: relative error {r:.2e}"
    print(f"ok {what} (rel {r:.1e})")


def rand_like(a, scale, k):
    return hax.named(jnp.asarray(np.random.default_rng(k).standard_normal(a.array.shape) * scale, a.array.dtype), a.axes)


# 1. gain init
m_lns, m_gi = model("lns"), model("preln", depth_gain_init="lns")
close(outs(m_gi), outs(m_lns), "1 preln gain-init == lns at init")
g = np.asarray(m_gi.transformer.layers.stacked.ln_1.weight.array)
assert np.allclose(g, (1 / np.sqrt(np.arange(1, L + 1)))[:, None]), g[:, 0]
assert np.allclose(np.asarray(m_gi.transformer.layers.stacked.ln_2.weight.array), g)
print("ok 1 scales 1/sqrt(l)")

# 2. hc static
m_hc, m_hs = model("hc"), model("hc", depth_route="static")
close(outs(m_hs), outs(m_hc), "2 hc static == hc at init")
st = m_hs.transformer.layers.stacked
hy = list(st.hyper)
for j in range(2):
    p = hy[j]
    hy[j] = dataclasses.replace(p, read=rand_like(p.read, 0.3, 10 + j), res=rand_like(p.res, 0.3, 20 + j), write=rand_like(p.write, 0.3, 30 + j),
                                w_read=rand_like(p.w_read, 1.0, 40 + j), w_res=rand_like(p.w_res, 1.0, 50 + j), w_write=rand_like(p.w_write, 1.0, 60 + j),
                                scale=rand_like(p.scale, 1.0, 70 + j))
m_hs2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_hs, tuple(hy))
zero_dyn = tuple(dataclasses.replace(p, w_read=p.w_read * 0, w_res=p.w_res * 0, w_write=p.w_write * 0) for p in hy)
m_hs3 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_hs, zero_dyn)
close(outs(m_hs2), outs(m_hs3), "2 hc static ignores the dynamic weights", tol=1e-6)
m_hd3 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_hc, zero_dyn)
close(outs(m_hs2), outs(m_hd3), "2 hc static == dynamic hc with zero dynamic weights")
m_hd2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_hc, tuple(hy))
diff = np.linalg.norm(outs(m_hd2) - outs(m_hs2)) / np.linalg.norm(outs(m_hs2))
assert diff > 1e-3, diff
print(f"ok 2 dynamic hc with the same weights differs (rel {diff:.1e})")


# einsum reference of static hc, layer by layer
def hc_ref(mm):
    tr = mm.transformer
    m = 4
    H = np.stack([np.asarray(x.array)] * m, 0)  # (stream, B, T, D)
    res = []
    for i in range(L):
        lay = tr.layer(i)
        for j in range(2):
            p = lay.hyper[j]
            c_read, c_res, c_write = (np.asarray(c.array) for c in _hc_constants(m, 2 * i + j))
            read, rs, write = c_read + np.asarray(p.read.array), c_res + np.asarray(p.res.array), c_write + np.asarray(p.write.array)
            xin = np.einsum("s,sbtd->btd", read, H)
            f = lay.branch(j, hax.named(jnp.asarray(xin), x.axes), mask, key=None, pos_ids=None)
            f = np.asarray(f.rearrange(x.axes).array)
            H = np.einsum("so,sbtd->obtd", rs, H) + write[:, None, None, None] * f[None]
        res.append(H.sum(0))
    return np.stack(res)


close(outs(m_hs2), hc_ref(m_hs2), "2 hc static == einsum reference", tol=1e-4)

# 3. attnres static
m_ar, m_as = model("attnres"), model("attnres", depth_route="static")
close(outs(m_as), outs(m_ar), "3 attnres static == attnres at init")
sl = m_as.transformer.layers.stacked.src_logit
m_as2 = eqx.tree_at(lambda mm: (mm.transformer.layers.stacked.src_logit, mm.transformer.final_src_logit), m_as,
                    (rand_like(sl, 0.3, 80), rand_like(m_as.transformer.final_src_logit, 0.3, 81)))
close(outs(m_as2), outs(m_as2, loop=True), "3 attnres static: scan == loop", tol=1e-4)


def ar_ref(mm):
    tr = mm.transformer
    th = np.asarray(tr.layers.stacked.src_logit.rearrange(("layer", "sub", "src")).array)
    thf = np.asarray(tr.final_src_logit.array)
    vals = [np.asarray(x.array)]

    def mixw(t, n):
        w = np.exp(math.sqrt(D) * t[:n] - (math.sqrt(D) * t[:n]).max())
        w = w / w.sum()
        return sum(w[k] * vals[k] for k in range(n))

    res = []
    for i in range(L):
        lay = tr.layer(i)
        for j in range(2):
            xin = mixw(th[i, j], len(vals))
            f = lay.branch(j, hax.named(jnp.asarray(xin, jnp.float32), x.axes), mask, key=None, pos_ids=None)
            vals.append(np.asarray(f.rearrange(x.axes).array))
        res.append(mixw(th[i + 1, 0] if i + 1 < L else thf, len(vals)))
    return np.stack(res)


close(outs(m_as2), ar_ref(m_as2), "3 attnres static == reference", tol=1e-4)

# 4. optimizer labels
opt = DepthArchMuonHConfig(learning_rate=0.02, adam_lr=0.008)
for mm, frag in ((m_as2, "src_logit"), (m_hs2, "hyper")):
    labels = opt.create_mask(mm)
    lab = [(jax.tree_util.keystr(p), v) for p, v in jax.tree_util.tree_leaves_with_path(labels, is_leaf=lambda z: isinstance(z, str))]
    hits = [v for p, v in lab if frag in p]
    assert hits and all(v == "adam_new" for v in hits), (frag, hits[:5])
    print(f"ok 4 {frag} -> adam_new ({len(hits)} leaves)")

# 5. training steps
tok = rng.integers(2, V, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=mask)
for name, kw, arch in (("gain-init", dict(depth_gain_init="lns"), "preln"), ("hc static", dict(depth_route="static"), "hc"), ("attnres static", dict(depth_route="static"), "attnres")):
    for bb in (False, True):
        mm = model(arch, pls_detach_backbone=bb, **kw)

        def lf(m_):
            out = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
            loss = out[0] if isinstance(out, tuple) else out
            return loss.array if isinstance(loss, hax.NamedArray) else loss

        loss, gr = eqx.filter_value_and_grad(lf)(mm)
        assert np.isfinite(float(loss)), (name, bb, loss)
        for leaf in jax.tree.leaves(gr):
            assert np.all(np.isfinite(np.asarray(leaf))), name
        st = gr.transformer.layers.stacked
        if arch == "attnres":
            gn = float(jnp.linalg.norm(st.src_logit.array)) + float(jnp.linalg.norm(gr.transformer.final_src_logit.array))
        elif arch == "hc":
            gn = sum(float(jnp.linalg.norm(getattr(p, a).array)) for p in st.hyper for a in ("read", "res", "write"))
        else:
            gn = float(jnp.linalg.norm(st.ln_1.weight.array))
        assert gn > 0, (name, bb)
        print(f"ok 5 {name} {'probes' if bb else 'separate heads'}: loss {float(loss):.3f}, routing/scale grad {gn:.2e}")

for arch, kw in (("preln", dict(depth_route="static")), ("attnres_block", dict(depth_route="static")), ("lns", dict(depth_gain_init="lns"))):
    try:
        model(arch, **kw)
    except ValueError as e:
        print(f"ok 5 rejects {arch} {kw}: {str(e)[:60]}")
    else:
        raise AssertionError((arch, kw))
print("ALL OK")
