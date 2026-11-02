"""CPU tests for depth_hc_rwdyn="joint" (owner popup 2026-11-02): hc whose read and write take their dynamic terms from
mhc's joint projection of the RMS-normalised concatenated streams; the mixing stays hc's. Run on a vis node from this
worktree:
    CUDA_VISIBLE_DEVICES= nice -n 19 $PY -u scripts/della/depth_arch_hcrwj_test.py
1. The default hc model has no rw_phi / rw_gain (no new parameters, so old checkpoints load), and at init the joint
   variant computes hc's layer outputs (rw_phi starts at 0).
2. With the mixing fixed to I and rw_phi, rw_gain and the read / write offsets set from mhc's read / write columns, gains
   and bias offsets, the joint variant computes mhc with depth_mhc_res="identity", depth_mhc_rw="linear": the two
   designs differ only in how read and write get their dynamic terms. Per-stream hc with the same offsets differs.
3. A separate-heads and a probes training step are finite; w_read and w_write get exactly zero gradient, rw_phi gets a
   non-zero gradient, and the learned mixing (res, w_res) still trains.
4. The joint variant with mhc, static routing, sigmoid read/write or a shared write, and an unknown value, are rejected.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import dataclasses  # noqa: E402

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
rng = np.random.default_rng(0)
B, T, V, L, D, M = 4, 16, 300, 4, 48, 4
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


def outs(m):
    tr = m.transformer
    return np.asarray(tr.outputs(x, mask).rearrange((tr.config.Layers, Batch, Pos, tr.config.Embed)).array)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def rand_like(a, scale, k):
    return hax.named(jnp.asarray(np.random.default_rng(k).standard_normal(a.array.shape) * scale, a.array.dtype), a.axes)


def lf(m_):
    out = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
    loss = out[0] if isinstance(out, tuple) else out
    return loss.array if isinstance(loss, hax.NamedArray) else loss


def with_hyper(m_, hy):
    return eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_, tuple(hy))


def gsum(a):
    return float(jnp.sum(jnp.abs(a.array)))


# 1. default hc unchanged; joint equals hc at init -----------------------------------------------------------------------
m_hc, m_j = model("hc"), model("hc", depth_hc_rwdyn="joint")
assert all(p.rw_phi is None and p.rw_gain is None for p in m_hc.transformer.layers.stacked.hyper), "default hc grew parameters"
n_hc, n_j = len(jax.tree_util.tree_leaves(m_hc)), len(jax.tree_util.tree_leaves(m_j))
assert n_j == n_hc + 4, (n_hc, n_j)  # rw_phi and rw_gain in each of the two sublayers
p0 = m_j.transformer.layers.stacked.hyper[0]
assert p0.rw_phi.axes[-3:] == (Axis("stream", M), Axis("embed", D), Axis("hcrw", 2 * M)) and float(jnp.abs(p0.rw_phi.array).max()) == 0.0, p0.rw_phi.axes
r1 = rel(outs(m_j), outs(m_hc))
assert r1 <= 1e-6, f"init: relative error {r1:.2e}"
print(f"ok 1 default hc has no rw_phi / rw_gain ({n_hc} leaves, joint {n_j}); at init joint equals hc (rel {r1:.1e})")

# 2. with mixing I: joint hc == mhc (identity mixing, linear read/write) under matched parameters --------------------------
m_jid = model("hc", depth_hc_res="identity", depth_hc_rwdyn="joint")
m_mid = model("mhc", depth_mhc_res="identity", depth_mhc_rw="linear")
for name in ("self_attn.q_proj.weight", "mlp.down_proj.weight"):
    get =lambda mm: mm.transformer.layers.stacked.self_attn.q_proj.weight if name.startswith("self_attn") else mm.transformer.layers.stacked.mlp.down_proj.weight  # noqa: E731
    assert np.array_equal(np.asarray(get(m_jid).array), np.asarray(get(m_mid).array)), f"{name} differs between the hc and mhc inits"
mh = list(m_mid.transformer.layers.stacked.hyper)
mh_rnd = [dataclasses.replace(p, phi=rand_like(p.phi, 0.3, 100 + j), bias=rand_like(p.bias, 0.4, 110 + j), gain=rand_like(p.gain, 1.0, 120 + j)) for j, p in enumerate(mh)]
hj = list(m_jid.transformer.layers.stacked.hyper)
hj_match = []
for j, (p, q) in enumerate(zip(hj, mh_rnd)):
    phi_rw = q.phi["mhc", 0 : 2 * M].rename({"mhc": "hcrw"})
    hj_match.append(dataclasses.replace(
        p, rw_phi=phi_rw.rearrange(p.rw_phi.axes), rw_gain=q.gain["hp_scale", 0:2].rearrange(p.rw_gain.axes),
        read=q.bias["mhc", 0:M].rename({"mhc": "stream"}).rearrange(p.read.axes),
        write=q.bias["mhc", M : 2 * M].rename({"mhc": "stream_out"}).rearrange(p.write.axes),
        # hc's per-stream dynamic weights and mixing get random values too: under joint + identity they must not matter
        w_read=rand_like(p.w_read, 0.5, 130 + j), w_write=rand_like(p.w_write, 0.5, 140 + j), res=rand_like(p.res, 0.5, 150 + j),
        w_res=rand_like(p.w_res, 0.5, 160 + j), scale=rand_like(p.scale, 1.0, 170 + j)))
o_mhc = outs(with_hyper(m_mid, mh_rnd))
o_j = outs(with_hyper(m_jid, hj_match))
r2 = rel(o_j, o_mhc)
# float32 rounding only: mhc's linear read adds its +-8 bias constant and subtracts it again ((k > 0) + pre - k), which
# costs ~1e-6 absolute per read weight, and 8 sublayers with random routing amplify it (1.7e-5 measured, 2026-11-02);
# a structural difference is >= 1e-2 (the per-stream check below)
assert r2 <= 1e-4, f"joint hc with I mixing vs mhc identity + linear: rel {r2:.2e}"
m_sid = model("hc", depth_hc_res="identity")  # per-stream hc, the same offsets and random per-stream weights
hs = [dataclasses.replace(p, read=q.read, write=q.write, w_read=q.w_read, w_write=q.w_write, scale=q.scale) for p, q in zip(m_sid.transformer.layers.stacked.hyper, hj_match)]
d2 = rel(outs(with_hyper(m_sid, hs)), o_mhc)
assert d2 > 1e-2, f"per-stream hc should differ from mhc: rel {d2:.2e}"
print(f"ok 2 mixing I: joint hc equals mhc identity + linear under matched parameters (rel {r2:.1e}); per-stream hc differs ({d2:.2e})")

# 3. training steps ---------------------------------------------------------------------------------------------------
for detach in (False, True):
    mm = model("hc", depth_hc_rwdyn="joint", pls_detach_backbone=detach)
    hy = [dataclasses.replace(p, rw_phi=rand_like(p.rw_phi, 0.3, 200 + j), rw_gain=rand_like(p.rw_gain, 1.0, 210 + j), res=rand_like(p.res, 0.3, 220 + j),
                              w_res=rand_like(p.w_res, 0.3, 230 + j), w_read=rand_like(p.w_read, 0.5, 240 + j), w_write=rand_like(p.w_write, 0.5, 250 + j))
          for j, p in enumerate(mm.transformer.layers.stacked.hyper)]
    loss, g = eqx.filter_value_and_grad(lf)(with_hyper(mm, hy))
    assert np.isfinite(float(loss)), loss
    gh = g.transformer.layers.stacked.hyper
    g_old = sum(gsum(p.w_read) + gsum(p.w_write) for p in gh)
    g_phi = sum(gsum(p.rw_phi) + gsum(p.rw_gain) for p in gh)
    g_mix = sum(gsum(p.res) + gsum(p.w_res) for p in gh)
    assert g_old == 0.0 and g_phi > 0 and g_mix > 0, (g_old, g_phi, g_mix)
    print(f"ok 3 {'probes' if detach else 'separate heads'} step: loss {float(loss):.4f}, |grad w_read, w_write| {g_old:.1e}, "
          f"|grad rw_phi, rw_gain| {g_phi:.2e}, |grad res, w_res| {g_mix:.2e}")

# 4. validation -------------------------------------------------------------------------------------------------------
for arch, kw in (("mhc", dict(depth_hc_rwdyn="joint")), ("hc", dict(depth_hc_rwdyn="joint", depth_route="static")),
                 ("hc", dict(depth_hc_rwdyn="joint", depth_hc_rw="sigmoid")), ("hc", dict(depth_hc_rwdyn="joint", depth_hc_write="shared")),
                 ("hc", dict(depth_hc_rwdyn="bogus"))):
    try:
        model(arch, **kw)
        raise AssertionError(f"{arch} {kw} was accepted")
    except ValueError as e:
        print(f"ok 4 rejected: {str(e)[:110]}")
print("ALL OK")
