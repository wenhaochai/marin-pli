"""CPU tests for depth_hc_res="sum" and "scalar" (owner popup 2026-11-02): hc whose mixing keeps the stream sum, and hc whose
mixing is one learned number per sublayer. Run on a vis node from this worktree:
    CUDA_VISIBLE_DEVICES= nice -n 19 $PY -u scripts/della/depth_arch_hcsum_test.py
1. No new parameters, and at init both compute hc's layer outputs (hc's res is I).
2. Exact behaviour of one sublayer pair, with random routing and with the writes switched off (write = -1, w_write = 0):
   "sum" leaves the stream sum unchanged, hc does not; "sum" still moves state between streams; "scalar" gives H_out = g H_in
   for every stream with g the mean diagonal of I + res. "scalar" does not depend on w_res or on the off-diagonal res.
3. With random routing including writes, both differ from hc and from identity mixing.
4. A separate-heads and a probes training step are finite. "sum": res and w_res get non-zero gradient. "scalar": the diagonal
   of res gets non-zero gradient; the off-diagonal res and w_res get exactly zero.
5. "sum" or "scalar" with another design, and an unknown value, are rejected.
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
Stream, Embed = Axis("stream", M), Axis("embed", D)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True)
key0 = jrandom.PRNGKey(0)
mask = AttentionMask.causal()
x = hax.named(jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32), (Batch, Pos, Embed))
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


def step0(m_, H0):
    """One layer (attention and MLP sublayers) of layer 0; returns H as (stream, batch, position, embed)."""
    H = m_.transformer.layer(0).hyper_step(H0, mask, key=None, pos_ids=None, layer=0)
    return np.asarray(H.rearrange((Stream, Batch, Pos, Embed)).array)


m_hc, m_sum, m_sca, m_id = model("hc"), model("hc", depth_hc_res="sum"), model("hc", depth_hc_res="scalar"), model("hc", depth_hc_res="identity")

# 1. no new parameters, hc's function at init -------------------------------------------------------------------------------
n = len(jax.tree_util.tree_leaves(m_hc))
assert len(jax.tree_util.tree_leaves(m_sum)) == n and len(jax.tree_util.tree_leaves(m_sca)) == n
o_hc = outs(m_hc)
r_s, r_c = rel(outs(m_sum), o_hc), rel(outs(m_sca), o_hc)
assert r_s <= 1e-5 and r_c <= 1e-5, (r_s, r_c)
print(f"ok 1 same {n} leaves as hc; at init sum equals hc (rel {r_s:.1e}), scalar equals hc (rel {r_c:.1e})")

# 2. the mixing alone, writes off -------------------------------------------------------------------------------------------
hy = list(m_hc.transformer.layers.stacked.hyper)
def no_write(hy_, seed):
    out = []
    for j, p in enumerate(hy_):
        out.append(dataclasses.replace(p, read=rand_like(p.read, 0.3, seed + j), res=rand_like(p.res, 0.3, seed + 10 + j), w_read=rand_like(p.w_read, 0.3, seed + 20 + j),
                                       w_res=rand_like(p.w_res, 0.3, seed + 30 + j), scale=rand_like(p.scale, 1.0, seed + 40 + j),
                                       write=hax.named(jnp.full(p.write.array.shape, -1.0, p.write.array.dtype), p.write.axes),
                                       w_write=hax.named(jnp.zeros_like(p.w_write.array), p.w_write.axes)))
    return out
nw = no_write(hy, 100)
H0 = hax.named(jnp.asarray(rng.standard_normal((M, B, T, D)), jnp.float32), (Stream, Batch, Pos, Embed))
H0n = np.asarray(H0.array)
Hs, Hh, Hc = step0(with_hyper(m_sum, nw), H0), step0(with_hyper(m_hc, nw), H0), step0(with_hyper(m_sca, nw), H0)
keep_s, keep_h = rel(Hs.sum(0), H0n.sum(0)), rel(Hh.sum(0), H0n.sum(0))
moved = rel(Hs, H0n)
assert keep_s <= 2e-5, f"sum: the stream sum changed by rel {keep_s:.2e}"
assert keep_h > 1e-2, f"hc with random mixing should change the stream sum: rel {keep_h:.2e}"
assert moved > 1e-2, f"sum should still move state between streams: rel {moved:.2e}"
print(f"ok 2 writes off, one layer: sum keeps the stream sum (rel {keep_s:.1e}; hc changes it, rel {keep_h:.2e}) and moves state between streams (rel {moved:.2e})")
g = 1.0
for p in nw[:2]:   # the two sublayers of layer 0
    res0 = np.asarray(p.res.rearrange(("layer", "stream", "stream_out")).array[0], np.float64) if "layer" in [a.name for a in p.res.axes] else None
    if res0 is None:
        Ln = [a.name for a in p.res.axes if a.name not in ("stream", "stream_out")][0]
        res0 = np.asarray(p.res.rearrange((Ln, "stream", "stream_out")).array[0], np.float64)
    g *= 1.0 + np.trace(res0) / M
r_g = rel(Hc, g * H0n)
assert r_g <= 2e-5, f"scalar: H_out vs g H_in, rel {r_g:.2e}"
def other_offdiag(p, k):
    Ln = [a.name for a in p.res.axes if a.name not in ("stream", "stream_out")][0]
    ax = (Ln, "stream", "stream_out")
    a = np.array(p.res.rearrange(ax).array)
    r = np.random.default_rng(k).standard_normal(a.shape).astype(a.dtype)
    a[:, ~np.eye(M, dtype=bool)] = r[:, ~np.eye(M, dtype=bool)]
    return hax.named(jnp.asarray(a), ax).rearrange(p.res.axes)
nw2 = [dataclasses.replace(p, w_res=rand_like(p.w_res, 0.7, 300 + j), res=other_offdiag(p, 310 + j)) for j, p in enumerate(nw)]
r_w = rel(step0(with_hyper(m_sca, nw2), H0), Hc)
assert r_w == 0.0, f"scalar depends on w_res or the off-diagonal res: rel {r_w:.2e}"
print(f"ok 2 writes off, one layer: scalar gives H_out = g H_in, g = {g:.3f} (rel {r_g:.1e}); no dependence on w_res or the off-diagonal res")

# 3. full routing differs from hc and from identity mixing ------------------------------------------------------------------
rnd = [dataclasses.replace(p, read=rand_like(p.read, 0.5, 10 + j), write=rand_like(p.write, 0.5, 20 + j), res=rand_like(p.res, 0.3, 60 + j),
                           w_read=rand_like(p.w_read, 0.5, 30 + j), w_write=rand_like(p.w_write, 0.5, 40 + j), w_res=rand_like(p.w_res, 0.5, 70 + j),
                           scale=rand_like(p.scale, 1.0, 50 + j)) for j, p in enumerate(hy)]
o = {k: outs(with_hyper(m_, rnd)) for k, m_ in (("hc", m_hc), ("id", m_id), ("sum", m_sum), ("scalar", m_sca))}
for a, b in (("sum", "hc"), ("sum", "id"), ("scalar", "hc"), ("scalar", "id"), ("sum", "scalar")):
    d = rel(o[a], o[b])
    assert d > 1e-2, f"{a} should differ from {b}: rel {d:.2e}"
print("ok 3 random routing: sum, scalar, hc and identity mixing all differ")

# 4. training steps ---------------------------------------------------------------------------------------------------------
def grads(mode, detach):
    mm = with_hyper(model("hc", depth_hc_res=mode, pls_detach_backbone=detach), rnd)
    loss, gr = eqx.filter_value_and_grad(lf)(mm)
    assert np.isfinite(float(loss)), loss
    return float(loss), gr.transformer.layers.stacked.hyper
for detach in (False, True):
    name = "probes" if detach else "separate heads"
    loss, gh = grads("sum", detach)
    g_res = sum(float(jnp.sum(jnp.abs(p.res.array))) for p in gh); g_w = sum(float(jnp.sum(jnp.abs(p.w_res.array))) for p in gh)
    assert g_res > 0 and g_w > 0, (g_res, g_w)
    print(f"ok 4 sum, {name} step: loss {loss:.4f}, |grad res| {g_res:.2e}, |grad w_res| {g_w:.2e}")
    loss, gh = grads("scalar", detach)
    eye = np.eye(M, dtype=bool)
    def part(p, diag):
        Ln = [a.name for a in p.res.axes if a.name not in ("stream", "stream_out")][0]
        a = np.asarray(p.res.rearrange((Ln, "stream", "stream_out")).array)
        return float(np.abs(a[:, eye if diag else ~eye]).sum())
    g_d, g_o = sum(part(p, True) for p in gh), sum(part(p, False) for p in gh)
    g_w = sum(float(jnp.sum(jnp.abs(p.w_res.array))) for p in gh)
    assert g_d > 0 and g_o == 0.0 and g_w == 0.0, (g_d, g_o, g_w)
    print(f"ok 4 scalar, {name} step: loss {loss:.4f}, |grad res diagonal| {g_d:.2e}, |grad res off-diagonal| {g_o:.1e}, |grad w_res| {g_w:.1e}")

# 5. validation -------------------------------------------------------------------------------------------------------------
for arch, kw in (("mhc", dict(depth_hc_res="sum")), ("preln", dict(depth_hc_res="scalar")), ("hc", dict(depth_hc_res="bogus"))):
    try:
        model(arch, **kw)
        raise AssertionError(f"{arch} {kw} was accepted")
    except ValueError as e:
        print(f"ok 5 rejected: {str(e)[:110]}")
print("ALL OK")
