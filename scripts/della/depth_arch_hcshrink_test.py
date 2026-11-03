"""CPU tests for depth_hc_res="sumshrink" (owner popup 2026-11-03): hc whose mixing only shrinks the stream sum, with fixed gains. Run on
a vis node from this worktree:
    CUDA_VISIBLE_DEVICES= nice -n 19 $PY -u scripts/della/depth_arch_hcshrink_test.py
The states are tested in float32 and in bfloat16, the state's dtype in training (the compute policy is c=bfloat16). The tests of hc_res="sum"
ran in float32 only, so they never measured what bfloat16 storage does to the stream sum; part 2 also measures that.
1. No new parameters. The gains are in the table for 96 sublayers, with the cumulative value the popup stated at layer 12.
2. The mixing alone, writes off, one layer (two sublayers). Float32 and bfloat16: the stream sum gets g0 g1, every difference between streams
   stays, and nothing depends on res or w_res. Bfloat16 errors are printed next to those of hc_res="sum".
3. With random routing including writes, sumshrink differs from hc and from identity mixing.
4. A separate-heads and a probes training step, with parameters cast to the bfloat16 compute dtype as the trainer does, are finite; res and
   w_res get exactly zero gradient; read, write and their dynamic weights get non-zero gradient.
5. sumshrink with another design, more than 48 layers, are rejected.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import dataclasses  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import jmp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402

from experiments.references.depth_arch_qwen3 import HC_SUM_GAIN  # noqa: E402
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
mp = jmp.get_policy("p=f32,c=bfloat16")


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
    """One layer (attention and MLP sublayers) of layer 0; returns H as (stream, batch, position, embed), in float64 for comparing."""
    H = m_.transformer.layer(0).hyper_step(H0, mask, key=None, pos_ids=None, layer=0)
    return np.asarray(H.rearrange((Stream, Batch, Pos, Embed)).array.astype(jnp.float32), np.float64)


m_hc, m_id, m_shr, m_sum = model("hc"), model("hc", depth_hc_res="identity"), model("hc", depth_hc_res="sumshrink"), model("hc", depth_hc_res="sum")

# 1. parameters and the table -------------------------------------------------------------------------------------------------
n = len(jax.tree_util.tree_leaves(m_hc))
assert len(jax.tree_util.tree_leaves(m_shr)) == n, (n, len(jax.tree_util.tree_leaves(m_shr)))
assert len(HC_SUM_GAIN) == 96 and min(HC_SUM_GAIN) > 0
cum12 = float(np.prod(HC_SUM_GAIN[:24]))
assert 3e-5 < cum12 < 1.2e-4, cum12
print(f"ok 1 same {n} leaves as hc; 96 gains, range {min(HC_SUM_GAIN):.3f}-{max(HC_SUM_GAIN):.3f}, cumulative at layer 12 {cum12:.1e}, at layer 48 {float(np.prod(HC_SUM_GAIN)):.1e}")

# 2. the mixing alone, writes off ---------------------------------------------------------------------------------------------
hy = list(m_hc.transformer.layers.stacked.hyper)


def no_write(hy_, seed):
    return [dataclasses.replace(p, read=rand_like(p.read, 0.3, seed + j), res=rand_like(p.res, 0.3, seed + 10 + j), w_read=rand_like(p.w_read, 0.3, seed + 20 + j),
                                w_res=rand_like(p.w_res, 0.3, seed + 30 + j), scale=rand_like(p.scale, 1.0, seed + 40 + j),
                                write=hax.named(jnp.full(p.write.array.shape, -1.0, p.write.array.dtype), p.write.axes),
                                w_write=hax.named(jnp.zeros_like(p.w_write.array), p.w_write.axes)) for j, p in enumerate(hy_)]


nw = no_write(hy, 100)
g01 = HC_SUM_GAIN[0] * HC_SUM_GAIN[1]
H0np = rng.standard_normal((M, B, T, D))
for name, dtype, lim_s, lim_d in (("float32", jnp.float32, 2e-6, 2e-6), ("bfloat16", jnp.bfloat16, 2e-2, 2e-2)):
    H0 = hax.named(jnp.asarray(H0np, dtype), (Stream, Batch, Pos, Embed))
    h0 = np.asarray(H0.array.astype(jnp.float32), np.float64)
    Hs = step0(with_hyper(m_shr, nw), H0)
    assert H0.array.dtype == dtype
    r_sum = rel(Hs.sum(0), g01 * h0.sum(0))
    d_in, d_out = h0 - h0.mean(0, keepdims=True), Hs - Hs.mean(0, keepdims=True)
    r_dif = rel(d_out, d_in)
    assert r_sum <= lim_s, f"{name}: stream sum vs g0 g1 x sum: rel {r_sum:.2e}"
    assert r_dif <= lim_d, f"{name}: differences between streams changed by rel {r_dif:.2e}"
    other = [dataclasses.replace(p, res=rand_like(p.res, 0.7, 300 + j), w_res=rand_like(p.w_res, 0.7, 310 + j)) for j, p in enumerate(nw)]
    r_dep = rel(step0(with_hyper(m_shr, other), H0), Hs)
    assert r_dep == 0.0, f"{name}: depends on res or w_res: rel {r_dep:.2e}"
    print(f"ok 2 {name}, writes off, one layer: sum gain g0 g1 = {g01:.3f} (rel {r_sum:.1e}); differences kept (rel {r_dif:.1e}); no dependence on res, w_res")
    # the same state through hc_res="sum": what storing the state in this dtype does to the sum it should keep, and how far the differences move
    Hq = step0(with_hyper(m_sum, nw), H0)
    d_q = Hq - Hq.mean(0, keepdims=True)
    print(f"   hc_res='sum' here: stream sum rel {rel(Hq.sum(0), h0.sum(0)):.1e}; differences grow by {np.linalg.norm(d_q) / np.linalg.norm(d_in):.2f} times")

# 3. full routing differs from hc and from identity mixing ----------------------------------------------------------------------
rnd = [dataclasses.replace(p, read=rand_like(p.read, 0.5, 10 + j), write=rand_like(p.write, 0.5, 20 + j), res=rand_like(p.res, 0.3, 60 + j),
                           w_read=rand_like(p.w_read, 0.5, 30 + j), w_write=rand_like(p.w_write, 0.5, 40 + j), w_res=rand_like(p.w_res, 0.5, 70 + j),
                           scale=rand_like(p.scale, 1.0, 50 + j)) for j, p in enumerate(hy)]
o = {k: outs(with_hyper(m_, rnd)) for k, m_ in (("hc", m_hc), ("id", m_id), ("shrink", m_shr))}
for a, b in (("shrink", "hc"), ("shrink", "id")):
    d = rel(o[a], o[b])
    assert d > 1e-2, f"{a} should differ from {b}: rel {d:.2e}"
print("ok 3 random routing: sumshrink differs from hc and from identity mixing")

# 4. training steps in the compute dtype ----------------------------------------------------------------------------------------
for detach in (False, True):
    name = "probes" if detach else "separate heads"
    mm = with_hyper(model("hc", depth_hc_res="sumshrink", pls_detach_backbone=detach), rnd)
    mc = mp.cast_to_compute(mm)
    assert any(a.dtype == jnp.bfloat16 for a in jax.tree_util.tree_leaves(eqx.filter(mc, eqx.is_inexact_array))), "the cast left no bfloat16 parameter"
    loss, gr = eqx.filter_value_and_grad(lf)(mc)
    assert np.isfinite(float(loss)), loss
    gh = gr.transformer.layers.stacked.hyper
    s = lambda f: sum(float(jnp.sum(jnp.abs(getattr(p, f).array.astype(jnp.float32)))) for p in gh)  # noqa: E731
    g_mix, g_rw, g_dyn = s("res") + s("w_res"), s("read") + s("write"), s("w_read") + s("w_write")
    assert g_mix == 0.0 and g_rw > 0 and g_dyn > 0, (g_mix, g_rw, g_dyn)
    print(f"ok 4 {name} step, bfloat16 compute: loss {float(loss):.4f}, |grad res, w_res| {g_mix:.1e}, |grad read, write| {g_rw:.2e}, |grad w_read, w_write| {g_dyn:.2e}")

# 5. validation -----------------------------------------------------------------------------------------------------------------
for arch, kw in (("mhc", dict(depth_hc_res="sumshrink")), ("preln", dict(depth_hc_res="sumshrink")), ("hc", dict(depth_hc_res="sumshrink", num_layers=49))):
    try:
        model(arch, **kw)
        raise AssertionError(f"{arch} {kw} was accepted")
    except ValueError as e:
        print(f"ok 5 rejected: {str(e)[:110]}")
print("ALL OK")
