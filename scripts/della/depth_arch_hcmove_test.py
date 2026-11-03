"""CPU tests for the three designs of the owner popup of 2026-11-03: depth_hc_res="move" (hc whose mixing only moves content between streams),
depth_shrink_power (half the dose of the fixed gains) and depth_preln_leak (Pre-LN with h <- g h + F). Run on a vis node from this worktree:
    CUDA_VISIBLE_DEVICES= nice -n 19 $PY -u scripts/della/depth_arch_hcmove_test.py
The states are tested in float32 and in bfloat16, the state's dtype in training.
1. move: no new parameters, and at init it computes hc's layer outputs. Equal streams pass this for any row-stochastic P, so the next check gives the
   streams different contents: with init parameters and writes off, one layer of move keeps the differences as hc does (audit 2026-11-03: gain 0.765
   per layer without HC_MOVE_SHIFT).
2. move, static mixing only, writes off, one layer: H_out equals the formula, the stream sum stays, and many layers in a row do not grow the state
   (hc_res="sum" with the same parameters does, in the same loop). A row with no positive entry keeps its own stream.
3. Half dose: the stream sum gets (g0 g1) ** 0.5, the differences stay.
4. Pre-LN leak: the default Pre-LN has the same leaves and is unchanged at depth_preln_leak=False; one layer equals the formula h <- g h + F with
   the branches computed by hand, with g ** p for p = 1 and p = 0.5.
5. A separate-heads and a probes training step, with parameters cast to the bfloat16 compute dtype, are finite for all three. move: res and w_res
   get non-zero gradient.
6. Wrong combinations are rejected.
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

from experiments.references.depth_arch_qwen3 import HC_MOVE_BETA, HC_MOVE_SHIFT, HC_SUM_GAIN  # noqa: E402
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


def step(m_, H0, layer=0):
    H = m_.transformer.layer(layer).hyper_step(H0, mask, key=None, pos_ids=None, layer=layer)
    return H.rearrange((Stream, Batch, Pos, Embed))


def arr(H):
    return np.asarray(H.array.astype(jnp.float32), np.float64)


m_hc, m_mv, m_sum = model("hc"), model("hc", depth_hc_res="move"), model("hc", depth_hc_res="sum")

# 1. move at init --------------------------------------------------------------------------------------------------------------
n = len(jax.tree_util.tree_leaves(m_hc))
assert len(jax.tree_util.tree_leaves(m_mv)) == n
r = rel(outs(m_mv), outs(m_hc))
assert r <= 1e-5, r
print(f"ok 1 move has the same {n} leaves as hc; at init it computes hc's layer outputs (rel {r:.1e})")
Hd_np = rng.standard_normal((M, B, T, D))
Hd = hax.named(jnp.asarray(Hd_np, jnp.float32), (Stream, Batch, Pos, Embed))
init_nw = [dataclasses.replace(p, write=hax.named(jnp.full(p.write.array.shape, -1.0, p.write.array.dtype), p.write.axes)) for p in m_hc.transformer.layers.stacked.hyper]
d_in = Hd_np - Hd_np.mean(0, keepdims=True)
gain = {}
for name_, m_ in (("hc", m_hc), ("move", m_mv)):
    o_ = np.asarray(step(with_hyper(m_, init_nw), Hd).array, np.float64)
    gain[name_] = float(np.linalg.norm(o_ - o_.mean(0, keepdims=True)) / np.linalg.norm(d_in))
    assert abs(float(np.linalg.norm(o_.sum(0) - Hd_np.sum(0)) / np.linalg.norm(Hd_np.sum(0)))) < 1e-5, name_
assert abs(gain["hc"] - 1.0) < 1e-5 and gain["move"] >= 0.998, gain
print(f"ok 1 init parameters, different streams, writes off, one layer: the differences get gain {gain['move']:.4f} in move, {gain['hc']:.4f} in hc; the stream sum stays")


# 2. move, static mixing, writes off -------------------------------------------------------------------------------------------
def static_no_write(hy_, seed, res_scale=0.3):
    return [dataclasses.replace(p, read=rand_like(p.read, 0.3, seed + j), res=rand_like(p.res, res_scale, seed + 10 + j), w_read=rand_like(p.w_read, 0.3, seed + 20 + j),
                                w_res=rand_like(p.w_res, 0.0, seed + 30 + j), scale=hax.named(jnp.zeros_like(p.scale.array), p.scale.axes),
                                write=hax.named(jnp.full(p.write.array.shape, -1.0, p.write.array.dtype), p.write.axes),
                                w_write=hax.named(jnp.zeros_like(p.w_write.array), p.w_write.axes)) for j, p in enumerate(hy_)]


hy = list(m_hc.transformer.layers.stacked.hyper)
nw = static_no_write(hy, 100)


H0np = rng.standard_normal((M, B, T, D))
for name, dtype, lim_f, lim_s in (("float32", jnp.float32, 2e-6, 2e-6), ("bfloat16", jnp.bfloat16, 2e-2, 2e-2)):
    H0 = hax.named(jnp.asarray(H0np, dtype), (Stream, Batch, Pos, Embed))
    h0 = arr(H0)
    # as the trainer does: parameters in the compute dtype, so the writes and the stored state stay in that dtype
    mc = with_hyper(m_mv, nw) if dtype == jnp.float32 else mp.cast_to_compute(with_hyper(m_mv, nw))
    hc_ = list(mc.transformer.layers.stacked.hyper)
    def P_c(p):
        Ln = [a.name for a in p.res.axes if a.name not in ("stream", "stream_out")][0]
        a = np.logaddexp(0.0, HC_MOVE_BETA * (np.eye(M) + np.asarray(p.res.rearrange((Ln, "stream", "stream_out")).array[0].astype(jnp.float32), np.float64) - HC_MOVE_SHIFT)) / HC_MOVE_BETA + 1e-6 * np.eye(M)
        return a / a.sum(1, keepdims=True)
    P0, P1 = P_c(hc_[0]), P_c(hc_[1])
    assert (P0 >= 0).all() and (P1 >= 0).all() and abs(P0.sum(1) - 1).max() < 1e-9
    out = step(mc, H0)
    assert out.array.dtype == dtype, out.array.dtype
    expect = np.einsum("oi,i...->o...", P1.T, np.einsum("oi,i...->o...", P0.T, h0))
    r_f, r_s = rel(arr(out), expect), rel(arr(out).sum(0), h0.sum(0))
    assert r_f <= lim_f and r_s <= lim_s, (name, r_f, r_s)
    moved = rel(arr(out), h0)
    assert moved > 1e-2, moved
    print(f"ok 2 {name}, static mixing, writes off, one layer: H_out equals the formula (rel {r_f:.1e}); the stream sum stays (rel {r_s:.1e}); state moved (rel {moved:.2e})")
# many layers in a row: move stays bounded, hc_res="sum" with the same parameters grows
big = static_no_write(hy, 500, res_scale=0.5)
H_mv = H_su = hax.named(jnp.asarray(H0np, jnp.float32), (Stream, Batch, Pos, Embed))
for _ in range(24):
    H_mv = step(with_hyper(m_mv, big), H_mv)
    H_su = step(with_hyper(m_sum, big), H_su)
g_mv, g_su = np.linalg.norm(arr(H_mv)) / np.linalg.norm(H0np), np.linalg.norm(arr(H_su)) / np.linalg.norm(H0np)
assert g_mv <= 2.0 and g_su > 10.0, (g_mv, g_su)
print(f"ok 2 static mixing, 24 layers in a row: state norm x {g_mv:.2f} for move, x {g_su:.2e} for hc_res='sum'")
# a row far below 0 keeps its own stream
neg = [dataclasses.replace(p, res=hax.named(jnp.full(p.res.array.shape, -5.0, p.res.array.dtype), p.res.axes)) for p in nw]
H0f = hax.named(jnp.asarray(H0np, jnp.float32), (Stream, Batch, Pos, Embed))
r_n = rel(arr(step(with_hyper(m_mv, neg), H0f)), H0np)
assert r_n <= 1e-6, r_n
print(f"ok 2 every entry far below 0: every stream keeps itself (rel {r_n:.1e})")

# 3. half dose -----------------------------------------------------------------------------------------------------------------
m_half = model("hc", depth_hc_res="sumshrink", depth_shrink_power=0.5)
out = arr(step(with_hyper(m_half, nw), H0f))
g = (HC_SUM_GAIN[0] * HC_SUM_GAIN[1]) ** 0.5
r_s, r_d = rel(out.sum(0), g * H0np.sum(0)), rel(out - out.mean(0, keepdims=True), H0np - H0np.mean(0, keepdims=True))
assert r_s <= 2e-6 and r_d <= 2e-6, (r_s, r_d)
print(f"ok 3 half dose: stream sum gain (g0 g1) ** 0.5 = {g:.3f} (rel {r_s:.1e}); differences stay (rel {r_d:.1e})")

# 4. Pre-LN leak ---------------------------------------------------------------------------------------------------------------
m_pl, m_lk, m_lk5 = model("preln"), model("preln", depth_preln_leak=True), model("preln", depth_preln_leak=True, depth_shrink_power=0.5)
assert len(jax.tree_util.tree_leaves(m_pl)) == len(jax.tree_util.tree_leaves(m_lk))
r_d = rel(outs(m_lk), outs(m_pl))
assert r_d > 1e-3, r_d
for m_, p_, name in ((m_lk, 1.0, "p = 1"), (m_lk5, 0.5, "p = 0.5")):
    lay = m_.transformer.layer(1)
    h = hax.named(jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32), (Batch, Pos, Embed))
    h_manual = h
    for j in (0, 1):
        f = lay.branch(j, h_manual, mask, key=None, pos_ids=None, layer=1)
        h_manual = (float(HC_SUM_GAIN[2 + j]) ** p_) * h_manual + f
    h_out = lay.residual_step(h, mask, key=None, pos_ids=None, layer=1)
    r_m = rel(np.asarray(h_out.rearrange((Batch, Pos, Embed)).array), np.asarray(h_manual.rearrange((Batch, Pos, Embed)).array))
    assert r_m <= 2e-5, (name, r_m)
    print(f"ok 4 Pre-LN leak, {name}: layer 2 equals h <- g h + F with g ** p (rel {r_m:.1e})")

# 5. training steps in the compute dtype ---------------------------------------------------------------------------------------
rnd = [dataclasses.replace(p, read=rand_like(p.read, 0.5, 10 + j), write=rand_like(p.write, 0.5, 20 + j), res=rand_like(p.res, 0.3, 60 + j),
                           w_read=rand_like(p.w_read, 0.5, 30 + j), w_write=rand_like(p.w_write, 0.5, 40 + j), w_res=rand_like(p.w_res, 0.5, 70 + j),
                           scale=rand_like(p.scale, 1.0, 50 + j)) for j, p in enumerate(hy)]
for detach in (False, True):
    name = "probes" if detach else "separate heads"
    mm = with_hyper(model("hc", depth_hc_res="move", pls_detach_backbone=detach), rnd)
    loss, gr = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(mm))
    assert np.isfinite(float(loss)), loss
    gh = gr.transformer.layers.stacked.hyper
    s = lambda f: sum(float(jnp.sum(jnp.abs(getattr(p, f).array.astype(jnp.float32)))) for p in gh)  # noqa: E731
    assert s("res") > 0 and s("w_res") > 0, (s("res"), s("w_res"))
    # at init too: soft non-negativity leaves a gradient on every entry
    loss0, gr0 = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(model("hc", depth_hc_res="move", pls_detach_backbone=detach)))
    gh0 = gr0.transformer.layers.stacked.hyper
    s0 = sum(float(jnp.sum(jnp.abs(p.res.array.astype(jnp.float32)))) for p in gh0)
    assert s0 > 0, s0
    print(f"ok 5 move, {name} step: loss {float(loss):.4f}, |grad res| {s('res'):.2e}, |grad w_res| {s('w_res'):.2e}; at init loss {float(loss0):.4f}, |grad res| {s0:.2e}")
    for tag, kw in (("half dose", dict(depth_hc_res="sumshrink", depth_shrink_power=0.5)),):
        l_, _ = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(with_hyper(model("hc", pls_detach_backbone=detach, **kw), rnd)))
        assert np.isfinite(float(l_)), (tag, l_)
        print(f"ok 5 {tag}, {name} step: loss {float(l_):.4f}")
    l_, gp = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(model("preln", depth_preln_leak=True, pls_detach_backbone=detach)))
    assert np.isfinite(float(l_)), l_
    print(f"ok 5 Pre-LN leak, {name} step: loss {float(l_):.4f}")

# 6. validation ----------------------------------------------------------------------------------------------------------------
for arch, kw in (("preln", dict(depth_hc_res="move")), ("hc", dict(depth_preln_leak=True)), ("hc", dict(depth_shrink_power=0.5)), ("preln", dict(depth_shrink_power=0.5)),
                 ("preln", dict(depth_preln_leak=True, depth_shrink_power=0.0)), ("preln", dict(depth_preln_leak=True, num_layers=49))):
    try:
        model(arch, **kw)
        raise AssertionError(f"{arch} {kw} was accepted")
    except ValueError as e:
        print(f"ok 6 rejected: {str(e)[:110]}")
print("ALL OK")
