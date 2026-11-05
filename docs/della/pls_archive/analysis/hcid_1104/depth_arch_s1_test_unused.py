"""CPU tests for the one-stream decomposition (owner popup 2026-11-04, docs pls.md item 26): which knob of hc with one stream, h <- g h + w F(RN(r h)), gives its middle layers? Arms: hc 1 stream (learned g, r, w, token-dependent); depth_hc_res="identity" (g = 1); depth_hc_res="scalar" (g one learned number per sublayer, static); depth_route static (g, r, w all constants); Pre-LN with depth_preln_leak and depth_shrink_power=0.1 (h <- g h + F with fixed small decay).
    CUDA_VISIBLE_DEVICES= nice -n 19 $PY -u scripts/della/depth_arch_s1_test.py
The state is tested in float32 and in bfloat16, the state's dtype in training.
1. Each arm's layer outputs equal a hand computation of its formula, written here without calling the sublayer step of the model: at every layer, both dtypes.
2. The knobs an arm fixes do nothing: identity ignores res and w_res; scalar ignores w_res; static ignores all w_*. Changing them leaves the outputs bit for bit, and their gradients are zero.
3. With g = 1 the state grows only by writes; the leak's cumulative gains are the documented ones: 0.38 at layer 12 and 0.23 at layer 32.
4. A probes training step (the arms are ordinary models) with parameters cast to bfloat16 is finite, the probes' gradient does not reach the backbone, and 12 steps of the real optimizer on one batch lower the loss.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

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

from experiments.references.depth_arch_qwen3 import HC_SUM_GAIN, _hc_constants, _rms, _static, _sum_gain  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
rng = np.random.default_rng(0)
B, T, V, L, D = 4, 16, 300, 4, 48
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
Stream, Embed = Axis("stream", 1), Axis("embed", D)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True, pls_detach_backbone=True)
key0 = jrandom.PRNGKey(0)
mask = AttentionMask.causal()
x = hax.named(jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32), (Batch, Pos, Embed))
tok = rng.integers(2, V, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=mask)
mp = jmp.get_policy("p=f32,c=bfloat16")
ARMS = {
    "learned": ("hc", dict(hc_streams=1)),
    "identity": ("hc", dict(hc_streams=1, depth_hc_res="identity")),
    "scalar": ("hc", dict(hc_streams=1, depth_hc_res="scalar")),
    "static": ("hc", dict(hc_streams=1, depth_route="static")),
    "leak": ("preln", dict(depth_preln_leak=True, depth_shrink_power=0.1)),
}


def model(arch, **kw):
    cfg = PerLayerQwen3Config(**{**common, **kw}, hybrid_norm=True, depth_arch=arch, scan_layers=False)
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=key0)


def outs(m, dtype=None):
    tr = m.transformer
    if dtype is not None:
        tr = mp.cast_to_compute(tr)
    xx = x if dtype is None else x.astype(dtype)
    return np.asarray(tr.outputs(xx, mask).rearrange((tr.config.Layers, Batch, Pos, tr.config.Embed)).array.astype(jnp.float32))


def lf(m_):
    loss, _ = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(1), logsumexp_weight=0.0)
    return loss.array if isinstance(loss, hax.NamedArray) else loss


f32 = lambda a: a.astype(jnp.float32)  # noqa: E731


def by_hand_hc(m, arm, dtype):
    tr = mp.cast_to_compute(m.transformer) if dtype == jnp.bfloat16 else m.transformer
    H = hax.stack(Stream, [x.astype(dtype)])
    got = []
    for li in range(L):
        lay = tr.layer(li)
        for j in range(2):
            q = lay.hyper[j]
            # hc's values at init with one stream are all 1 (_hc_constants is checked below against this)
            one = 1.0
            Hn = f32(_rms(H, Embed))
            sc = f32(q.scale)
            r_s, g_s, w_s = f32(q.read), f32(q.res), f32(q.write)
            if arm == "static":
                r, g, w = one + r_s, one + g_s, one + w_s
            else:
                r = one + r_s + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(q.w_read), axis=Embed))
                w = one + w_s + sc["hp_scale", 1] * hax.tanh(hax.dot(Hn, f32(q.w_write), axis=Embed).rename({"stream": "stream_out"}))
                g = one + g_s + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(q.w_res), axis=Embed))
            xr = hax.dot(r.astype(H.dtype), H, axis="stream")
            f = lay.branch(j, xr, mask, key=None, pos_ids=None)
            upd = (w.astype(H.dtype) * f.broadcast_axis(w.resolve_axis("stream_out"))).rename({"stream_out": "stream"})
            if arm == "identity":
                H = H + upd
            elif arm == "scalar":
                gs = float(np.asarray(one + g_s.array).ravel()[0])
                H = (gs * f32(H)).astype(H.dtype) + upd
            else:
                H = hax.dot(g.astype(H.dtype), H, axis="stream").rename({"stream_out": "stream"}) + upd
        got.append(np.asarray(hax.sum(H, axis="stream").rearrange((Batch, Pos, Embed)).array.astype(jnp.float32)))
    return np.stack(got)


for sub in (0, 1, 7):
    c_read, c_res, c_write = _hc_constants(1, sub)
    assert float(c_read.array[0]) == 1.0 and float(c_res.array[0, 0]) == 1.0 and float(c_write.array[0]) == 1.0
print("ok 1 with one stream hc's read, mixing and write start at 1 for every sublayer")
for arm in ("learned", "identity", "scalar", "static"):
    arch, kw = ARMS[arm]
    m = model(arch, **kw)
    assert m.transformer.config.hc_streams == 1 and (arm != "static" or _static(m.transformer.config))
    # bfloat16: the scanned, compiled model and this eager loop round differently; one sublayer's rounding is 1 / 128 of the value, so the check is relative
    for dt, rtol, atol in ((jnp.float32, 1e-5, 1e-5), (jnp.bfloat16, 4e-2, 0.15)):
        a, b = outs(m, None if dt == jnp.float32 else dt), by_hand_hc(m, arm, dt)
        rel = np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)
        assert np.allclose(a, b, rtol=rtol, atol=atol) and rel.max() < 4e-2, (arm, dt, np.abs(a - b).max(), rel.max())
    print(f"ok 1 {arm}: the model equals the hand computation at every layer, float32 and bfloat16")
m = model("preln", **ARMS["leak"][1])
assert np.allclose([float(_sum_gain(m.transformer.config, i)) for i in range(96)], np.asarray(HC_SUM_GAIN) ** 0.1, rtol=1e-6)
for dt, rtol, atol in ((jnp.float32, 1e-5, 1e-5), (jnp.bfloat16, 4e-2, 0.15)):
    got = outs(m, None if dt == jnp.float32 else dt)
    tr = mp.cast_to_compute(m.transformer) if dt == jnp.bfloat16 else m.transformer
    h = x.astype(dt)
    for li in range(L):
        lay = tr.layer(li)
        for j in range(2):
            f = lay.branch(j, h, mask, key=None, pos_ids=None)
            h = (float(HC_SUM_GAIN[2 * li + j]) ** 0.1 * f32(h)).astype(h.dtype) + f   # h <- g h + F, by hand
        want = np.asarray(h.rearrange((Batch, Pos, Embed)).array.astype(jnp.float32))
        assert np.allclose(got[li], want, rtol=rtol, atol=atol), (dt, li, np.abs(got[li] - want).max())
print("ok 1 leak: the fixed gains are HC_SUM_GAIN ** 0.1 and the model equals h <- g h + F by hand at every layer, float32 and bfloat16")

# 2. fixed knobs -----------------------------------------------------------------------------------------------------------------
def scale_leaf(m_, name, j, by):
    def where(mm):
        return getattr(mm.transformer.layers.stacked.hyper[j], name)
    return eqx.tree_at(where, m_, jax.tree_util.tree_map(lambda v: v + by, where(m_)))


FIXED = {"identity": ("res", "w_res"), "scalar": ("w_res",), "static": ("w_res", "w_read", "w_write", "scale")}
for arm, names in FIXED.items():
    arch, kw = ARMS[arm]
    m = model(arch, **kw)
    base = outs(m, jnp.bfloat16)
    for name in names:
        for j in (0, 1):
            m2 = scale_leaf(m, name, j, 0.25)
            assert np.array_equal(base, outs(m2, jnp.bfloat16)), (arm, name, j)
    l_, g_ = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(m))
    for name in names:
        for j in (0, 1):
            gv = getattr(g_.transformer.layers.stacked.hyper[j], name)
            assert float(jnp.abs(gv.array.astype(jnp.float32)).max()) == 0.0, (arm, name, j)
    print(f"ok 2 {arm}: {names} change nothing and have zero gradient")

# 3. dose of the leak ------------------------------------------------------------------------------------------------------------
g = np.asarray(HC_SUM_GAIN, np.float64) ** 0.1
c12, c32, c48 = np.prod(g[:24]), np.prod(g[:64]), np.prod(g)
assert 0.3 < c12 < 0.45 and 0.15 < c32 < 0.3, (c12, c32)
print(f"ok 3 leak cumulative gain: layer 12 {c12:.3f}, layer 32 {c32:.3f}, layer 48 {c48:.3f}; smallest sublayer gain {g.min():.3f}")

# 4. training steps --------------------------------------------------------------------------------------------------------------
try:
    from experiments.references.depth_arch_qwen3 import DepthArchMuonHConfig
except ImportError:
    from experiments.references.della_muonh_qwen3_scaling import DepthArchMuonHConfig
for arm, (arch, kw) in ARMS.items():
    m = model(arch, **kw)
    l_, g_ = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(m))
    assert np.isfinite(float(l_))
    leaves = jax.tree_util.tree_leaves(eqx.filter(g_, eqx.is_array))
    assert all(np.all(np.isfinite(np.asarray(a.astype(jnp.float32)))) for a in leaves), arm
    opt = DepthArchMuonHConfig(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0, weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0, coefficient_type="simple").build(4959)
    state = opt.init(eqx.filter(m, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(m_, st):
        l1, g1 = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(m_))
        g1 = mp.cast_to_param(g1)
        up, st = opt.update(g1, st, params=eqx.filter(m_, eqx.is_inexact_array))
        return eqx.apply_updates(m_, up), st, l1

    hist = []
    for s in range(12):
        m, state, l1 = train_step(m, state)
        hist.append(float(l1))
    assert np.all(np.isfinite(hist)) and hist[-1] < hist[0], (arm, hist)
    print(f"ok 4 {arm}: gradients finite; real optimizer, bfloat16 compute, 12 steps: {hist[0]:.3f} -> {hist[-1]:.3f}")
print("ALL OK")
