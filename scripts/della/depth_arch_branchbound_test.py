"""CPU tests for DepthConfig.depth_branch_bound (hc: per token, a branch output with norm above the bound is scaled down to the bound, before the write; docs pls.md item 22). Run on a vis node from this worktree:
    CUDA_VISIBLE_DEVICES= nice -n 19 $PY -u scripts/della/depth_arch_branchbound_test.py
The state is tested in float32 and in bfloat16, the state's dtype in training.
1. _bound_norm: below the bound the value is unchanged, bit for bit, in both dtypes. Above it the norm is the bound and the direction stays. A zero vector stays zero with finite gradient. Each token is scaled alone.
2. The default (0) adds no leaves and computes the same layer outputs as without the option; a bound far above every branch output computes the same outputs bit for bit (identity mixing and learned mixing), with the same gradients.
3. A small bound: in one hand-computed layer of hc with mixing = I, the update of each sublayer is write x (bounded F): the model's outputs equal the hand computation. |F| at each sublayer is at most the bound: the hook recomputes F.
4. Gradient through the bound: for a token above the bound, moving the input along the radial direction of F gives zero change in F's norm: the numerical and the autodiff directional derivatives agree.
5. A separate-heads and a probes training step with parameters cast to bfloat16 are finite, with a small bound, and a few steps of the real optimizer (DepthArchMuonHConfig with the 130m numbers) stay finite and lower the loss on one batch.
6. Wrong combinations are rejected.
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

from experiments.references.depth_arch_qwen3 import _bound_norm, _hc_constants, _rms  # noqa: E402
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


def model(arch="hc", **kw):
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


# 1. the bound itself -------------------------------------------------------------------------------------------------------
for dt in (jnp.float32, jnp.bfloat16):
    v = jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32)
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True) * jnp.asarray(rng.uniform(0.5, 50.0, size=(B, T, 1)), jnp.float32)
    f = hax.named(v, (Batch, Pos, Embed)).astype(dt)
    g = _bound_norm(f, Embed, 1000.0)
    assert g.dtype == dt and g.axes == f.axes
    assert np.array_equal(np.asarray(g.array.astype(jnp.float32)), np.asarray(f.array.astype(jnp.float32)))
    b = 10.0
    g = _bound_norm(f, Embed, b)
    n0 = np.linalg.norm(np.asarray(f.array.astype(jnp.float32), np.float64), axis=-1)
    n1 = np.linalg.norm(np.asarray(g.array.astype(jnp.float32), np.float64), axis=-1)
    tol = 1e-5 if dt == jnp.float32 else 2e-2
    assert np.all(n1[n0 > b * (1 + tol)] <= b * (1 + tol)) and np.all(n1[n0 > b * (1 + tol)] >= b * (1 - tol)), (n1.max(), n1.min())
    below = n0 < b * (1 - tol)
    assert np.array_equal(np.asarray(g.array.astype(jnp.float32))[below], np.asarray(f.array.astype(jnp.float32))[below])
    cosv = np.sum(np.asarray(g.array.astype(jnp.float32), np.float64) * np.asarray(f.array.astype(jnp.float32), np.float64), -1) / (n0 * n1)
    assert cosv.min() > 1 - (1e-5 if dt == jnp.float32 else 1e-2), cosv.min()
    print(f"ok 1 {dt.__name__ if hasattr(dt, '__name__') else dt}: below unchanged ({int(below.sum())} tokens), above at the bound ({int((n0 > b).sum())} tokens), norm range {n1.min():.3f}-{n1.max():.3f}, min cosine {cosv.min():.6f}")
    z = hax.named(jnp.zeros((B, T, D), dt), (Batch, Pos, Embed))
    assert np.all(np.asarray(_bound_norm(z, Embed, b).array.astype(jnp.float32)) == 0)
    gz = jax.grad(lambda a: jnp.sum(_bound_norm(hax.named(a, (Batch, Pos, Embed)), Embed, b).array.astype(jnp.float32) ** 2))(jnp.zeros((B, T, D), jnp.float32))
    assert np.all(np.isfinite(np.asarray(gz)))
print("ok 1 zero vector stays zero with finite gradient")

# 2. default and a far bound --------------------------------------------------------------------------------------------------
for res in ("identity", "learned"):
    kw = {} if res == "learned" else {"depth_hc_res": res}
    m0, m9 = model(**kw), model(depth_branch_bound=1e6, **kw)
    assert [a.shape for a in jax.tree_util.tree_leaves(eqx.filter(m0, eqx.is_array))] == [a.shape for a in jax.tree_util.tree_leaves(eqx.filter(m9, eqx.is_array))]
    assert np.array_equal(outs(m0), outs(m9)), res
    assert np.array_equal(outs(m0, jnp.bfloat16), outs(m9, jnp.bfloat16)), res
    l0, g0 = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(m0))
    l9, g9 = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(m9))
    assert float(l0) == float(l9)
    for a, b_ in zip(jax.tree_util.tree_leaves(eqx.filter(g0, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(g9, eqx.is_array))):
        assert np.array_equal(np.asarray(a.astype(jnp.float32) if hasattr(a, "astype") else a), np.asarray(b_.astype(jnp.float32) if hasattr(b_, "astype") else b_))
    print(f"ok 2 mixing {res}: no new leaves; a far bound gives the same outputs and gradients as no bound, float32 and bfloat16")

# 3. a small bound, by hand ---------------------------------------------------------------------------------------------------
bound = 1.0
mb = model(depth_hc_res="identity", depth_branch_bound=bound)
got = outs(mb)
tr = mb.transformer
f32 = lambda a: a.astype(jnp.float32)  # noqa: E731
H = hax.stack(Stream, [x] * M)
maxn, nclip = 0.0, 0
for li in range(L):
    lay = tr.layer(li)
    for j in range(2):
        q = lay.hyper[j]
        c_read, _, c_write = _hc_constants(M, 2 * li + j)
        Hn = f32(_rms(H, Embed))
        sc = f32(q.scale)
        read = c_read + f32(q.read) + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(q.w_read), axis=Embed))
        write = c_write + f32(q.write) + sc["hp_scale", 1] * hax.tanh(hax.dot(Hn, f32(q.w_write), axis=Embed).rename({"stream": "stream_out"}))
        xr = hax.dot(read.astype(H.dtype), H, axis="stream")
        raw = lay.branch(j, xr, mask, key=None, pos_ids=None)
        n = np.linalg.norm(np.asarray(raw.rearrange((Batch, Pos, Embed)).array, np.float64), axis=-1)
        nclip += int((n > bound).sum())
        fb = _bound_norm(raw, Embed, bound)
        maxn = max(maxn, float(np.linalg.norm(np.asarray(fb.rearrange((Batch, Pos, Embed)).array, np.float64), axis=-1).max()))
        H = H + (write.astype(H.dtype) * fb.broadcast_axis(write.resolve_axis("stream_out"))).rename({"stream_out": "stream"})
    want = np.asarray(hax.sum(H, axis="stream").rearrange((Batch, Pos, Embed)).array)
    assert np.allclose(got[li], want, rtol=1e-4, atol=1e-4), (li, np.abs(got[li] - want).max())
assert nclip > 0 and maxn <= bound * (1 + 1e-5), (nclip, maxn)
print(f"ok 3 small bound {bound}: the model equals the hand computation at every layer; {nclip} token-sublayer pairs were scaled, the largest bounded |F| is {maxn:.6f}")

# 4. the derivative above the bound --------------------------------------------------------------------------------------------
v0 = jnp.asarray(rng.standard_normal((1, 1, D)), jnp.float32)
v0 = v0 / jnp.linalg.norm(v0) * 5.0
Bb, Pp = Axis("batch", 1), Axis("position", 1)
fn = lambda a: _bound_norm(hax.named(a, (Bb, Pp, Embed)), Embed, 1.0).array  # noqa: E731
d = jnp.asarray(rng.standard_normal((1, 1, D)), jnp.float32)
_, tang = jax.jvp(fn, (v0,), (d,))
eps = 1e-3
num = (fn(v0 + eps * d) - fn(v0 - eps * d)) / (2 * eps)
assert np.allclose(np.asarray(tang), np.asarray(num), rtol=2e-2, atol=2e-3), (np.abs(np.asarray(tang) - np.asarray(num)).max())
_, rad = jax.jvp(fn, (v0,), (v0,))
assert np.abs(np.asarray(rad)).max() < 1e-4, np.abs(np.asarray(rad)).max()
print(f"ok 4 above the bound: jvp agrees with a central difference; moving along F itself changes nothing (max {np.abs(np.asarray(rad)).max():.2e})")

# 5. training steps ------------------------------------------------------------------------------------------------------------
for name, detach in (("separate heads", False), ("probes", True)):
    m = model(depth_hc_res="identity", depth_branch_bound=1.0, pls_detach_backbone=detach)
    l_, g_ = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(m))
    assert np.isfinite(float(l_))
    for a in jax.tree_util.tree_leaves(eqx.filter(g_, eqx.is_array)):
        assert np.all(np.isfinite(np.asarray(a.astype(jnp.float32)))), name
    print(f"ok 5 {name} step in bfloat16 with bound 1.0: loss {float(l_):.4f}, every gradient finite")
try:
    from experiments.references.depth_arch_qwen3 import DepthArchMuonHConfig
except ImportError:
    from experiments.references.della_muonh_qwen3_scaling import DepthArchMuonHConfig
opt = DepthArchMuonHConfig(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0, weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0, coefficient_type="simple").build(4959)
m = model(depth_hc_res="identity", depth_branch_bound=20.0)
params = eqx.filter(m, eqx.is_inexact_array)
state = opt.init(params)


@eqx.filter_jit
def train_step(m_, st):
    l_, g_ = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(m_))
    g_ = mp.cast_to_param(g_)
    up, st = opt.update(g_, st, params=eqx.filter(m_, eqx.is_inexact_array))
    return eqx.apply_updates(m_, up), st, l_


hist = []
for s in range(12):
    m, state, l_ = train_step(m, state)
    hist.append(float(l_))
assert np.all(np.isfinite(hist)) and hist[-1] < hist[0], hist
print(f"ok 5 real optimizer, bfloat16 compute, 12 steps on one batch: loss {hist[0]:.3f} -> {hist[-1]:.3f}")

# 6. validation ----------------------------------------------------------------------------------------------------------------
for arch, kw in (("preln", dict(depth_branch_bound=100.0)), ("mhc", dict(depth_branch_bound=100.0)), ("hc", dict(depth_branch_bound=-1.0))):
    try:
        model(arch, **kw)
        raise AssertionError(f"{arch} {kw} was accepted")
    except ValueError as e:
        print(f"ok 6 rejected: {str(e)[:110]}")
print("ALL OK")
