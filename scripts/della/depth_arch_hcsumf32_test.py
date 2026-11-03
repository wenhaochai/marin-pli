"""CPU tests for depth_hc_state_f32 with depth_hc_res="sum" (owner popup 2026-11-03): the streams in float32, everything else in the compute dtype.
Parameters and inputs are cast to bfloat16 as in training. Run on a vis node from this worktree:
    CUDA_VISIBLE_DEVICES= nice -n 19 $PY -u scripts/della/depth_arch_hcsumf32_test.py
1. No new parameters. At init the layer outputs equal hc's up to rounding, and they come out in bfloat16.
2. The mixing alone, writes off, in bfloat16 compute: the stream state stays float32 and keeps the stream sum to float32 rounding, against about
   1e-2 with bfloat16 streams. With differences 1e5 times the sum, the float32 streams still return the sum; the bfloat16 streams do not.
3. The branch runs in bfloat16: one sublayer's write has the compute dtype.
4. A separate-heads and a probes training step are finite; res and w_res get non-zero gradient.
5. Wrong combinations are rejected.
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

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
rng = np.random.default_rng(0)
B, T, V, L, D, M = 4, 16, 300, 4, 48, 4
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
Stream, Embed = Axis("stream", M), Axis("embed", D)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True)
key0 = jrandom.PRNGKey(0)
mask = AttentionMask.causal()
x16 = hax.named(jnp.asarray(rng.standard_normal((B, T, D)), jnp.bfloat16), (Batch, Pos, Embed))
tok = rng.integers(2, V, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=mask)
mp = jmp.get_policy("p=f32,c=bfloat16")


def model(arch, **kw):
    cfg = PerLayerQwen3Config(**{**common, **kw}, hybrid_norm=True, depth_arch=arch, scan_layers=False)
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=key0)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def rand_like(a, scale, k):
    return hax.named(jnp.asarray(np.random.default_rng(k).standard_normal(a.array.shape) * scale, a.array.dtype), a.axes)


def with_hyper(m_, hy):
    return eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_, tuple(hy))


def lf(m_):
    out = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
    loss = out[0] if isinstance(out, tuple) else out
    return loss.array if isinstance(loss, hax.NamedArray) else loss


def arr(H):
    return np.asarray(H.array.astype(jnp.float32), np.float64)


m_hc, m_f, m_b = model("hc"), model("hc", depth_hc_res="sum", depth_hc_state_f32=True), model("hc", depth_hc_res="sum")

# 1. no new parameters; init ---------------------------------------------------------------------------------------------------
n = len(jax.tree_util.tree_leaves(m_hc))
assert len(jax.tree_util.tree_leaves(m_f)) == n
o_f = mp.cast_to_compute(m_f).transformer.outputs(x16, mask)
o_h = mp.cast_to_compute(m_hc).transformer.outputs(x16, mask)
assert o_f.array.dtype == jnp.bfloat16, o_f.array.dtype
r = rel(arr(o_f.rearrange(o_h.axes)), arr(o_h))
assert r <= 2e-2, r
print(f"ok 1 same {n} leaves as hc; at init the layer outputs equal hc's up to bfloat16 rounding (rel {r:.1e}) and come out in bfloat16")

# 2. the mixing alone, writes off ----------------------------------------------------------------------------------------------
hy = list(m_hc.transformer.layers.stacked.hyper)
nw = [dataclasses.replace(p, read=rand_like(p.read, 0.3, 100 + j), res=rand_like(p.res, 0.3, 110 + j), w_read=rand_like(p.w_read, 0.3, 120 + j), w_res=rand_like(p.w_res, 0.3, 130 + j), scale=rand_like(p.scale, 1.0, 140 + j),
                              write=hax.named(jnp.full(p.write.array.shape, -1.0, p.write.array.dtype), p.write.axes), w_write=hax.named(jnp.zeros_like(p.w_write.array), p.w_write.axes)) for j, p in enumerate(hy)]
H_np = rng.standard_normal((M, B, T, D))


def stepped(m_, H0):
    return mp.cast_to_compute(with_hyper(m_, nw)).transformer.layer(0).hyper_step(H0, mask, key=None, pos_ids=None, layer=0)


H32 = hax.named(jnp.asarray(H_np, jnp.float32), (Stream, Batch, Pos, Embed))
H16 = hax.named(jnp.asarray(H_np, jnp.bfloat16), (Stream, Batch, Pos, Embed))
out32, out16 = stepped(m_f, H32), stepped(m_b, H16)
assert out32.array.dtype == jnp.float32 and out16.array.dtype == jnp.bfloat16, (out32.array.dtype, out16.array.dtype)
s_in32, s_in16 = arr(H32).sum(0), arr(H16).sum(0)
r32, r16 = rel(arr(out32.rearrange((Stream, Batch, Pos, Embed))).sum(0), s_in32), rel(arr(out16.rearrange((Stream, Batch, Pos, Embed))).sum(0), s_in16)
assert r32 <= 2e-5 and r16 > 1e-3, (r32, r16)
print(f"ok 2 writes off, bfloat16 compute, one layer: the stream sum changes by rel {r32:.1e} with float32 streams, {r16:.1e} with bfloat16 streams")
# differences 1e5 times the sum: storing the streams in float32 returns the sum; storing them in bfloat16 loses it
base = rng.standard_normal((1, B, T, D))
dif = rng.standard_normal((M, B, T, D)); dif = dif - dif.mean(0, keepdims=True)
big = base / M + 1e5 * dif * np.linalg.norm(base) / np.linalg.norm(dif)
true_sum = big.sum(0)
s32 = np.asarray(jnp.asarray(big, jnp.float32), np.float64).sum(0)
s16 = np.asarray(jnp.asarray(big, jnp.bfloat16).astype(jnp.float32), np.float64).sum(0)
e32, e16 = rel(s32, true_sum), rel(s16, true_sum)
assert e32 < 0.1 and e16 > 10, (e32, e16)
print(f"ok 2 differences 1e5 times the sum: relative error of the stream sum {e32:.1e} in float32, {e16:.1e} in bfloat16")

# 3. the branch runs in the compute dtype --------------------------------------------------------------------------------------
lay = mp.cast_to_compute(m_f).transformer.layer(0)
h = hax.named(jnp.asarray(rng.standard_normal((B, T, D)), jnp.bfloat16), (Batch, Pos, Embed))
f = lay.branch(0, h, mask, key=None, pos_ids=None)
assert f.array.dtype == jnp.bfloat16, f.array.dtype
print("ok 3 the branch output on a compute-dtype read has the compute dtype")

# 4. training steps ------------------------------------------------------------------------------------------------------------
rnd = [dataclasses.replace(p, read=rand_like(p.read, 0.5, 10 + j), write=rand_like(p.write, 0.5, 20 + j), res=rand_like(p.res, 0.3, 60 + j), w_read=rand_like(p.w_read, 0.5, 30 + j), w_write=rand_like(p.w_write, 0.5, 40 + j),
                           w_res=rand_like(p.w_res, 0.5, 70 + j), scale=rand_like(p.scale, 1.0, 50 + j)) for j, p in enumerate(hy)]
for detach in (False, True):
    name = "probes" if detach else "separate heads"
    mm = with_hyper(model("hc", depth_hc_res="sum", depth_hc_state_f32=True, pls_detach_backbone=detach), rnd)
    loss, gr = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(mm))
    assert np.isfinite(float(loss)), loss
    gh = gr.transformer.layers.stacked.hyper
    g_res = sum(float(jnp.sum(jnp.abs(p.res.array.astype(jnp.float32)))) for p in gh)
    g_w = sum(float(jnp.sum(jnp.abs(p.w_res.array.astype(jnp.float32)))) for p in gh)
    assert g_res > 0 and g_w > 0, (g_res, g_w)
    print(f"ok 4 {name} step: loss {float(loss):.4f}, |grad res| {g_res:.2e}, |grad w_res| {g_w:.2e}")

# 5. validation ----------------------------------------------------------------------------------------------------------------
for arch, kw in (("hc", dict(depth_hc_state_f32=True)), ("hc", dict(depth_hc_res="move", depth_hc_state_f32=True)), ("preln", dict(depth_hc_state_f32=True))):
    try:
        model(arch, **kw)
        raise AssertionError(f"{arch} {kw} was accepted")
    except ValueError as e:
        print(f"ok 5 rejected: {str(e)[:110]}")
print("ALL OK")
