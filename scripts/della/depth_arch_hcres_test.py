"""CPU tests for depth_hc_res="identity" (owner popup 2026-11-01): hc with the residual mixing fixed to the identity,
H <- H + w (x) F(RN(x)), read and write learned as in hc. Run on a vis node from this worktree:
    CUDA_VISIBLE_DEVICES= nice -n 19 $PY -u scripts/della/depth_arch_hcres_test.py
1. At init hc's mixing is I (static I, offsets 0, w_res 0), so identity mixing computes hc's layer outputs.
2. With random read, write, res offsets and w_res, identity mixing equals hc whose res offsets and w_res are zero (its
   mixing is then I for every token) and differs from hc with the random mixing.
3. A separate-heads and a probes training step are finite; under identity mixing the res offsets and w_res get exactly
   zero gradient and the read and write parameters get non-zero gradient.
4. depth_hc_res="identity" with another design, and an unknown value, are rejected.
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


def outs(m):
    tr = m.transformer
    return np.asarray(tr.outputs(x, mask).rearrange((tr.config.Layers, Batch, Pos, tr.config.Embed)).array)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def rand_like(a, scale, k):
    return hax.named(jnp.asarray(np.random.default_rng(k).standard_normal(a.array.shape) * scale, a.array.dtype), a.axes)


def zeros_like(a):
    return hax.named(jnp.zeros_like(a.array), a.axes)


def lf(m_):
    out = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
    loss = out[0] if isinstance(out, tuple) else out
    return loss.array if isinstance(loss, hax.NamedArray) else loss


def with_hyper(m_, hy):
    return eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_, tuple(hy))


# 1. at init ------------------------------------------------------------------------------------------------------------
m_hc, m_id = model("hc"), model("hc", depth_hc_res="identity")
r1 = rel(outs(m_id), outs(m_hc))
assert r1 <= 1e-6, f"init: relative error {r1:.2e}"
print(f"ok 1 at init identity mixing equals hc (rel {r1:.1e})")

# 2. random routing -----------------------------------------------------------------------------------------------------
hy = list(m_hc.transformer.layers.stacked.hyper)
rnd = [dataclasses.replace(p, read=rand_like(p.read, 0.5, 10 + j), write=rand_like(p.write, 0.5, 20 + j), res=rand_like(p.res, 0.5, 60 + j),
                           w_read=rand_like(p.w_read, 0.5, 30 + j), w_write=rand_like(p.w_write, 0.5, 40 + j), w_res=rand_like(p.w_res, 0.5, 70 + j),
                           scale=rand_like(p.scale, 1.0, 50 + j)) for j, p in enumerate(hy)]
no_mix = [dataclasses.replace(p, res=zeros_like(p.res), w_res=zeros_like(p.w_res)) for p in rnd]
o_id = outs(with_hyper(m_id, rnd))
r2 = rel(o_id, outs(with_hyper(m_hc, no_mix)))
d2 = rel(o_id, outs(with_hyper(m_hc, rnd)))
assert r2 <= 1e-6, f"identity mixing vs hc with zero mixing offsets: rel {r2:.2e}"
assert d2 > 1e-2, f"identity mixing should differ from hc with random mixing: rel {d2:.2e}"
print(f"ok 2 random routing: identity mixing equals hc with zero mixing offsets (rel {r2:.1e}), differs from hc ({d2:.2e})")

# 3. training steps -----------------------------------------------------------------------------------------------------
for detach in (False, True):
    mm = with_hyper(model("hc", depth_hc_res="identity", pls_detach_backbone=detach), rnd)
    loss, g = eqx.filter_value_and_grad(lf)(mm)
    assert np.isfinite(float(loss)), loss
    gh = g.transformer.layers.stacked.hyper
    g_mix = sum(float(jnp.sum(jnp.abs(p.res.array))) + float(jnp.sum(jnp.abs(p.w_res.array))) for p in gh)
    g_rw = sum(float(jnp.sum(jnp.abs(getattr(p, n).array))) for p in gh for n in ("read", "write", "w_read", "w_write"))
    assert g_mix == 0.0 and g_rw > 0, (g_mix, g_rw)
    print(f"ok 3 {'probes' if detach else 'separate heads'} step: loss {float(loss):.4f}, |grad res, w_res| {g_mix:.1e}, |grad read/write| {g_rw:.2e}")

# 4. validation ---------------------------------------------------------------------------------------------------------
for arch, kw in (("mhc", dict(depth_hc_res="identity")), ("hc", dict(depth_hc_res="bogus"))):
    try:
        model(arch, **kw)
        raise AssertionError(f"{arch} {kw} was accepted")
    except ValueError as e:
        print(f"ok 4 rejected: {e}")
print("ALL OK")
