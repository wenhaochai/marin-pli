"""CPU tests for depth_mhc_rw="linear" (mhc with hc's signed, unbounded read and write maps; owner popup 2026-10-27). Run on
a vis node from this worktree:  nice -n 19 $PY -u scripts/della/depth_arch_mhcrw_test.py
1. At init the variant computes mhc's layer outputs up to mhc's read leak sigmoid(-8) = 3.4e-4 (relative error <= 2e-3);
   with random biases, gains and projections the two differ, and the variant's read and write take negative values.
2. A separate-heads and a probes training step of the variant are finite and its routing parameters get non-zero
   gradients; depth_mhc_rw="linear" with another design is rejected.
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


def lf(m_):
    out = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
    loss = out[0] if isinstance(out, tuple) else out
    return loss.array if isinstance(loss, hax.NamedArray) else loss


# 1. init and random routing ----------------------------------------------------------------------------------------
from experiments.references.depth_arch_qwen3 import _mhc_constants  # noqa: E402
m_mh, m_ln = model("mhc"), model("mhc", depth_mhc_rw="linear")
r0 = rel(outs(m_ln), outs(m_mh))
assert r0 <= 2e-3, f"init: relative error {r0:.2e}"
print(f"ok 1 at init the variant equals mhc up to the read leak (rel {r0:.1e})")
hy = list(m_mh.transformer.layers.stacked.hyper)
rnd = tuple(dataclasses.replace(p, phi=rand_like(p.phi, 0.3, 10 + j), bias=rand_like(p.bias, 2.0, 20 + j), gain=rand_like(p.gain, 1.0, 30 + j)) for j, p in enumerate(hy))
m_mh2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_mh, rnd)
m_ln2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_ln, rnd)
d = rel(outs(m_ln2), outs(m_mh2))
assert d > 1e-2, f"random routing: the variant should differ from mhc, rel {d:.2e}"
print(f"ok 1 with random routing the variant differs from mhc (rel {d:.2e})")
# the read and write the variant computes for layer 0's attention, from the embedding copied to every stream
p0 = jax.tree_util.tree_map(lambda a: a["layer", 0] if isinstance(a, hax.NamedArray) and "layer" in a.axis_names else a, rnd[0], is_leaf=lambda a: isinstance(a, hax.NamedArray))
m = 4
H = hax.stack(Axis("stream", m), [x] * m)
S, E = H.resolve_axis("stream"), x.resolve_axis("embed")
inv = hax.rsqrt(hax.mean(H * H, axis=(S, E)) + 1e-6)
mix = hax.dot(H, p0.phi, axis=(S, E)) * inv
c = _mhc_constants(m, 0) + p0.bias
pre, post = mix["mhc", 0:m] * p0.gain["hp_scale", 0] + c["mhc", 0:m], mix["mhc", m:2 * m] * p0.gain["hp_scale", 1] + c["mhc", m:2 * m]
k_read = _mhc_constants(m, 0)["mhc", 0:m]
read, write = (k_read > 0).astype(jnp.float32) + pre - k_read, 1.0 + post
rmin, wmin = float(jnp.min(read.array)), float(jnp.min(write.array))
assert rmin < 0 and wmin < 0, (rmin, wmin)
print(f"ok 1 the variant's read and write go negative (min read {rmin:.2f}, min write {wmin:.2f})")

# 2. training steps and validation ----------------------------------------------------------------------------------
for detach in (False, True):
    mm = model("mhc", depth_mhc_rw="linear", pls_detach_backbone=detach)
    loss, g = eqx.filter_value_and_grad(lf)(mm)
    assert np.isfinite(float(loss)), loss
    gh = g.transformer.layers.stacked.hyper
    gp = sum(float(jnp.sum(jnp.abs(p.phi.array))) for p in gh)
    gb = sum(float(jnp.sum(jnp.abs(p.bias.array))) + float(jnp.sum(jnp.abs(p.gain.array))) for p in gh)
    assert gp > 0 and gb > 0, (gp, gb)
    print(f"ok 2 {'probes' if detach else 'separate heads'} step: loss {float(loss):.4f}, |grad phi| {gp:.2e}, |grad bias, gain| {gb:.2e}")
for arch in ("hc", "preln"):
    try:
        model(arch, depth_mhc_rw="linear")
        raise AssertionError(f"depth_mhc_rw='linear' with {arch} was accepted")
    except ValueError as e:
        print(f"ok 2 rejected: {e}")
print("ALL OK")
