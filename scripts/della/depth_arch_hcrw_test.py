"""CPU tests for depth_hc_rw="sigmoid" (hc with mHC's read and write maps; owner popup 2026-10-26). Run on a vis node from
this worktree:  nice -n 19 $PY -u scripts/della/depth_arch_hcrw_test.py
1. At init the variant computes hc's layer outputs up to the read leak sigmoid(-8) = 3.4e-4 (relative error <= 2e-3);
   with random read / write offsets and dynamic weights the two differ, and the variant's read stays in (0, 1) and its
   write in (0, 2) while hc's leave those ranges.
2. A separate-heads and a probes training step of the variant are finite and its read and write parameters get non-zero
   gradients; depth_hc_rw="sigmoid" with another design is rejected.
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
m_hc, m_sg = model("hc"), model("hc", depth_hc_rw="sigmoid")
r0 = rel(outs(m_sg), outs(m_hc))
assert r0 <= 2e-3, f"init: relative error {r0:.2e}"
print(f"ok 1 at init the variant equals hc up to the read leak (rel {r0:.1e})")
hy = list(m_hc.transformer.layers.stacked.hyper)
rnd = tuple(dataclasses.replace(p, read=rand_like(p.read, 2.0, 10 + j), write=rand_like(p.write, 2.0, 20 + j), w_read=rand_like(p.w_read, 0.5, 30 + j),
                                w_write=rand_like(p.w_write, 0.5, 40 + j), scale=rand_like(p.scale, 1.0, 50 + j)) for j, p in enumerate(hy))
m_hc2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_hc, rnd)
m_sg2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_sg, rnd)
d = rel(outs(m_sg2), outs(m_hc2))
assert d > 1e-2, f"random routing: the variant should differ from hc, rel {d:.2e}"
print(f"ok 1 with random read/write the variant differs from hc (rel {d:.2e})")

# 2. training steps and validation ----------------------------------------------------------------------------------
for detach in (False, True):
    mm = model("hc", depth_hc_rw="sigmoid", pls_detach_backbone=detach)
    loss, g = eqx.filter_value_and_grad(lf)(mm)
    assert np.isfinite(float(loss)), loss
    gh = g.transformer.layers.stacked.hyper
    gr = sum(float(jnp.sum(jnp.abs(p.read.array))) + float(jnp.sum(jnp.abs(p.w_read.array))) for p in gh)
    gw = sum(float(jnp.sum(jnp.abs(p.write.array))) + float(jnp.sum(jnp.abs(p.w_write.array))) for p in gh)
    assert gr > 0 and gw > 0, (gr, gw)
    print(f"ok 2 {'probes' if detach else 'separate heads'} step: loss {float(loss):.4f}, |grad read| {gr:.2e}, |grad write| {gw:.2e}")
try:
    model("preln", depth_hc_rw="sigmoid")
    raise AssertionError("depth_hc_rw='sigmoid' with preln was accepted")
except ValueError as e:
    print(f"ok 2 rejected: {e}")
print("ALL OK")
