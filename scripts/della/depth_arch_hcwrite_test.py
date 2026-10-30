"""CPU tests for the two causal-test designs of 2026-10-30 (owner popup): depth_hc_write="shared" (hc whose write is one
weight shared by every stream) and mhc with depth_mhc_res="identity" plus depth_mhc_rw="linear". Run on a vis node from
this worktree:  nice -n 19 $PY -u scripts/della/depth_arch_hcwrite_test.py
1. At init hc with shared writes computes hc's layer outputs; with random write offsets the two differ.
2. With shared writes, identity mixing (res and w_res zero) and random read / write, the streams stay equal through every
   sublayer (the embedding enters every stream alike), so no sublayer writes into the stream differences; hc's own
   per-stream writes make them unequal.
3. mhc with identity mixing and linear read/write equals mhc with identity mixing at init up to mhc's read leak
   sigmoid(-8) (relative error <= 2e-3).
4. A separate-heads and a probes training step of both designs are finite, and hc's write parameters and mhc's phi get
   non-zero gradients; depth_hc_write="shared" with another design is rejected.
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


def stream_spread(m_):
    """max over sublayers of |H_c - mean_c H| / |H| after each layer, run layer by layer through hyper_step."""
    tr = m_.transformer
    m = tr.config.hc_streams
    H = hax.stack(Axis("stream", m), [x] * m)
    worst = 0.0
    for i in range(tr.config.num_layers):
        ln = tr.config.Layers.name
        layer = jax.tree.map(lambda a: a[ln, i] if isinstance(a, hax.NamedArray) and ln in a.axis_names else a, tr.layers.stacked, is_leaf=lambda a: isinstance(a, hax.NamedArray))
        H = layer.hyper_step(H, mask, key=None, pos_ids=None, layer=i)
        a = np.asarray(H.rearrange(("stream", ...)).array)
        worst = max(worst, float(np.linalg.norm(a - a.mean(0, keepdims=True)) / np.linalg.norm(a)))
    return worst


# 1. hc shared writes: init and random writes ---------------------------------------------------------------------------
m_hc, m_sh = model("hc"), model("hc", depth_hc_write="shared")
r0 = rel(outs(m_sh), outs(m_hc))
assert r0 <= 1e-6, f"init: relative error {r0:.2e}"
print(f"ok 1 at init shared writes equal hc (rel {r0:.1e})")
hy = list(m_hc.transformer.layers.stacked.hyper)
rnd = tuple(dataclasses.replace(p, read=rand_like(p.read, 2.0, 10 + j), write=rand_like(p.write, 2.0, 20 + j), w_read=rand_like(p.w_read, 0.5, 30 + j),
                                w_write=rand_like(p.w_write, 0.5, 40 + j), scale=rand_like(p.scale, 1.0, 50 + j)) for j, p in enumerate(hy))
m_hc2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_hc, rnd)
m_sh2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_sh, rnd)
d = rel(outs(m_sh2), outs(m_hc2))
assert d > 1e-2, f"random writes: shared should differ from hc, rel {d:.2e}"
print(f"ok 1 with random writes shared differs from hc (rel {d:.2e})")

# 2. no write into the stream differences (identity mixing: rnd keeps res = 0 and w_res = 0) ---------------------------
s_sh, s_hc = stream_spread(m_sh2), stream_spread(m_hc2)
assert s_sh <= 1e-5 and s_hc > 1e-2, (s_sh, s_hc)
print(f"ok 2 shared writes keep the streams equal (spread {s_sh:.1e}); hc's own writes do not ({s_hc:.2e})")

# 3. mhc identity mixing + linear read/write at init ----------------------------------------------------------------
m_id, m_idl = model("mhc", depth_mhc_res="identity"), model("mhc", depth_mhc_res="identity", depth_mhc_rw="linear")
r3 = rel(outs(m_idl), outs(m_id))
assert r3 <= 2e-3, f"mhc identity+linear at init: relative error {r3:.2e}"
print(f"ok 3 mhc identity + linear equals mhc identity at init up to the read leak (rel {r3:.1e})")

# 4. training steps and validation ----------------------------------------------------------------------------------
for detach in (False, True):
    for nm, kw in (("hc shared", dict(depth_hc_write="shared")), ("mhc id+lin", dict(depth_mhc_res="identity", depth_mhc_rw="linear"))):
        mm = model("hc" if nm.startswith("hc") else "mhc", pls_detach_backbone=detach, **kw)
        loss, g = eqx.filter_value_and_grad(lf)(mm)
        assert np.isfinite(float(loss)), loss
        gh = g.transformer.layers.stacked.hyper
        if nm.startswith("hc"):
            gw = sum(float(jnp.sum(jnp.abs(p.write.array))) + float(jnp.sum(jnp.abs(p.w_write.array))) for p in gh)
        else:
            gw = sum(float(jnp.sum(jnp.abs(p.phi.array))) for p in gh)
        assert gw > 0, (nm, gw)
        print(f"ok 4 {nm} {'probes' if detach else 'separate heads'} step: loss {float(loss):.4f}, |grad write/phi| {gw:.2e}")
try:
    model("mhc", depth_hc_write="shared")
    raise AssertionError("depth_hc_write='shared' with mhc was accepted")
except ValueError as e:
    print(f"ok 4 rejected: {e}")
print("ALL OK")
