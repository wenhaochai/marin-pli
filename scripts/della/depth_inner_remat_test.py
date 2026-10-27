"""CPU test of DEPTH_INNER_REMAT=save (depth_arch_qwen3._two_level_scan: the inner scan saves each layer's internals while
the backward pass recomputes a group, in place of a second checkpoint per layer). For attnres (dynamic, vnorm, static
routing) and moda at 8 layers (2 groups of 4) with separate heads: the loss and every gradient leaf with the knob equal the
default's (rel <= 1e-6), and the outputs equal the Python-loop reference. Run on a vis node from this worktree:
    JAX_PLATFORMS=cpu nice -n 19 $PY -u scripts/della/depth_inner_remat_test.py
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

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
B, T, V, L, D = 2, 16, 300, 8, 48
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2,
              attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True)
mask = AttentionMask.causal()
tok = rng.integers(2, V, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)),
               loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=mask)
x = hax.named(jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32), (Batch, Pos, Axis("embed", D)))


def model(arch, **kw):
    cfg = PerLayerQwen3Config(**{**common, **kw}, hybrid_norm=True, depth_arch=arch, scan_layers=False)
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))


def lf(m_):
    out = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
    loss = out[0] if isinstance(out, tuple) else out
    return loss.array if isinstance(loss, hax.NamedArray) else loss


def run(m_, knob):
    if knob:
        os.environ["DEPTH_INNER_REMAT"] = "save"
    else:
        os.environ.pop("DEPTH_INNER_REMAT", None)
    jax.clear_caches()
    loss, g = eqx.filter_value_and_grad(lf)(m_)
    return float(loss), [np.asarray(a) for a in jax.tree.leaves(eqx.filter(g, eqx.is_array))]


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


ok = True
for name, arch, kw in (("attnres", "attnres", {}), ("attnres vnorm", "attnres", {"depth_attnres_vnorm": True}),
                       ("attnres static", "attnres", {"depth_route": "static"}), ("moda", "moda", {})):
    m_ = model(arch, **kw)
    tr = m_.transformer
    scan_out = np.asarray(tr.outputs(x, mask).rearrange((tr.config.Layers, Batch, Pos, tr.config.Embed)).array)
    loop_out = np.stack([np.asarray(o.rearrange((Batch, Pos, tr.config.Embed)).array) for o in tr._loop_outputs(x, mask)])
    l0, g0 = run(m_, False)
    l1, g1 = run(m_, True)
    worst = max(rel(a, b) for a, b in zip(g1, g0))
    bitwise = all(np.array_equal(a, b) for a, b in zip(g1, g0))
    r_out = rel(scan_out, loop_out)
    good = abs(l1 - l0) <= 1e-6 * abs(l0) and worst <= 1e-6 and r_out <= 2e-5 and len(g0) == len(g1)
    ok &= good
    print(f"{'ok' if good else 'FAIL'} {name}: loss {l0:.6f} vs {l1:.6f}; {len(g0)} gradient leaves, worst rel {worst:.1e}, bitwise {bitwise}; scan vs loop outputs rel {r_out:.1e}", flush=True)
os.environ.pop("DEPTH_INNER_REMAT", None)
print("ALL OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
