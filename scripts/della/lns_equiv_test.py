"""CPU test of the exact LNS equivalence (depth_arch_qwen3 docstring: depth_gain_init="lns_bf16" + LnsStepMuonHConfig;
Lean ArchLean.PracTrain). A small model (4 layers, width 48, probes: separate heads, backbone detached, as the 48-layer
runs) trains on one fixed random batch for N steps in float32 (compute and params; the factor dtype is then float32),
with the runs' optimizer constants and a small clip threshold so every group clips. Runs:
  lns        depth_arch lns, MuonHConfig (the reference)
  equiv      preln, scales start at lns's factor, LnsStepMuonHConfig      -> must follow lns to rounding
  start      preln, scales start at lns's factor, MuonHConfig (wq w15's setting)
  step       preln, scales start at 1, LnsStepMuonHConfig
  noclip     preln, start as lns, update times c_l but the clip sees Pre-LN's gradients (the clip term alone)
Per step: |loss - lns loss|, and after N steps max relative difference of every parameter from lns's (pre-norm scales
compared as g' versus c_l g). Pass: equiv within 1e-4 relative on every parameter and 1e-5 on the loss; the other runs
differ by much more (they show the test can fail). With LNS_EQUIV_X64=1 everything runs in float64 (the factor stays
the float32 value both paths use), so rounding drops to ~1e-12 and the pass needs equiv within 1e-9 on both and every
other run at least 1e4 times further from lns than equiv.
    source scripts/della/pls_env.sh; JAX_PLATFORMS=cpu LNS_EQUIV_X64=1 nice -n 19 $PY scripts/della/lns_equiv_test.py
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402

from experiments.references.depth_arch_qwen3 import LnsStepMuonHConfig, _is_prenorm_scale, lns_factor, scale_prenorm_scales  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
X64 = os.environ.get("LNS_EQUIV_X64") == "1"
if X64:
    jax.config.update("jax_enable_x64", True)
N, B, T, V, L, D = 40, 4, 16, 300, 4, 48
RUN = dict(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5,
           muon_epsilon=1e-5, max_grad_norm=0.05, weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0,
           coefficient_type="simple")
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
rng = np.random.default_rng(0)
tok = rng.integers(2, V, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=AttentionMask.causal())


def model(arch, gain_init="one"):
    cfg = PerLayerQwen3Config(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA,
                              pls_weight=1.0, pls_separate_heads=True, pls_detach_backbone=True, hybrid_norm=True, depth_arch=arch, depth_gain_init=gain_init, scan_layers=False)
    m = PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
    if X64:  # some leaves are created as float32 explicitly; store every parameter in float64
        m = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating) else x, m)
    return m


def lf(m):
    out = m.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
    loss = out[0] if isinstance(out, tuple) else out
    return loss.array if isinstance(loss, hax.NamedArray) else loss


def train(m, tx):
    p = eqx.filter(m, eqx.is_array)
    st = tx.init(p)
    vg = eqx.filter_jit(eqx.filter_value_and_grad(lf))
    upd = eqx.filter_jit(lambda g, st, p: tx.update(g, st, p))
    losses = []
    for _ in range(N):
        loss, g = vg(m)
        losses.append(float(loss))
        u, st = upd(eqx.filter(g, eqx.is_array), st, eqx.filter(m, eqx.is_array))
        m = eqx.apply_updates(m, u)
    return m, np.array(losses)


def converted(m):
    """lns's parameters written as Pre-LN's: the pre-norm scales of layer l times c_l (float32)."""
    return jax.tree_util.tree_map_with_path(lambda kp, x: x * lns_factor(x.shape[0], jnp.float32).reshape((-1,) + (1,) * (x.ndim - 1)) if eqx.is_array(x) and _is_prenorm_scale(kp) else x, m)


def maxrel(a, b):
    out = {}
    for (kp, x), y in zip(jax.tree_util.tree_leaves_with_path(eqx.filter(a, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_array))):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        out[jax.tree_util.keystr(kp)] = float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-30))
    return out


hits = [jax.tree_util.keystr(kp) for kp, x in jax.tree_util.tree_leaves_with_path(eqx.filter(model("preln"), eqx.is_array)) if _is_prenorm_scale(kp)]
print("pre-norm scales the optimizer rescales:", hits)
assert len(hits) == 2, hits
opt, opt_l = MuonHConfig(**RUN), LnsStepMuonHConfig(**RUN, lns_step_dtype="float64" if X64 else "float32")


def noclip_tx():   # lns's step size, but the clip sees Pre-LN's gradients: update times c_l only
    full = opt.build(1000)
    return optax.chain(full, scale_prenorm_scales(jnp.float32))


m_ref0 = model("lns")
print("parameter dtypes:", sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(eqx.filter(m_ref0, eqx.is_array))}))
ref_m, ref_l = train(m_ref0, opt.build(1000))
ref = converted(ref_m)
runs = {"equiv": (model("preln", "lns"), opt_l.build(1000)), "start": (model("preln", "lns"), opt.build(1000)),
        "step": (model("preln"), opt_l.build(1000)), "noclip": (model("preln", "lns"), noclip_tx())}
res = {}
for name, (m0, tx) in runs.items():
    m1, l1 = train(m0, tx)
    r = maxrel(m1, ref)
    worst = max(r, key=r.get)
    res[name] = (float(np.abs(l1 - ref_l).max()), r[worst], worst)
    print(f"{name:7s} max |loss - lns| over {N} steps {res[name][0]:.2e} | final loss {l1[-1]:.5f} (lns {ref_l[-1]:.5f}) | worst parameter {worst} rel {r[worst]:.2e}")
    d = np.abs(l1 - ref_l)
    print("        |loss - lns| at steps " + " ".join(f"{k + 1}:{d[k]:.1e}" for k in (0, 1, 2, 4, 9, 19, N - 1)))
others = ("start", "step", "noclip")
if X64:
    ok = res["equiv"][0] < 1e-9 and res["equiv"][1] < 1e-9 and all(res[k][i] > 1e4 * res["equiv"][i] for k in others for i in (0, 1))
else:
    ok = res["equiv"][0] < 1e-5 and res["equiv"][1] < 1e-4 and all(res[k][1] > 1e-2 for k in others)
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
