"""CPU tests for per_layer_qwen3.SplitAuxMuonHConfig and depth_arch_qwen3.DepthArchSplitAuxMuonHConfig (PLS_OPT_SPLIT=1).

    WT=<worktree>; PYTHONPATH=$WT:$WT/lib/*/src JAX_PLATFORMS=cpu nice -n 19 <venv>/bin/python scripts/della/pls_optsplit_cpu_test.py
1. Labels: aux_lm_heads -> aux_adamh, aux_norms -> aux_adam; every other leaf keeps MuonHConfig's label (DepthArch: the
   DepthArchMuonHConfig label, adam_new included).
2. Probes (separate heads, backbone detached) at L = 6 and 48, the 130m recipe's optimizer constants, clipping on: after
   every step the backbone (embeddings, blocks, final norm, lm_head) equals the baseline Qwen3's trained with MuonHConfig,
   up to float rounding; with MuonHConfig instead it drifts away (the check is not vacuous).
3. Without clipping, SplitAux and MuonHConfig give the same model after a step up to float rounding (same transforms for
   every leaf; the two optax graphs round differently, and training amplifies that over steps).
4. DepthArchSplitAux (arch hc, separate heads): labels, and without clipping one step equals DepthArchMuonHConfig's.
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
import haliax.nn as hnn  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402

from experiments.references.depth_arch_qwen3 import DepthArchMuonHConfig, DepthArchSplitAuxMuonHConfig  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel, SplitAuxMuonHConfig  # noqa: E402

B, T, V = 8, 16, 128
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
perm = np.random.default_rng(0).permutation(V)
FAIL = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def batch(i):
    r = np.random.default_rng(1000 + i)
    tok = np.zeros((B, T), np.int64)
    tok[:, 0] = r.integers(0, V, B)
    for t in range(1, T):
        tok[:, t] = np.where(r.random(B) < 0.2, r.integers(0, V, B), perm[tok[:, t - 1]])
    lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)
    return LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)), attn_mask=AttentionMask.causal())


def common(L):
    return dict(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True, use_qk_norm=False, attn_backend=AttentionBackend.VANILLA)


def recipe(cls, clip, steps, **kw):
    return cls(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5, muon_epsilon=1e-5,
               max_grad_norm=clip, weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0, coefficient_type="simple", **kw)


def train(model, cfg, steps):
    opt = cfg.build(steps)
    state = opt.init(eqx.filter(model, eqx.is_array))

    def lf(m, ex):
        out = m.compute_next_token_loss(ex, key=jrandom.PRNGKey(1), logsumexp_weight=0.0)
        loss = out[0] if isinstance(out, tuple) else out
        return loss.array if isinstance(loss, hax.NamedArray) else loss

    @eqx.filter_jit
    def step(m, s, ex):
        g = eqx.filter_grad(lf)(m, ex)
        upd, s = opt.update(g, s, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, upd), s

    out = []
    for i in range(steps):
        model, state = step(model, state, batch(i))
        out.append(model)
    return out


def backbone(m):
    return eqx.filter((m.embeddings, m.transformer, m.lm_head), eqx.is_array)


def maxdiff(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    return max(float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)))) for x, y in zip(la, lb))


def labels(mask, params):
    """(path, label) for every array leaf, Linear label trees expanded."""
    flat = jax.tree_util.tree_leaves_with_path(mask, is_leaf=lambda x: isinstance(x, str))
    return {jax.tree_util.keystr(p): lab for p, lab in flat}


key0 = jrandom.PRNGKey(0)

# 1. labels
m6 = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common(6), pls_weight=1.0, pls_separate_heads=True, pls_detach_backbone=True), key=key0)
p6 = eqx.filter(m6, eqx.is_array)
base_l, split_l = labels(MuonHConfig().create_mask(p6), p6), labels(SplitAuxMuonHConfig().create_mask(p6), p6)
check(base_l.keys() == split_l.keys(), f"1. same leaves labelled ({len(base_l)})")
aux = {k for k in base_l if k.startswith((".aux_lm_heads", ".aux_norms"))}
check(len(aux) > 0 and all(split_l[k] == "aux_" + base_l[k] for k in aux), f"1. {len(aux)} aux leaves relabelled aux_<label>: {sorted({split_l[k] for k in aux})}")
check(all(split_l[k] == base_l[k] for k in base_l if k not in aux), "1. every other leaf keeps MuonHConfig's label")
check({base_l[k] for k in aux} == {"adamh", "adam"}, "1. aux heads were adamh, aux norms adam under MuonHConfig")

# 2-3. probes vs baseline, L = 6 and 48
for L, steps in ((6, 25), (48, 25)):
    base = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common(L)), key=key0)
    probe = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common(L), pls_weight=1.0, pls_separate_heads=True, pls_detach_backbone=True), key=key0)
    check(maxdiff(backbone(base), backbone(probe)) == 0.0, f"2. L={L}: backbone init identical")
    tb = train(base, recipe(MuonHConfig, 1.0, steps), steps)
    ts = train(probe, recipe(SplitAuxMuonHConfig, 1.0, steps), steps)
    tm = train(probe, recipe(MuonHConfig, 1.0, steps), steps)
    d_split = max(maxdiff(backbone(a), backbone(b)) for a, b in zip(tb, ts))
    d_mix = max(maxdiff(backbone(a), backbone(b)) for a, b in zip(tb, tm))
    tn_s = train(probe, recipe(SplitAuxMuonHConfig, None, steps), steps)
    tn_m = train(probe, recipe(MuonHConfig, None, steps), steps)
    d1 = maxdiff(eqx.filter(tn_s[0], eqx.is_array), eqx.filter(tn_m[0], eqx.is_array))
    d_noclip = max(maxdiff(eqx.filter(a, eqx.is_array), eqx.filter(b, eqx.is_array)) for a, b in zip(tn_s, tn_m))
    check(d_split < 1e-4, f"2. L={L}, clip 1.0, {steps} steps: SplitAux probes backbone == baseline (max diff {d_split:.2e})")
    check(d_mix > 100 * max(d_split, d_noclip), f"2. L={L}: MuonHConfig probes backbone drifts from the baseline (max diff {d_mix:.2e}), "
                                                f"over 100x the rounding drift of test 3 ({d_noclip:.1e}), so the check is not vacuous")
    check(d1 < 1e-6, f"3. L={L}, no clipping: SplitAux == MuonHConfig for every parameter after step 1 up to rounding (max diff {d1:.1e}; "
                     f"after {steps} steps {d_noclip:.1e}: the two optax graphs round differently and training amplifies it)")

# 4. depth-arch variant (hc, separate heads)
mh = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common(4), pls_weight=1.0, pls_separate_heads=True, depth_arch="hc", scan_layers=False), key=key0)
ph = eqx.filter(mh, eqx.is_array)
da_l, das_l = labels(DepthArchMuonHConfig().create_mask(ph), ph), labels(DepthArchSplitAuxMuonHConfig().create_mask(ph), ph)
aux_h = {k for k in da_l if k.startswith((".aux_lm_heads", ".aux_norms"))}
check(len(aux_h) > 0 and all(das_l[k] == "aux_" + da_l[k] for k in aux_h) and all(das_l[k] == da_l[k] for k in da_l if k not in aux_h), f"4. hc: {len(aux_h)} aux leaves relabelled, the rest (adam_new {sum(v == 'adam_new' for v in da_l.values())}) unchanged")
a = train(mh, recipe(DepthArchSplitAuxMuonHConfig, None, 3), 3)
b = train(mh, recipe(DepthArchMuonHConfig, None, 3), 3)
d4 = maxdiff(eqx.filter(a[0], eqx.is_array), eqx.filter(b[0], eqx.is_array))
check(d4 < 1e-6, f"4. hc, no clipping: DepthArchSplitAux == DepthArchMuonHConfig after step 1 up to rounding (max diff {d4:.1e})")
c = train(mh, recipe(DepthArchSplitAuxMuonHConfig, 1.0, 3), 3)
check(all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(eqx.filter(c[-1], eqx.is_array))), "4. hc, clip 1.0, 3 steps: finite")

print("ALL OPTSPLIT CPU TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILED")
