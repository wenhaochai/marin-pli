"""CPU tests for the late start (per_layer_qwen3: pls_on_start, pls_on_end; launcher PLS_ON).

Run from the worktree on a vis node, the worktree first on PYTHONPATH:
    nice -n 19 .venv/bin/python -u scripts/della/pls_lateon_cpu_test.py
1. The forward pass does not depend on the step: loss and every per-layer metric equal plain separate heads at every
   step of the schedule.
2. Before pls_on_start every gradient (heads and backbone) equals the probes' (pls_detach_backbone).
3. From pls_on_end on every gradient equals plain separate heads'.
4. Inside the ramp the backbone gradient is linear in the gain: g = g_probes + a (g_sep - g_probes), a = (step - S)/(E - S);
   the heads' gradients equal both endpoints'; pls/bb_gain reports a.
5. A switch (S == E): gain 0 at step S - 1, 1 from step S on (step E is the first full step, as PLS_OFF's E is the first off step).
6. The config rejects invalid combinations.
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
from levanter.trainer import _TRACED_TRAIN_STEP  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

B, T, V, L = 4, 32, 300, 6
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
S, E = 10, 20


def batch(seed):
    r = np.random.default_rng(seed)
    tok = r.integers(2, V, size=(B, T))
    return LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)),
                     loss_weight=hax.named(jnp.asarray((np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)), (Batch, Pos)),
                     attn_mask=AttentionMask.causal())


common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA,
              pls_separate_heads=True, hybrid_norm=True, pls_weight=1.0)


def with_cfg(m, **kw):
    """The same weights under another config (the config lives on the model and its transformer)."""
    c = dataclasses.replace(m.config, **kw)
    return dataclasses.replace(m, transformer=dataclasses.replace(m.transformer, config=c))


def trained(m, steps=8):
    """A few Adam steps (plain separate heads) so the heads and backbone are not at init."""
    import optax
    opt = optax.adam(3e-3)
    st = opt.init(eqx.filter(m, eqx.is_array))

    @eqx.filter_jit
    def step(m, st, ex):
        g = eqx.filter_grad(lambda mm: mm.compute_next_token_loss(ex, key=jrandom.PRNGKey(1))[0].array)(m)
        u, st = opt.update(g, st, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, u), st

    for i in range(steps):
        m, st = step(m, st, batch(100 + i))
    return m


def run(m, ex, step):
    """(loss, metrics, grads) at train step ``step`` (None: outside a train step)."""
    token = _TRACED_TRAIN_STEP.set(None if step is None else jnp.asarray(step, jnp.int32))
    try:
        def f(mm):
            loss, stats = mm.compute_next_token_loss(ex, key=jrandom.PRNGKey(3))
            return loss.array, stats
        (loss, stats), g = eqx.filter_value_and_grad(f, has_aux=True)(m)
    finally:
        _TRACED_TRAIN_STEP.reset(token)
    return float(loss), {k: float(v.value()) for k, v in stats.items()}, g


def parts(g):
    """(backbone leaves, head leaves): the transformer blocks + embeddings, and aux heads/norms + lm_head + final norm."""
    leaves = lambda t: [np.asarray(x.array if isinstance(x, hax.NamedArray) else x, np.float64) for x in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array), is_leaf=lambda x: isinstance(x, hax.NamedArray))]
    backbone = leaves(g.transformer.layers) + leaves(g.embeddings)
    heads = leaves(g.aux_lm_heads) + leaves(g.aux_norms) + leaves(g.lm_head) + leaves(g.transformer.norm)
    return backbone, heads


def close(a, b, tol=1e-6):
    worst = 0.0
    for x, y in zip(a, b, strict=True):
        worst = max(worst, float(np.max(np.abs(x - y)) / (np.max(np.abs(y)) + 1e-12)))
    return worst <= tol, worst


passed = failed = 0


def check(name, ok, info=""):
    global passed, failed
    passed, failed = passed + bool(ok), failed + (not ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))


base = trained(PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common), key=jrandom.PRNGKey(0)))
late = with_cfg(base, pls_on_start=S, pls_on_end=E)
probes = with_cfg(base, pls_detach_backbone=True)
ex = batch(7)
l_sep, m_sep, g_sep = run(base, ex, None)
l_pr, m_pr, g_pr = run(probes, ex, None)
bb_sep, hd_sep = parts(g_sep)
bb_pr, hd_pr = parts(g_pr)
check("setup: probes and separate heads differ on the backbone", not close(bb_pr, bb_sep, 1e-3)[0], f"rel {close(bb_pr, bb_sep)[1]:.3g}")

# 1. forward
for step in (0, S - 1, S, (S + E) // 2, E, E + 50):
    l, m, _ = run(late, ex, step)
    same = float(l) == float(l_sep) and all(float(m[k]) == float(m_sep[k]) for k in m_sep)
    check(f"1. forward at step {step} equals separate heads (loss {float(l):.6f})", same)

# 2. before the start
for step in (0, S - 1, S):
    _, m, g = run(late, ex, step)
    bb, hd = parts(g)
    ok_b, w_b = close(bb, bb_pr)
    ok_h, w_h = close(hd, hd_pr)
    check(f"2. step {step}: gradients equal the probes'", ok_b and ok_h and float(m["pls/bb_gain"]) == 0.0, f"backbone rel {w_b:.2g}, heads rel {w_h:.2g}")

# 3. after the end
for step in (E, E + 1, 10_000):
    _, m, g = run(late, ex, step)
    bb, hd = parts(g)
    ok_b, w_b = close(bb, bb_sep)
    ok_h, w_h = close(hd, hd_sep)
    check(f"3. step {step}: gradients equal separate heads'", ok_b and ok_h and float(m["pls/bb_gain"]) == 1.0, f"backbone rel {w_b:.2g}, heads rel {w_h:.2g}")

# 4. inside the ramp
for step in (S + 2, (S + E) // 2, E - 1):
    a = (step - S) / (E - S)
    _, m, g = run(late, ex, step)
    bb, hd = parts(g)
    want = [p + a * (s - p) for p, s in zip(bb_pr, bb_sep)]
    ok_b, w_b = close(bb, want, 1e-5)
    ok_h, w_h = close(hd, hd_sep)
    check(f"4. step {step}: backbone = probes + {a:.2f} (sep - probes), heads unchanged", ok_b and ok_h and abs(float(m["pls/bb_gain"]) - a) < 1e-7, f"backbone rel {w_b:.2g}, heads rel {w_h:.2g}")

# 5. a switch
sw = with_cfg(base, pls_on_start=S, pls_on_end=S)
for step, want_bb, gain in ((S - 1, bb_pr, 0.0), (S, bb_sep, 1.0), (S + 1, bb_sep, 1.0)):
    _, m, g = run(sw, ex, step)
    ok, w = close(parts(g)[0], want_bb)
    check(f"5. switch at {S}: step {step} gain {gain}", ok and float(m["pls/bb_gain"]) == gain, f"rel {w:.2g}")

# outside a train step the late start must refuse to guess
try:
    run(late, ex, None)
    check("1b. no train step: raises", False)
except RuntimeError:
    check("1b. no train step: raises", True)

# 6. validation
bad = [dict(pls_on_start=5), dict(pls_on_start=30, pls_on_end=20), dict(pls_on_end=20, pls_separate_heads=False),
       dict(pls_on_end=20, pls_detach_backbone=True), dict(pls_on_end=20, pls_local_heads=True), dict(pls_on_end=20, pls_off_end=30),
       dict(pls_on_end=20, pls_pcgrad=True), dict(pls_on_end=20, pls_probe=True), dict(pls_on_end=20, pls_heads_only=True)]
for kw in bad:
    try:
        PerLayerQwen3Config(**{**common, **kw})
        check(f"6. rejects {kw}", False)
    except ValueError:
        check(f"6. rejects {kw}", True)
PerLayerQwen3Config(**common, pls_on_start=300, pls_on_end=400)
check("6. accepts pls_on_start=300, pls_on_end=400", True)

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
