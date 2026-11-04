"""CPU tests of experiments.references.vocab_shrink_qwen3 (owner 2026-11-04), on the full 128,256-token vocabulary with a tiny trunk.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/vocab_shrink_cpu_test.py
1. With K = every row the loss is the baseline's, in training and in evaluation.
2. At a train step in a stretch with K, the loss and its gradient equal a hand-written softmax over the ids below K and the
   specials (targets outside K get weight 0). The switch takes the right K at every stretch (traced step).
3. Evaluation uses vs_eval_k, reduced and per position.
4. Dead rows get a zero gradient from the softmax. MuonH steps across a cut: before it the optimizer is plain MuonH bit for
   bit; after it the dead rows do not move, the kept rows keep their norm, and the kept rows turn at AdamH's rate
   (lr x norm(p_kept) before the renormalization), the same as an independent K-row model given the same gradient.
5. The schedule parser refuses bad schedules.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.optim.muonh import MuonHConfig
from levanter.trainer import _TRACED_TRAIN_STEP

from experiments.references.vocab_shrink_qwen3 import VocabShrinkMuonHConfig, VocabShrinkQwen3Config, VocabShrinkQwen3LMHeadModel, parse_schedule

V, B, T, D = 128256, 2, 32, 48
Vocab, Batch, Pos = hax.Axis("vocab", V), hax.Axis("batch", B), hax.Axis("position", T)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


tok = jrandom.randint(jrandom.PRNGKey(1), (B, T), 0, 7000)
tok = tok.at[0, :8].set(jnp.array([5, 127999, 9000, 128000, 7, 64000, 100000, 3]))   # outside K, specials and common ids, at known places
tokens = hax.named(tok, (Batch, Pos))
lw = hax.ones((Batch, Pos)).at[Pos, 5].set(0.0)
ex = LmExample(tokens=tokens, loss_weight=lw, attn_mask=AttentionMask.causal())
key = jrandom.PRNGKey(3)


def loss_of(model, k=key):
    out = model.compute_next_token_loss(ex, key=k)
    out = out[0] if isinstance(out, tuple) else out
    return out.scalar() if hasattr(out, "scalar") else out


def at_step(fn, step):
    tokn = _TRACED_TRAIN_STEP.set(step)
    try:
        return fn()
    finally:
        _TRACED_TRAIN_STEP.reset(tokn)


def manual_loss(model, k):
    logits = Qwen3LMHeadModel.__call__(model, tokens, AttentionMask.causal()).rearrange(("batch", "position", "vocab")).array.astype(jnp.float32)
    keep = jnp.concatenate([jnp.arange(k), jnp.arange(V - 256, V)])
    lp = jax.nn.log_softmax(logits[..., keep], axis=-1)
    tgt = jnp.roll(tok, -1, axis=1)
    outside = (tgt >= k) & (tgt < V - 256)
    tgt2 = jnp.where(tgt < k, tgt, jnp.where(tgt >= V - 256, tgt - (V - 256) + k, 0))
    ell = -jnp.take_along_axis(lp, tgt2[..., None], axis=-1)[..., 0]
    w = lw.rearrange(("batch", "position")).array.at[:, -1].set(0.0)
    m = ~outside
    return jnp.sum(w * m * ell) / jnp.sum(w * m)


base = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common), key=jrandom.PRNGKey(0))
cfg = VocabShrinkQwen3Config(**common, vs_ks="128000,16000,8000", vs_steps="0,10,20", vs_eval_k=8000)
mdl = VocabShrinkQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
check("init builds the class and equals the baseline", type(mdl) is VocabShrinkQwen3LMHeadModel and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree_util.tree_leaves(mdl), jax.tree_util.tree_leaves(base))))
lb = float(loss_of(base))
jl = jax.jit(lambda m, st: at_step(lambda: loss_of(m), st))
got0 = float(jl(mdl, jnp.int32(9)))
check("step in the 128K stretch: loss = baseline", np.isclose(got0, lb, rtol=1e-6), f"{got0:.7f} vs {lb:.7f}")
for st, k in ((10, 16000), (19, 16000), (20, 8000), (500, 8000)):
    got, want = float(jl(mdl, jnp.int32(st))), float(manual_loss(base, k))
    check(f"step {st}: loss = softmax over K={k}", np.isclose(got, want, rtol=1e-5), f"{got:.6f} vs {want:.6f}")
ge = float(loss_of(mdl))
we = float(manual_loss(base, 8000))
check("no train step: evaluation uses vs_eval_k", np.isclose(ge, we, rtol=1e-5), f"{ge:.6f} vs {we:.6f}")
pp = mdl.compute_next_token_loss(ex, key=None, reduction=None, reduction_axis=())
check("evaluation, per position: same mean", np.isclose(float(jnp.sum(pp.array) / float(jnp.sum((lw.array.at[:, -1].set(0.0)) * ~((jnp.roll(tok, -1, axis=1) >= 8000) & (jnp.roll(tok, -1, axis=1) < V - 256))))), we, rtol=1e-4), f"shape {pp.array.shape}")
full = VocabShrinkQwen3LMHeadModel.init(Vocab, VocabShrinkQwen3Config(**common), key=jrandom.PRNGKey(0))
check("K = every row: evaluation and training = baseline", float(loss_of(full)) == lb and np.isclose(float(jl(full, jnp.int32(5))), lb, rtol=1e-6))

# gradients
g_cut = jax.jit(lambda m, st: at_step(lambda: eqx.filter_grad(lambda mm: loss_of(mm))(m), st))(mdl, jnp.int32(20))
g_man = eqx.filter_grad(lambda mm: manual_loss(mm, 8000))(base)
errs = [(float(np.abs(np.asarray(a) - np.asarray(b)).max()), float(np.abs(np.asarray(b)).max())) for a, b in zip(jax.tree_util.tree_leaves(g_cut), jax.tree_util.tree_leaves(g_man))]
rel = max(e / max(m, 1e-12) for e, m in errs)
check("gradient = gradient of the hand-written loss (max error / leaf max < 1e-3)", rel < 1e-3, f"worst relative {rel:.2e}")
gh = g_cut.lm_head.weight.rearrange(("vocab", "embed")).array
check("dead rows get a zero gradient, kept rows do not", float(jnp.abs(gh[8000:V - 256]).max()) == 0 and float(jnp.abs(gh[:8000]).max()) > 0)

# the optimizer across a cut
CUT = 2
kw = dict(learning_rate=0.01, adam_lr=0.002, weight_decay=0.1, warmup=0.0, lr_schedule="constant")
ks = ",".join(str(x) for x in (128000, 8000))
opt_vs = VocabShrinkMuonHConfig(**kw, vs_ks=ks, vs_steps=f"0,{CUT}").build(100)
opt_plain = MuonHConfig(**kw).build(100)
m1 = VocabShrinkQwen3LMHeadModel.init(Vocab, VocabShrinkQwen3Config(**common, vs_ks=ks, vs_steps=f"0,{CUT}", vs_eval_k=8000), key=jrandom.PRNGKey(0))
m2 = m1
s1, s2 = opt_vs.init(eqx.filter(m1, eqx.is_inexact_array)), opt_plain.init(eqx.filter(m2, eqx.is_inexact_array))
grad = jax.jit(lambda m, st: at_step(lambda: eqx.filter_grad(lambda mm: loss_of(mm))(m), st))


def head(m):
    return m.lm_head.weight.rearrange(("vocab", "embed")).array


for st in range(CUT):
    g = grad(m1, jnp.int32(st))
    u1, s1 = opt_vs.update(eqx.filter(g, eqx.is_inexact_array), s1, eqx.filter(m1, eqx.is_inexact_array))
    g2 = grad(m2, jnp.int32(st))
    u2, s2 = opt_plain.update(eqx.filter(g2, eqx.is_inexact_array), s2, eqx.filter(m2, eqx.is_inexact_array))
    m1, m2 = eqx.apply_updates(m1, u1), eqx.apply_updates(m2, u2)
check("before the cut: plain MuonH bit for bit", all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree_util.tree_leaves(m1), jax.tree_util.tree_leaves(m2))))
for st in range(CUT, CUT + 3):
    before = head(m1)
    g = grad(m1, jnp.int32(st))
    u1, s1 = opt_vs.update(eqx.filter(g, eqx.is_inexact_array), s1, eqx.filter(m1, eqx.is_inexact_array))
    m1 = eqx.apply_updates(m1, u1)
    after = head(m1)
    keep = jnp.concatenate([jnp.arange(8000), jnp.arange(V - 256, V)])
    dead_moved = float(jnp.abs(after[8000:V - 256] - before[8000:V - 256]).max())
    nb, na = float(jnp.linalg.norm(before[keep])), float(jnp.linalg.norm(after[keep]))
    turn = float(jnp.linalg.norm(after[keep] - before[keep]) / nb)
    check(f"update {st}: dead rows still, kept norm kept, kept rows turn about lr", dead_moved == 0 and np.isclose(nb, na, rtol=1e-4) and 0.005 < turn < 0.02, f"moved {dead_moved}, norm {nb:.4f} -> {na:.4f}, turn {turn:.5f} (lr 0.01)")

for bad in (("128000,8000", "0"), ("8000,128000", "0,5"), ("128000,8000", "1,5"), ("128000,8000", "5,5")):
    try:
        parse_schedule(*bad)
        check(f"refuses {bad}", False)
    except ValueError:
        check(f"refuses {bad}", True)
print("ALL PASS" if not fails else f"FAILED: {fails}")
raise SystemExit(1 if fails else 0)
