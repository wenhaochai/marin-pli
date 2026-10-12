"""CPU tests for the Q10 variants in experiments.references.table_variants_qwen3 (owner, 2026-10-11), on the full
128,256-token vocabulary with a tiny trunk.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/q10_cpu_test.py
Rare-target mask (rm):
1. Same initialization as the baseline; outside a train step (evaluation) and before rm_from_step the loss is the
   baseline's, bit for bit.
2. From rm_from_step on (the step a traced jax value, as in Trainer._train_step), the loss equals sum(w m l) / sum(w)
   computed by hand from the logits (m = 0 where the target id lies in [rm_k, V - 256)), and the gradient equals the
   gradient of that hand-written loss.
3. The config refuses rm_k = 0 and a negative rm_from_step.
Composed input with residual (cvr):
4. At init the residuals are zero and the model is the composed-input (cv) model bit for bit (same key).
5. With random residuals, the input row of a rare token is mean(pieces) + 0.1 r, the rows below K are unchanged, and the
   residual gradient is 0.1 x the full table's gradient rows.
6. MuonH labels the residuals 'adam' (as the input table); three real MuonH steps lower the loss and move the residuals.
7. The config refuses a composed head and a non-positive scale.
"""
import dataclasses
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

from experiments.references.composed_vocab_qwen3 import ComposedVocabQwen3Config, ComposedVocabQwen3LMHeadModel
from experiments.references.table_variants_qwen3 import (ComposedResidualQwen3Config, ComposedResidualQwen3LMHeadModel,
                                                         RareTargetMaskQwen3Config, RareTargetMaskQwen3LMHeadModel)

MAP = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenizers/marin-small/v8000/expand_full.npz"
K, V, B, T, D = 8000, 128256, 2, 32, 48
Vocab, Batch, Pos = hax.Axis("vocab", V), hax.Axis("batch", B), hax.Axis("position", T)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


A = lambda x: np.asarray(x.array if hasattr(x, "array") else x)  # noqa: E731
tok = jrandom.randint(jrandom.PRNGKey(1), (B, T), 0, V)
# rare targets (8000 <= id < 128000), common ones and specials, at known places
tok = tok.at[0, :8].set(jnp.array([5, 127999, 9000, 128000, 7, 64000, 100000, 3]))
tokens = hax.named(tok, (Batch, Pos))
lw = hax.ones((Batch, Pos)).at[Pos, 5].set(0.0)   # a caller weight of 0 somewhere, to check sum(w)
ex = LmExample(tokens=tokens, loss_weight=lw, attn_mask=AttentionMask.causal())
key = jrandom.PRNGKey(3)


def loss_of(model, k=key):
    out = model.compute_next_token_loss(ex, key=k)
    out = out[0] if isinstance(out, tuple) else out
    return out.scalar() if hasattr(out, "scalar") else out


def at_step(fn, step):
    """Run fn with the trainer's traced-step contextvar set to `step` (a traced value under jit)."""
    tokn = _TRACED_TRAIN_STEP.set(step)
    try:
        return fn()
    finally:
        _TRACED_TRAIN_STEP.reset(tokn)


def muonh_steps(model, n=3):
    opt = MuonHConfig(learning_rate=0.01, adam_lr=0.002, weight_decay=0.1, warmup=0.0, lr_schedule="constant").build(100)
    state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    out = [model]
    for _ in range(n):
        g = eqx.filter_grad(lambda mm: loss_of(mm))(model)
        upd, state = opt.update(eqx.filter(g, eqx.is_inexact_array), state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, upd)
        out.append(model)
    return out


# ---------------- rare-target mask ----------------
THR = 100
base = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common), key=jrandom.PRNGKey(0))
rm = RareTargetMaskQwen3LMHeadModel.init(Vocab, RareTargetMaskQwen3Config(**common, rm_k=K, rm_from_step=THR), key=jrandom.PRNGKey(0))
check("[rm] init builds the rm class (Qwen3LMHeadModel.init returns its base class)", type(rm) is RareTargetMaskQwen3LMHeadModel, type(rm).__name__)
check("[rm] init = baseline", all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree_util.tree_leaves(rm), jax.tree_util.tree_leaves(base))))
lb = float(loss_of(base))
check("[rm] no train step (evaluation): loss = baseline", float(loss_of(rm)) == lb)
jl = jax.jit(lambda m, st: at_step(lambda: loss_of(m), st))
check("[rm] traced step < rm_from_step: loss = baseline", np.isclose(float(jl(rm, jnp.int32(THR - 1))), lb, rtol=1e-6, atol=1e-7), f"{float(jl(rm, jnp.int32(THR - 1))):.7f} vs {lb:.7f}")


def manual_loss(model):
    logits = model(tokens, AttentionMask.causal()).rearrange(("batch", "position", "vocab")).array.astype(jnp.float32)
    lp = jax.nn.log_softmax(logits, axis=-1)
    tgt = jnp.roll(tok, -1, axis=1)
    ell = -jnp.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0]
    w = lw.rearrange(("batch", "position")).array.at[:, -1].set(0.0)   # the last position predicts nothing
    m = ~((tgt >= K) & (tgt < V - 256))
    return jnp.sum(w * m * ell) / jnp.sum(w)


got = float(jl(rm, jnp.int32(THR)))
want = float(manual_loss(base))
check("[rm] traced step >= rm_from_step: loss = sum(w m l) / sum(w)", np.isclose(got, want, rtol=1e-5), f"{got:.6f} vs {want:.6f} (baseline {lb:.6f})")
tgt = np.roll(np.asarray(tok), -1, axis=1)
check("[rm] the test batch has rare and common targets", ((tgt >= K) & (tgt < V - 256)).any() and ((tgt < K) | (tgt >= V - 256)).any())
g_rm = jax.jit(lambda m, st: at_step(lambda: eqx.filter_grad(lambda mm: loss_of(mm))(m), st))(rm, jnp.int32(THR))
g_man = eqx.filter_grad(manual_loss)(base)
errs = [(float(np.abs(np.asarray(a) - np.asarray(b)).max()), float(np.abs(np.asarray(b)).max())) for a, b in zip(jax.tree_util.tree_leaves(g_rm), jax.tree_util.tree_leaves(g_man))]
rel = max(e / max(m, 1e-12) for e, m in errs)
check("[rm] gradient = gradient of the hand-written loss (max error / leaf max < 1e-3)", rel < 1e-3, f"worst relative {rel:.2e}; per leaf max abs err {max(e for e, _ in errs):.2e}")
g_base = jax.jit(lambda m, st: at_step(lambda: eqx.filter_grad(lambda mm: loss_of(mm))(m), st))(rm, jnp.int32(THR - 1))
d = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(jax.tree_util.tree_leaves(g_rm), jax.tree_util.tree_leaves(g_base)))
check("[rm] the mask changes the gradient (so the test bites)", d > 1e-4, f"max change {d:.2e}")
for bad in (dict(rm_k=0), dict(rm_from_step=-1)):
    try:
        RareTargetMaskQwen3Config(**common, **bad)
        check(f"[rm] refuses {bad}", False)
    except ValueError:
        check(f"[rm] refuses {bad}", True)

# ---------------- composed input with residual ----------------
cv = ComposedVocabQwen3LMHeadModel.init(Vocab, ComposedVocabQwen3Config(**common, cv_map=MAP, cv_input=True), key=jrandom.PRNGKey(0))
cvr = ComposedResidualQwen3LMHeadModel.init(Vocab, ComposedResidualQwen3Config(**common, cv_map=MAP, cv_input=True, cv_res_scale=0.1), key=jrandom.PRNGKey(0))
check("[cvr] init builds the cvr class", type(cvr) is ComposedResidualQwen3LMHeadModel, type(cvr).__name__)
check("[cvr] residual rows: one per rare token, zero", cvr.embeddings_res.array.shape == (V - 256 - K, D) and not A(cvr.embeddings_res).any())
full_cv = A(cv._full().embeddings.token_embeddings.weight.rearrange(("vocab", "embed")))
full_cvr = A(cvr._full().embeddings.token_embeddings.weight.rearrange(("vocab", "embed")))
check("[cvr] init: input table = composed model's", np.array_equal(full_cv, full_cvr))
check("[cvr] init: loss = composed model's", float(loss_of(cvr)) == float(loss_of(cv)))
r = jrandom.normal(jrandom.PRNGKey(7), (V - 256 - K, D)) * 0.02
cvr2 = dataclasses.replace(cvr, embeddings_res=hax.named(r, cvr.embeddings_res.axes))
f2 = A(cvr2._full().embeddings.token_embeddings.weight.rearrange(("vocab", "embed")))
check("[cvr] rare rows = mean(pieces) + 0.1 r", np.allclose(f2[K:V - 256], full_cv[K:V - 256] + 0.1 * np.asarray(r), atol=1e-6))
check("[cvr] rows below K and specials unchanged", np.array_equal(f2[:K], full_cv[:K]) and np.array_equal(f2[V - 256:], full_cv[V - 256:]))
g = eqx.filter_grad(lambda mm: loss_of(mm))(cvr2)
plain = cvr2._full()
gp = eqx.filter_grad(lambda mm: loss_of(mm))(plain)
G = A(gp.embeddings.token_embeddings.weight.rearrange(("vocab", "embed")))
check("[cvr] residual gradient = 0.1 x table gradient rows", np.allclose(A(g.embeddings_res), 0.1 * G[K:V - 256], rtol=1e-4, atol=1e-9), f"max err {np.abs(A(g.embeddings_res) - 0.1 * G[K:V - 256]).max():.2e}")
mask = MuonHConfig(learning_rate=0.01, adam_lr=0.002).create_mask(cvr)
check("[cvr] MuonH: residuals 'adam', input table 'adam', head 'adamh'", mask.embeddings_res == "adam" and mask.embeddings.token_embeddings.weight == "adam"
      and (mask.lm_head == "adamh" or getattr(mask.lm_head, "weight", None) == "adamh"), f"{mask.embeddings_res}, {mask.embeddings.token_embeddings.weight}, {mask.lm_head}")
steps = muonh_steps(cvr)
l = [float(loss_of(m)) for m in steps]
check("[cvr] 3 MuonH steps: loss falls", l[-1] < l[0], " ".join(f"{x:.4f}" for x in l))
check("[cvr] 3 MuonH steps: residuals move", A(steps[-1].embeddings_res).any())
for bad in (dict(cv_input=True, cv_output=True), dict(cv_input=True, cv_res_scale=0.0)):
    try:
        ComposedResidualQwen3Config(**common, cv_map=MAP, **bad)
        check(f"[cvr] refuses {bad}", False)
    except ValueError:
        check(f"[cvr] refuses {bad}", True)

print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
