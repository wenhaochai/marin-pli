"""CPU tests for experiments.references.table_variants_qwen3 (vocabulary Q5 causal test and Q9 parameter control), with
the full 128,256-token vocabulary on a tiny trunk.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/table_variants_cpu_test.py
Frozen output rows (fo):
1. Init: the model at step 0 is the baseline (same key): head, input table, trunk, logits and loss bit for bit.
2. Gradients: the fixed rare rows get exactly zero; the trainable rows get the plain model's head gradient rows.
3. MuonH labels the fixed rows 'adam', the trainable head 'adamh', the input table 'adam', the trunk 'muonh'.
4. Three real MuonH steps: the fixed rows stay bit-identical, the trainable rows move, the trainable head keeps its
   Frobenius norm (AdamH), the loss falls.
5. The config refuses fo_k outside (0, V - 256) and a tied head.
Low-rank tables (lr):
6. Shapes and parameter count: [V, r] + [r, D] per table; the products' standard deviation is the baseline table's.
7. Loss, eval loss and logits equal the plain model given the product tables; the factor gradients equal the plain
   table gradient pulled back through the product.
8. MuonH labels: input factors 'adam', output factors 'adamh'; three real MuonH steps lower the loss and keep the AdamH
   factors' norms.
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

from experiments.references.table_variants_qwen3 import (FrozenOutQwen3Config, FrozenOutQwen3LMHeadModel, LowRankQwen3Config,
                                                         LowRankQwen3LMHeadModel)

K, V, B, T, D, R = 8000, 128256, 2, 32, 48, 8
Vocab, Batch, Pos = hax.Axis("vocab", V), hax.Axis("batch", B), hax.Axis("position", T)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


A = lambda x: np.asarray(x.array if hasattr(x, "array") else x)  # noqa: E731
tokens = hax.named(jrandom.randint(jrandom.PRNGKey(1), (B, T), 0, V), (Batch, Pos))
tokens = hax.named(tokens.array.at[0, :6].set(jnp.array([127999, 9000, 128000, 5, 64000, 100000])), (Batch, Pos))
ex = LmExample(tokens=tokens, loss_weight=hax.ones((Batch, Pos)).at[Pos, T - 1].set(0.0), attn_mask=AttentionMask.causal())
key = jrandom.PRNGKey(3)


def loss_of(model, k=key):
    out = model.compute_next_token_loss(ex, key=k)
    out = out[0] if isinstance(out, tuple) else out
    return out.scalar() if hasattr(out, "scalar") else out


def muonh_steps(model, n=3):
    """n real MuonH steps (the run's optimizer class, 300m-like settings) on the fixed batch; returns the models."""
    opt = MuonHConfig(learning_rate=0.01, adam_lr=0.002, weight_decay=0.1, warmup=0.0, lr_schedule="constant").build(100)
    params = eqx.filter(model, eqx.is_inexact_array)
    state = opt.init(params)
    out = [model]
    for _ in range(n):
        g = eqx.filter_grad(lambda mm: loss_of(mm))(model)
        upd, state = opt.update(eqx.filter(g, eqx.is_inexact_array), state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, upd)
        out.append(model)
    return out


base = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common), key=jrandom.PRNGKey(0))
bh = A(base.lm_head.weight.rearrange(("vocab", "embed")))

# ---------------- frozen output rows ----------------
fo = FrozenOutQwen3LMHeadModel.init(Vocab, FrozenOutQwen3Config(**common, fo_k=K), key=jrandom.PRNGKey(0))
check("[fo] init builds the fo class", type(fo) is FrozenOutQwen3LMHeadModel, type(fo).__name__)
check("[fo] init: head = baseline head", np.array_equal(A(fo.get_lm_head().rearrange(("vocab", "embed"))), bh))
check("[fo] init: input table = baseline", np.array_equal(A(fo.embeddings.token_embeddings.weight), A(base.embeddings.token_embeddings.weight)))
check("[fo] init: trunk = baseline", all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree_util.tree_leaves(fo.transformer), jax.tree_util.tree_leaves(base.transformer))))
check("[fo] sizes", fo.lm_head.weight.array.shape == (K + 256, D) and fo.fixed_out_rows.array.shape == (V - 256 - K, D), f"{fo.lm_head.weight.array.shape} + {fo.fixed_out_rows.array.shape}")
check("[fo] init: loss = baseline", float(loss_of(fo)) == float(loss_of(base)), f"{float(loss_of(fo)):.6f} vs {float(loss_of(base)):.6f}")
check("[fo] eval loss (key=None) = baseline", float(loss_of(fo, None)) == float(loss_of(base, None)))
check("[fo] init: logits = baseline", np.array_equal(A(fo(tokens, AttentionMask.causal())), A(base(tokens, AttentionMask.causal()))))
check("[fo] Vocab axis", fo.Vocab == Vocab)
g_fo = eqx.filter_grad(lambda mm: loss_of(mm))(fo)
g_b = eqx.filter_grad(lambda mm: loss_of(mm))(base)
check("[fo] fixed rows: gradient exactly zero", not A(g_fo.fixed_out_rows).any())
gb = A(g_b.lm_head.weight.rearrange(("vocab", "embed")))
own = np.concatenate([np.arange(K), np.arange(V - 256, V)])
gs = A(g_fo.lm_head.weight.rearrange(("fo_small", "embed")))
check("[fo] trainable rows: gradient = baseline head gradient rows", np.allclose(gs, gb[own], atol=1e-7, rtol=1e-5), f"max err {np.abs(gs - gb[own]).max():.2e}")
check("[fo] rare rows of the baseline do get gradient (so the test bites)", np.abs(gb[K:V - 256]).sum() > 0)
mask = MuonHConfig(learning_rate=0.01, adam_lr=0.002).create_mask(fo)
check("[fo] MuonH: fixed rows 'adam'", mask.fixed_out_rows == "adam", str(mask.fixed_out_rows))
check("[fo] MuonH: trainable head 'adamh'", mask.lm_head == "adamh" or getattr(mask.lm_head, "weight", None) == "adamh", str(mask.lm_head))
check("[fo] MuonH: input table 'adam'", mask.embeddings.token_embeddings.weight == "adam", str(mask.embeddings.token_embeddings.weight))
steps = muonh_steps(fo)
f0, f3 = A(steps[0].fixed_out_rows), A(steps[-1].fixed_out_rows)
s0, s3 = A(steps[0].lm_head.weight), A(steps[-1].lm_head.weight)
check("[fo] 3 MuonH steps: fixed rows bit-identical", np.array_equal(f0, f3))
check("[fo] 3 MuonH steps: trainable rows move", not np.array_equal(s0, s3), f"max change {np.abs(s3 - s0).max():.2e}")
check("[fo] 3 MuonH steps: trainable head keeps its norm (AdamH)", np.isclose(np.linalg.norm(s0), np.linalg.norm(s3), rtol=1e-5), f"{np.linalg.norm(s0):.6f} -> {np.linalg.norm(s3):.6f}")
l = [float(loss_of(m)) for m in steps]
check("[fo] 3 MuonH steps: loss falls", l[-1] < l[0], " ".join(f"{x:.4f}" for x in l))
for bad in (dict(fo_k=0), dict(fo_k=V - 256), dict(fo_k=K, tie_word_embeddings=True)):
    try:
        FrozenOutQwen3LMHeadModel.init(Vocab, FrozenOutQwen3Config(**common, **bad), key=jrandom.PRNGKey(0))
        check(f"[fo] refuses {bad}", False)
    except ValueError:
        check(f"[fo] refuses {bad}", True)

# ---------------- low-rank tables ----------------
lr = LowRankQwen3LMHeadModel.init(Vocab, LowRankQwen3Config(**common, lr_rank=R), key=jrandom.PRNGKey(0))
check("[lr] init builds the lr class", type(lr) is LowRankQwen3LMHeadModel, type(lr).__name__)
check("[lr] shapes", lr.embeddings.token_embeddings.weight.array.shape == (V, R) and lr.embeddings_b.array.shape == (R, D)
      and lr.lm_head.weight.array.shape == (V, R) and lr.lm_head_b.array.shape == (R, D))
n_tab = sum(x.size for x in (lr.embeddings.token_embeddings.weight.array, lr.embeddings_b.array, lr.lm_head.weight.array, lr.lm_head_b.array))
check("[lr] table parameters = 2 (V r + r D)", n_tab == 2 * (V * R + R * D), f"{n_tab}")
check("[lr] trunk = baseline", all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree_util.tree_leaves(lr.transformer), jax.tree_util.tree_leaves(base.transformer))))
plain = lr._plain()
for name, full, ref in (("input", plain.embeddings.token_embeddings.weight, base.embeddings.token_embeddings.weight), ("output", lr.get_lm_head(), base.lm_head.weight)):
    s_l, s_b = A(full).std(), A(ref).std()
    check(f"[lr] {name} product std = baseline table std", abs(s_l / s_b - 1) < 0.1, f"{s_l:.5f} vs {s_b:.5f}")
ai, bi = A(lr.embeddings.token_embeddings.weight.rearrange(("vocab", "lr_rank"))), A(lr.embeddings_b.rearrange(("lr_rank", "embed")))
check("[lr] input table = A @ B", np.allclose(A(plain.embeddings.token_embeddings.weight.rearrange(("vocab", "embed"))), ai @ bi, atol=1e-6))
check("[lr] loss = plain model with the product tables", np.isclose(float(loss_of(lr)), float(loss_of(plain)), atol=1e-6))
check("[lr] eval loss (key=None)", np.isclose(float(loss_of(lr, None)), float(loss_of(plain, None)), atol=1e-6))
check("[lr] logits", np.allclose(A(lr(tokens, AttentionMask.causal())), A(plain(tokens, AttentionMask.causal())), atol=1e-5))
g_l = eqx.filter_grad(lambda mm: loss_of(mm))(lr)
g_p = eqx.filter_grad(lambda mm: loss_of(mm))(plain)
for name, ga, gbf, gfull, a_, b_ in (("input", g_l.embeddings.token_embeddings.weight, g_l.embeddings_b, g_p.embeddings.token_embeddings.weight, ai, bi),
                                     ("output", g_l.lm_head.weight, g_l.lm_head_b, g_p.lm_head.weight, A(lr.lm_head.weight.rearrange(("vocab", "lr_rank"))), A(lr.lm_head_b.rearrange(("lr_rank", "embed"))))):
    G = A(gfull.rearrange(("vocab", "embed"))).astype(np.float64)
    gA, gB = A(ga.rearrange(("vocab", "lr_rank"))), A(gbf.rearrange(("lr_rank", "embed")))
    check(f"[lr] {name} factor gradients = pulled back (G B^T, A^T G)", np.allclose(gA, G @ b_.T, atol=1e-7, rtol=1e-4) and np.allclose(gB, a_.T @ G, atol=1e-7, rtol=1e-4),
          f"max err {max(np.abs(gA - G @ b_.T).max(), np.abs(gB - a_.T @ G).max()):.2e}")
mask = MuonHConfig(learning_rate=0.01, adam_lr=0.002).create_mask(lr)
check("[lr] MuonH: input factors 'adam'", mask.embeddings.token_embeddings.weight == "adam" and mask.embeddings_b == "adam", f"{mask.embeddings.token_embeddings.weight}, {mask.embeddings_b}")
check("[lr] MuonH: output factors 'adamh'", (mask.lm_head == "adamh" or getattr(mask.lm_head, "weight", None) == "adamh") and mask.lm_head_b == "adamh", f"{mask.lm_head}, {mask.lm_head_b}")
steps = muonh_steps(lr)
l = [float(loss_of(m)) for m in steps]
check("[lr] 3 MuonH steps: loss falls", l[-1] < l[0], " ".join(f"{x:.4f}" for x in l))
for nm, get in (("lm_head", lambda m: A(m.lm_head.weight)), ("lm_head_b", lambda m: A(m.lm_head_b))):
    n0, n3 = np.linalg.norm(get(steps[0])), np.linalg.norm(get(steps[-1]))
    check(f"[lr] 3 MuonH steps: {nm} keeps its norm (AdamH)", np.isclose(n0, n3, rtol=1e-5), f"{n0:.6f} -> {n3:.6f}")
check("[lr] 3 MuonH steps: input factors move", not np.array_equal(A(steps[0].embeddings_b), A(steps[-1].embeddings_b)))

print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
