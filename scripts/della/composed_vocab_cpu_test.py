"""CPU tests for experiments.references.composed_vocab_qwen3 (vocabulary Q4 and Q5), with the real 8K map and the full
128,256-token vocabulary on a tiny trunk.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/composed_vocab_cpu_test.py
1. The map: a token below 8K is its own piece, a special token maps to 8000 + its index, and a sample of rare tokens'
   pieces re-encode to the token's bytes (the K-token BPE).
2. Composition: every row of the composed table equals the numpy mean of its pieces' small rows.
3. For cv_input, cv_output and both: the training loss (with a key), the eval loss (key=None) and the logits of
   __call__ equal those of the plain Qwen3 model given the composed tables, and the gradient of every small table equals
   the plain model's table gradient pulled back through the mean (numpy).
4. MuonH labels the small input table 'adam' and the small head 'adamh', as for the baseline's tables, and the trunk 'muonh'.
5. Parameter counts: the composed side holds 8,256 rows instead of 128,256.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from tokenizers import Tokenizer

import haliax as hax
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.optim.muonh import MuonHConfig

from experiments.references.composed_vocab_qwen3 import ComposedVocabQwen3Config, ComposedVocabQwen3LMHeadModel, load_map

MAP_DIR = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenizers/marin-small/v8000"
MAP = f"{MAP_DIR}/expand_full.npz"
K, V, B, T, D = 8000, 128256, 2, 32, 48
Vocab, Batch, Pos = hax.Axis("vocab", V), hax.Axis("batch", B), hax.Axis("position", T)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


# 1. the map
flat, rows, counts, n_small = load_map(MAP)
offs = np.concatenate([[0], np.cumsum(counts.astype(np.int64))])
pieces = lambda t: flat[offs[t]:offs[t + 1]]  # noqa: E731
check("map size", len(counts) == V and n_small == K + 256, f"{len(counts)} tokens, {n_small} small rows")
check("below K: own piece", all(pieces(t).tolist() == [t] for t in range(0, K, 97)))
check("specials", all(pieces(128000 + j).tolist() == [K + j] for j in range(256)))
small_tok = Tokenizer.from_file(f"{MAP_DIR}/tokenizer.json")
rng = np.random.default_rng(0)
sample = rng.integers(K, 128000, 300)
from experiments.references.small_vocab_tokenizer import SRC  # noqa: E402
full_tok = Tokenizer.from_file(f"{SRC}/tokenizer.json")
if full_tok is not None:
    ok = all(small_tok.decode(pieces(t).tolist()) == full_tok.decode([int(t)]) for t in sample)
    check("rare tokens: pieces decode to the token", ok)
else:
    check("rare tokens: all pieces below K", all((pieces(t) < K).all() and len(pieces(t)) >= 1 for t in sample))


def mean_rows(w_small):
    return np.stack([w_small[pieces(t)].mean(0) for t in range(V)])


def plain_with(model: ComposedVocabQwen3LMHeadModel) -> Qwen3LMHeadModel:
    return model._full()


tokens = hax.named(jrandom.randint(jrandom.PRNGKey(1), (B, T), 0, V), (Batch, Pos))
# make sure rare tokens and specials occur
tokens = hax.named(tokens.array.at[0, :6].set(jnp.array([127999, 9000, 128000, 5, 64000, 100000])), (Batch, Pos))
ex = LmExample(tokens=tokens, loss_weight=hax.ones((Batch, Pos)).at[Pos, T - 1].set(0.0), attn_mask=AttentionMask.causal())

for side in ("in", "out", "both"):
    cfg = ComposedVocabQwen3Config(**common, cv_map=MAP, cv_input=side in ("in", "both"), cv_output=side in ("out", "both"))
    m = ComposedVocabQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
    # 2. composition, against numpy
    full = plain_with(m)
    if cfg.cv_input:
        ws = np.asarray(m.embeddings.token_embeddings.weight.rearrange(("small_vocab", "embed")).array, np.float64)
        got = np.asarray(full.embeddings.token_embeddings.weight.rearrange(("vocab", "embed")).array, np.float64)
        check(f"[{side}] input rows = mean of pieces", np.allclose(got, mean_rows(ws), atol=1e-5), f"max err {np.abs(got - mean_rows(ws)).max():.2e}")
    if cfg.cv_output:
        ws = np.asarray(m.lm_head.weight.rearrange(("small_vocab", "embed")).array, np.float64)
        got = np.asarray(m.get_lm_head().rearrange(("vocab", "embed")).array, np.float64)
        check(f"[{side}] output rows = mean of pieces", np.allclose(got, mean_rows(ws), atol=1e-5), f"max err {np.abs(got - mean_rows(ws)).max():.2e}")
    check(f"[{side}] Vocab axis", m.Vocab == Vocab)

    # 3. losses, logits and gradients against the plain model with the composed tables
    key = jrandom.PRNGKey(3)
    lc = m.compute_next_token_loss(ex, key=key)
    lc = lc[0] if isinstance(lc, tuple) else lc
    lp = full.compute_next_token_loss(ex, key=key)
    lp = lp[0] if isinstance(lp, tuple) else lp
    check(f"[{side}] train loss", np.allclose(float(lc.scalar() if hasattr(lc, 'scalar') else lc), float(lp.scalar() if hasattr(lp, 'scalar') else lp), atol=1e-5))
    ec = m.compute_next_token_loss(ex, key=None)
    ec = ec[0] if isinstance(ec, tuple) else ec
    check(f"[{side}] eval loss (key=None)", np.allclose(float(ec.scalar() if hasattr(ec, 'scalar') else ec), float(lp.scalar() if hasattr(lp, 'scalar') else lp), atol=1e-5))
    zc, zp = m(tokens, AttentionMask.causal()), full(tokens, AttentionMask.causal())
    check(f"[{side}] __call__ logits", np.allclose(np.asarray(zc.array), np.asarray(zp.array), atol=1e-4))

    def loss_of(model):
        out = model.compute_next_token_loss(ex, key=key)
        out = out[0] if isinstance(out, tuple) else out
        return out.scalar() if hasattr(out, "scalar") else out

    g_c = jax.grad(lambda mm: loss_of(mm))(m)
    g_p = jax.grad(lambda mm: loss_of(mm))(full)
    for which, small_g, full_g in (("input", g_c.embeddings.token_embeddings.weight if cfg.cv_input else None, g_p.embeddings.token_embeddings.weight),
                                   ("output", g_c.lm_head.weight if cfg.cv_output else None, g_p.lm_head.weight)):
        if small_g is None:
            continue
        gf = np.asarray(full_g.rearrange(("vocab", "embed")).array, np.float64)
        pull = np.zeros((n_small, D))
        np.add.at(pull, flat, (gf / counts[:, None])[rows])
        gs = np.asarray(small_g.rearrange(("small_vocab", "embed")).array, np.float64)
        check(f"[{side}] {which} gradient = pulled-back full gradient", np.allclose(gs, pull, atol=1e-6, rtol=1e-4), f"max err {np.abs(gs - pull).max():.2e}")
    # trunk gradients identical
    tg = jax.tree_util.tree_leaves(g_c.transformer)
    tp = jax.tree_util.tree_leaves(g_p.transformer)
    check(f"[{side}] trunk gradients", all(np.allclose(np.asarray(a), np.asarray(b), atol=1e-5) for a, b in zip(tg, tp)))

    # 4. MuonH labels
    mask = MuonHConfig(learning_rate=0.01, adam_lr=0.002).create_mask(m)
    check(f"[{side}] MuonH: input table adam", mask.embeddings.token_embeddings.weight == "adam", str(mask.embeddings.token_embeddings.weight))
    check(f"[{side}] MuonH: head adamh", mask.lm_head == "adamh" or getattr(mask.lm_head, "weight", None) == "adamh", str(mask.lm_head))
    q = mask.transformer.layers.stacked.mlp.gate_proj if hasattr(mask.transformer.layers, "stacked") else None
    check(f"[{side}] MuonH: trunk muonh", q is None or "muonh" in str(q), str(q)[:80])

    # 5. parameter counts
    n_in = m.embeddings.token_embeddings.weight.size
    n_out = m.lm_head.weight.size
    check(f"[{side}] table rows", (n_in == (K + 256) * D) == cfg.cv_input and (n_out == (K + 256) * D) == cfg.cv_output, f"input {n_in // D} rows, output {n_out // D} rows")

print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
