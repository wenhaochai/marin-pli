"""CPU tests for experiments.references.freeze_rows_muonh (vocabulary Q5b, owner 2026-10-12), on the full 128,256-token
vocabulary with a tiny trunk.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/fop_cpu_test.py
1. freeze_rows zeroes exactly rows K .. V-257 of lm_head.weight when frozen, changes nothing when not frozen, and leaves
   every other leaf alone; it refuses a K outside (0, V - 256).
2. The optimizer state has the baseline MuonH's structure, with learning_rate in its hyperparams (logging, checkpoints).
3. Real jitted MuonH steps on a fixed batch: before freeze_from_step every parameter equals the baseline optimizer's bit
   for bit; from it on, rows K .. V-257 of the head stay exactly at their values after step freeze_from_step - 1, while
   the rows below K, the specials and the trunk keep moving, and the loss keeps falling.
4. From the freeze on, AdamH's first moment of the frozen rows decays by beta1 per step (their gradient is zero).
5. The config refuses a negative freeze_from_step, and draccus encodes it as its registered choice.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.optim.config import OptimizerConfig
from levanter.optim.muonh import MuonHConfig

from experiments.references.freeze_rows_muonh import FreezeRowsMuonHConfig, freeze_rows

K, V, B, T, D = 8000, 128256, 2, 32, 48
Vocab, Batch, Pos = hax.Axis("vocab", V), hax.Axis("batch", B), hax.Axis("position", T)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


def head(m):
    return np.asarray(m.lm_head.weight.rearrange(("vocab", "embed")).array)


tok = jrandom.randint(jrandom.PRNGKey(1), (B, T), 0, V)
tok = tok.at[0, :8].set(jnp.array([5, 127999, 9000, 128000, 7, 64000, 100000, 3]))   # rare, common and special targets
ex = LmExample(tokens=hax.named(tok, (Batch, Pos)), loss_weight=hax.ones((Batch, Pos)), attn_mask=AttentionMask.causal())


def loss_of(model):
    out = model.compute_next_token_loss(ex, key=jrandom.PRNGKey(3))
    out = out[0] if isinstance(out, tuple) else out
    return out.scalar() if hasattr(out, "scalar") else out


model0 = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common), key=jrandom.PRNGKey(0))
P0 = eqx.filter(model0, eqx.is_inexact_array)

# ---- 1. freeze_rows ----
g = eqx.filter(eqx.filter_grad(loss_of)(model0), eqx.is_inexact_array)
gf, gn = freeze_rows(g, K, 256, jnp.bool_(True)), freeze_rows(g, K, 256, jnp.bool_(False))
h, hf = head(g), head(gf)
rare = np.zeros(V, bool); rare[K:V - 256] = True
check("[rows] frozen: rows K..V-257 zero", np.all(hf[rare] == 0) and np.any(h[rare] != 0))
check("[rows] frozen: rows below K and specials unchanged", np.array_equal(hf[~rare], h[~rare]))
check("[rows] frozen: other leaves unchanged", all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                                                   zip(jax.tree_util.tree_leaves(eqx.tree_at(lambda t: t.lm_head.weight, gf, None)),
                                                       jax.tree_util.tree_leaves(eqx.tree_at(lambda t: t.lm_head.weight, g, None)))))
check("[rows] not frozen: identity", all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree_util.tree_leaves(gn), jax.tree_util.tree_leaves(g))))
for bad in (0, V - 256):
    try:
        freeze_rows(g, bad, 256, jnp.bool_(True)); check(f"[rows] refuses K={bad}", False)
    except ValueError:
        check(f"[rows] refuses K={bad}", True)

# ---- 2.-4. optimizer ----
S, N = 3, 7
kw = dict(learning_rate=0.01, adam_lr=0.002, beta1=0.9, beta2=0.98, epsilon=1e-15, momentum=0.98, muon_epsilon=1e-5, weight_decay=0.1,
          warmup=0.0, lr_schedule="constant", coefficient_type="simple")
base_opt = MuonHConfig(**kw).build(100)
fop_opt = FreezeRowsMuonHConfig(**kw, freeze_lo=K, freeze_from_step=S).build(100)
sb, sf = base_opt.init(P0), fop_opt.init(P0)
check("[state] same structure as baseline MuonH", jax.tree_util.tree_structure(sb) == jax.tree_util.tree_structure(sf))
check("[state] learning_rate in hyperparams", "learning_rate" in sf.hyperparams, str(list(sf.hyperparams)))


def make_step(opt):
    @eqx.filter_jit
    def step(model, state):
        loss, grads = eqx.filter_value_and_grad(loss_of)(model)
        upd, state = opt.update(eqx.filter(grads, eqx.is_inexact_array), state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, upd), state, loss
    return step


def adamh_mu_head(state):
    """AdamH's first moment of lm_head.weight, [vocab, embed]."""
    found = []
    for leaf_path, leaf in jax.tree_util.tree_leaves_with_path(state):
        p = jax.tree_util.keystr(leaf_path)
        if ".mu" in p and "lm_head" in p and getattr(leaf, "shape", None) is not None and np.ndim(leaf) == 2 and leaf.shape[0] == V:
            found.append(np.asarray(leaf))
    if len(found) != 1:
        raise RuntimeError(f"found {len(found)} AdamH mu leaves for lm_head")
    return found[0]


bstep, fstep = make_step(base_opt), make_step(fop_opt)
mb, mf = model0, model0
losses, heads, mus = [], [], []
for i in range(N):
    mb, sb, _ = bstep(mb, sb)
    mf, sf, lf = fstep(mf, sf)
    losses.append(float(lf)); heads.append(head(mf))
    try:
        mus.append(adamh_mu_head(sf))
    except RuntimeError as e:
        mus.append(None); mu_err = str(e)
    if i == S - 1:
        same = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree_util.tree_leaves(mb), jax.tree_util.tree_leaves(mf)))
        check(f"[opt] after {S} updates (indices 0..{S - 1}): every parameter = baseline bit for bit", same)
hS = heads[S - 1]
check("[opt] frozen rows exactly at their values after update S-1 for every later update",
      all(np.array_equal(heads[i][rare], hS[rare]) for i in range(S, N)))
check("[opt] rows below K and specials keep moving", all(not np.array_equal(heads[i][~rare], heads[i - 1][~rare]) for i in range(S, N)))
trunk = lambda m: np.asarray(m.transformer.layers.stacked.mlp.down_proj.weight.array)  # noqa: E731
check("[opt] trunk differs from the baseline after the freeze", not np.array_equal(trunk(mb), trunk(mf)))
check("[opt] loss keeps falling after the freeze", losses[-1] < losses[S - 1], " ".join(f"{x:.4f}" for x in losses))
if all(m is not None for m in mus):
    r = [np.abs(mus[i][rare]).max() / max(np.abs(mus[i - 1][rare]).max(), 1e-30) for i in range(S, N)]
    check("[opt] AdamH mu of frozen rows decays by beta1 per update", np.allclose(r, 0.9, rtol=1e-5), " ".join(f"{x:.6f}" for x in r))
    check("[opt] AdamH mu of the other rows stays alive", np.abs(mus[-1][~rare]).max() > 0)
else:
    check("[opt] AdamH mu found in the state", False, mu_err)

# ---- 5. config ----
try:
    FreezeRowsMuonHConfig(**kw, freeze_from_step=-1).build(100); check("[cfg] refuses freeze_from_step < 0", False)
except ValueError:
    check("[cfg] refuses freeze_from_step < 0", True)
enc = draccus.encode(FreezeRowsMuonHConfig(**kw, freeze_lo=K, freeze_from_step=1430), OptimizerConfig)
check("[cfg] draccus encodes the registered choice", isinstance(enc, dict) and enc.get("type") == "muonH_freeze_rows" and enc.get("freeze_from_step") == 1430, str({k: enc.get(k) for k in ("type", "freeze_lo", "freeze_from_step")}))

print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}", flush=True)
