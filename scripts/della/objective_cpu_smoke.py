"""CPU smoke for experiments.references.objective_qwen3: shapes, finiteness, eval parity with plain Qwen3, gradients.

Run from the marin repo: JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python scripts/della/objective_cpu_smoke.py (about 3 min on a login node). Tiny model; a synthetic batch with an EOS inside each row so the
document-boundary masks are exercised.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import haliax as hax
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel

from experiments.references.objective_qwen3 import ObjectiveQwen3Config

B, T, V, EOS = 2, 16, 50, 1
common = dict(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA)
Vocab = hax.Axis("vocab", V)
key = jax.random.PRNGKey(0)

rng = np.random.default_rng(0)
tok = rng.integers(2, V, size=(B, T))
tok[0, 6] = EOS  # row 0: two documents, boundary after position 6
tok[1, 11] = EOS  # row 1: boundary after position 11
eos_mask = np.roll(tok, 1, axis=1) == EOS
eos_mask[:, 0] = False
seg = np.cumsum(eos_mask.astype(np.int32), axis=1)
lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)
Batch, Pos = hax.Axis("batch", B), hax.Axis("position", T)
tokens = hax.named(jnp.asarray(tok), (Batch, Pos))
weight = hax.named(jnp.asarray(lw), (Batch, Pos))
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos)))
ex = LmExample(tokens=tokens, loss_weight=weight, attn_mask=mask)

plain = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common), key=key)
ref_eval = float(plain.compute_next_token_loss(ex))
print(f"plain qwen3 eval loss {ref_eval:.6f}")

# Exact check of the eos targets on the synthetic rows (EOS at row0 pos 6, row1 pos 11; T=16).
from experiments.references.objective_qwen3 import ObjectiveQwen3LMHeadModel
bins, valid = ObjectiveQwen3LMHeadModel.eos_targets(tokens, weight, EOS, 13)
b, v = np.asarray(bins.array), np.asarray(valid.array)
exp_bins0 = [np.ceil(np.log2(6 - t)) if t < 6 else -1 for t in range(T)]
for t in range(T):
    if t < 6:
        assert v[0, t] == 1 and b[0, t] == exp_bins0[t], (t, v[0, t], b[0, t], exp_bins0[t])
    else:
        assert v[0, t] == 0, (t, v[0, t])  # t=6 is EOS itself; t>6 has no later EOS in the window
assert v[1, 3] == 1 and b[1, 3] == 3 and v[1, 10] == 1 and b[1, 10] == 0 and v[1, 11] == 0 and v[1, 12] == 0
print("eos targets: row0 valid", int(v[0].sum()), "row1 valid", int(v[1].sum()), "-- exact checks passed")

# Optimizer-group check: the whole point of free_heads is where MuonH's mask puts the auxiliary heads.
from levanter.optim.muonh import MuonHConfig
import jax.tree_util as jtu

def labels_for(cfg):
    model = cfg.model_type.init(Vocab, cfg, key=key)
    params = eqx.filter(model, eqx.is_inexact_array)
    mask = MuonHConfig().create_mask(params)
    out = {}
    for (p, lab), (q, _) in zip(jtu.tree_leaves_with_path(mask), jtu.tree_leaves_with_path(params)):
        out[jtu.keystr(p)] = lab
    return out

# ebm: sampler exactness (Gumbel-max over vocab blocks == categorical), corruption bookkeeping, NCE at init.
ebm_cfg = ObjectiveQwen3Config(**common, ebm=True, eos_id=EOS, ebm_blocks=7)  # 7 does not divide V=50 -> exercises padding
ebm_model = ebm_cfg.model_type.init(Vocab, ebm_cfg, key=key)
h_ = ebm_model.activations(tokens, mask)
logits_ = np.asarray(hax.dot(h_, ebm_model.get_lm_head(), axis="embed").astype(jnp.float32).array)  # (B, T, V)
p_ref = np.exp(logits_ - logits_.max(-1, keepdims=True)); p_ref /= p_ref.sum(-1, keepdims=True)
sample_jit = eqx.filter_jit(lambda m, h, k: jax.vmap(lambda kk: m._sample_next(h, key=kk).array)(k))
draws = np.asarray(sample_jit(ebm_model, h_, jax.random.split(jax.random.PRNGKey(100), 4000)))  # (n, B, T)
corrupt_jit = eqx.filter_jit(lambda m, h, e, s, k: m._ebm_corrupt(h, e, s, key=k)[1].array)
assert draws.min() >= 0 and draws.max() < V
emp = np.stack([(draws == v).mean(0) for v in range(V)], -1)  # (B, T, V)
tv = 0.5 * np.abs(emp - p_ref).sum(-1)
print(f"ebm sampler: max total-variation to softmax over {B*T} positions = {tv.max():.4f} (4000 draws; expect ~<0.05)")
assert tv.max() < 0.06, tv.max()
seg_named = hax.named(jnp.asarray(seg), (Batch, Pos))
x_noisy, replaced, informative = ebm_model._ebm_corrupt(h_, ex, seg_named, key=jax.random.PRNGKey(7))
xn, rp, inf_ = np.asarray(x_noisy.array), np.asarray(replaced.array), np.asarray(informative.array)
assert not rp[:, 0].any() and not rp[tok == EOS].any() and not (xn == EOS)[tok != EOS].any(), "protected positions were corrupted"
assert np.array_equal(xn != tok, rp), "replaced must mark exactly the changed tokens"
# informative = loss-carrying position with >= 1 replacement at or before it inside the same document
exp_inf = np.zeros_like(inf_)
for b in range(B):
    for t in range(T):
        same = seg[b] == seg[b, t]
        exp_inf[b, t] = float(rp[b, : t + 1][same[: t + 1]].any() and lw[b, t] > 0)
assert np.array_equal(inf_, exp_inf), (inf_, exp_inf)
rates = [np.asarray(corrupt_jit(ebm_model, h_, ex, seg_named, jax.random.PRNGKey(i))).mean() for i in range(200)]
print(f"ebm corruption: replaced {rp.sum()} / {B*T}, informative {int(inf_.sum())}; mean replaced rate over 200 draws {np.mean(rates):.3f} (rho ~ U(0, 0.5), minus protected)")
assert 0.1 < np.mean(rates) < 0.3

for free in (True, False):
    lab = labels_for(ObjectiveQwen3Config(**common, twin=True, sr=True, pi=True, eos=True, ebm=True, eos_id=EOS, free_heads=free))
    heads = {k: v for k, v in lab.items() if any(h in k for h in ("twin_proj", "sr_head", "pi_proj", "eos_head", "ebm_head")) and "backward" not in k}
    trunk = {k: v for k, v in lab.items() if ".transformer." in k and "weight" in k and "norm" not in k and "backward" not in k}
    lmh = {k: v for k, v in lab.items() if k.endswith("lm_head.weight") and "backward" not in k}
    exp_w = "adam" if free else "muonh"
    for k, v in heads.items():
        want = "adam" if (free or k.endswith(".bias")) else "muonh"
        assert v == want, f"free_heads={free}: {k} labelled {v}, expected {want}"
    assert all(v == "muonh" for v in trunk.values()), {k: v for k, v in trunk.items() if v != "muonh"}
    assert all(v == "adamh" for v in lmh.values()), lmh
    print(f"free_heads={free!s:5s}: heads -> {sorted(set(heads.values()))} ({len(heads)} leaves), trunk linears -> muonh ({len(trunk)}), lm_head -> adamh  OK")

for name, kw in [("twin", dict(twin=True)), ("sr", dict(sr=True)), ("twinsr", dict(twin=True, sr=True)), ("pi", dict(pi=True, pi_negatives=8)), ("eos", dict(eos=True, eos_id=EOS)), ("ebm", dict(ebm=True, eos_id=EOS, ebm_blocks=7))]:
    cfg = ObjectiveQwen3Config(**common, **kw)
    model = cfg.model_type.init(Vocab, cfg, key=key)
    ev = float(model.compute_next_token_loss(ex))
    tr = float(model.compute_next_token_loss(ex, key=jax.random.PRNGKey(1)))
    assert np.isfinite(tr), f"{name}: train loss not finite"
    assert abs(ev - ref_eval) < 1e-5, f"{name}: eval loss {ev} != plain {ref_eval}"

    def loss_fn(m):
        return m.compute_next_token_loss(ex, key=jax.random.PRNGKey(1)).scalar()

    grads = eqx.filter_grad(loss_fn)(model)
    leaves = [(p, g) for p, g in jax.tree_util.tree_leaves_with_path(grads) if g is not None]
    bad = [jax.tree_util.keystr(p) for p, g in leaves if not bool(jnp.all(jnp.isfinite(g)))]
    assert not bad, f"{name}: non-finite grads at {bad[:5]}"
    norms = {jax.tree_util.keystr(p): float(jnp.linalg.norm(g.astype(jnp.float32))) for p, g in leaves}
    def nz(prefix):
        return sum(v for k, v in norms.items() if prefix in k)
    print(f"{name:7s} eval {ev:.6f} (== plain) train {tr:.4f} | grad norms: forward-transformer {nz('.transformer'):.3e} "
          f"twin_proj {nz('twin_proj'):.3e} sr_head {nz('sr_head'):.3e} pi_proj {nz('pi_proj'):.3e} eos_head {nz('eos_head'):.3e} ebm_head {nz('ebm_head'):.3e} backward {nz('backward'):.3e}")
    if kw.get("ebm"):
        assert nz("ebm_head") > 0
        aux = (tr - ref_eval) / 0.1
        assert 0.5 * 2 * np.log(2) < aux < 1.5 * 2 * np.log(2), f"ebm aux {aux:.3f} vs 2 log 2 = {2*np.log(2):.3f} at init"
    if kw.get("twin"):
        assert nz("twin_proj") > 0 and nz("backward") > 0
    if kw.get("sr"):
        assert nz("sr_head") > 0
    if kw.get("pi"):
        assert nz("pi_proj") > 0
    if kw.get("eos"):
        assert nz("eos_head") > 0
        aux = (tr - ref_eval) / 0.1
        assert 0.3 * np.log(13) < aux < 3.0 * np.log(13), f"eos aux {aux:.3f} vs log(13)={np.log(13):.3f}"
        # InfoNCE with n negatives sits near log(n+1) at init; well below means a leak, far above means a bug.
        aux = tr - ref_eval
        assert 0.3 * np.log(9) < aux / 0.1 < 3.0 * np.log(9), f"pi aux {aux / 0.1:.3f} vs log(9)={np.log(9):.3f}"
    print(f"{name:7s} flops_per_token x{cfg.flops_per_token(V, T) / Qwen3Config(**common).flops_per_token(V, T):.1f}")
print("CPU SMOKE PASSED")
