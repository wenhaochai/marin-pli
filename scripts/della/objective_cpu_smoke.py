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
# Document starts must never be corrupted: samp[t] there is drawn from h_{t-1}, which belongs to the PREVIOUS
# document, so replacing it splices in an out-of-context token that the discriminator can spot for free.
starts = np.zeros_like(rp, dtype=bool)
starts[:, 0] = True
starts[:, 1:] = seg[:, 1:] != seg[:, :-1]
assert starts.sum() == 4, f"fixture must contain 4 document starts, has {starts.sum()}"
assert not rp[starts].any(), "a document-start position was corrupted"
rp_any = np.zeros_like(rp, dtype=bool)
for _i in range(200):
    rp_any |= np.asarray(corrupt_jit(ebm_model, h_, ex, seg_named, jax.random.PRNGKey(_i))) > 0
assert not rp_any[starts].any(), "a document-start position was corrupted under some key"
print(f"ebm boundary protection: {int(starts.sum())} document starts, none corrupted over 200 keys  OK")
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

def scored_key(model, ex, ev, start=1, tries=16):
    """First PRNGKey >= start under which the corruption scores at least one position (train loss != pure NTP).
    Document starts, EOS and same-token draws are protected, so on this 2x16 fixture a small rho draw can leave
    nothing to score; that key is a legitimate 'loss == NTP' case (checked below), not a usable gradient probe."""
    for k in range(start, start + tries):
        if abs(float(model.compute_next_token_loss(ex, key=jax.random.PRNGKey(k))) - ev) > 1e-7:
            return k
    raise AssertionError(f"no key in [{start}, {start + tries}) scored any position")


for name, kw in [("twin", dict(twin=True)), ("sr", dict(sr=True)), ("twinsr", dict(twin=True, sr=True)), ("pi", dict(pi=True, pi_negatives=8)), ("eos", dict(eos=True, eos_id=EOS)), ("ebm", dict(ebm=True, eos_id=EOS, ebm_blocks=7))]:
    cfg = ObjectiveQwen3Config(**common, **kw)
    model = cfg.model_type.init(Vocab, cfg, key=key)
    ev = float(model.compute_next_token_loss(ex))
    tk = scored_key(model, ex, ev) if (cfg.ebm or cfg.denoise) else 1
    tr = float(model.compute_next_token_loss(ex, key=jax.random.PRNGKey(tk)))
    assert np.isfinite(tr), f"{name}: train loss not finite"
    assert abs(ev - ref_eval) < 1e-5, f"{name}: eval loss {ev} != plain {ref_eval}"

    def loss_fn(m):
        return m.compute_next_token_loss(ex, key=jax.random.PRNGKey(tk)).scalar()

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
        # A key that scores no position must give EXACTLY the NTP loss: the auxiliary contributes 0, not NaN.
        zk = next((k for k in range(0, 32) if abs(float(model.compute_next_token_loss(ex, key=jax.random.PRNGKey(k))) - ev) <= 1e-7), None)
        if zk is not None:
            print(f"{name:7s} key {zk} scores no position and train loss == NTP exactly; gradient probe uses key {tk}  OK")
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

# aux_layer: the auxiliary heads read an intermediate residual stream; eval must stay identical and the top must be NTP's.
for name, kw in [("sr", dict(sr=True)), ("eos", dict(eos=True, eos_id=EOS)), ("ebm", dict(ebm=True, eos_id=EOS, ebm_blocks=7)), ("twin", dict(twin=True))]:
    for k in (1, 2):
        cfg = ObjectiveQwen3Config(**common, aux_layer=k, **kw)
        model = cfg.model_type.init(Vocab, cfg, key=key)
        ev = float(model.compute_next_token_loss(ex))
        assert abs(ev - ref_eval) < 1e-5, f"{name} L{k}: eval {ev} != plain {ref_eval}"
        tk = scored_key(model, ex, ev) if (cfg.ebm or cfg.denoise) else 1
        h, h_aux = model.forward_with_aux(tokens, mask)
        h_final = model.activations(tokens, mask)
        assert float(hax.max(hax.abs(h - h_final))) < 1e-4, f"{name} L{k}: forward_with_aux h differs from activations"
        rms = np.asarray(hax.sqrt(hax.mean(h_aux * h_aux, axis="embed")).array)
        assert np.allclose(rms, 1.0, atol=1e-3), f"{name} L{k}: h_aux not unit RMS"
        if k == 2:  # last layer: h = RmsNorm(x_L) = unit(x_L) * gain, so h_aux * gain must reproduce h (up to eps)
            gain = model.transformer.norm.weight
            assert float(hax.max(hax.abs(h_aux * gain - h.astype(jnp.float32)))) < 1e-3, f"{name} L2: h_aux*gain != h"
        tr = float(model.compute_next_token_loss(ex, key=jax.random.PRNGKey(tk)))
        grads = eqx.filter_grad(lambda m: m.compute_next_token_loss(ex, key=jax.random.PRNGKey(tk)).scalar())(model)
        leaves = [(jax.tree_util.keystr(p), g) for p, g in jax.tree_util.tree_leaves_with_path(grads) if g is not None]
        assert all(bool(jnp.all(jnp.isfinite(g))) for _, g in leaves), f"{name} L{k}: non-finite grads"
        head_g = sum(float(jnp.linalg.norm(g.astype(jnp.float32))) for n, g in leaves if any(t in n for t in ("sr_head", "eos_head", "ebm_head", "twin_proj")))
        assert head_g > 0, f"{name} L{k}: head got no gradient"
        print(f"{name:5s} aux_layer={k}: eval == plain, h_aux unit-RMS, train {tr:.4f}, head grad {head_g:.3e}")
# L1 and L2 readouts must differ (otherwise the index is ignored)
c1 = ObjectiveQwen3Config(**common, aux_layer=1, eos=True, eos_id=EOS)
c2 = ObjectiveQwen3Config(**common, aux_layer=2, eos=True, eos_id=EOS)
d = float(hax.max(hax.abs(c1.model_type.init(Vocab, c1, key=key).forward_with_aux(tokens, mask)[1] - c2.model_type.init(Vocab, c2, key=key).forward_with_aux(tokens, mask)[1])))
assert d > 1e-3, f"aux_layer 1 and 2 readouts identical ({d})"
print(f"aux_layer readouts differ between layers (max |diff| {d:.3f})")

# denoise: NTP on the corrupted copy with clean targets; alone (dn), with the energy head (ebm+dn), and ebm at T=2.
for name, kw in [("dn", dict(denoise=True, eos_id=EOS, ebm_blocks=7)), ("ebm+dn", dict(ebm=True, denoise=True, eos_id=EOS, ebm_blocks=7)), ("ebmT2", dict(ebm=True, eos_id=EOS, ebm_blocks=7, ebm_temp=2.0))]:
    cfg = ObjectiveQwen3Config(**common, **kw)
    model = cfg.model_type.init(Vocab, cfg, key=key)
    ev = float(model.compute_next_token_loss(ex))
    assert abs(ev - ref_eval) < 1e-5, f"{name}: eval {ev} != plain {ref_eval}"
    tk = scored_key(model, ex, ev) if (cfg.ebm or cfg.denoise) else 1
    tr = float(model.compute_next_token_loss(ex, key=jax.random.PRNGKey(tk)))
    grads = eqx.filter_grad(lambda m: m.compute_next_token_loss(ex, key=jax.random.PRNGKey(tk)).scalar())(model)
    leaves = [(jax.tree_util.keystr(p), g) for p, g in jax.tree_util.tree_leaves_with_path(grads) if g is not None]
    assert all(bool(jnp.all(jnp.isfinite(g))) for _, g in leaves), f"{name}: non-finite grads"
    if name == "dn":
        aux = (tr - ref_eval) / 0.1  # denoising NTP on a corrupted prefix at init ~ the plain NTP loss (random model)
        assert 0.7 * ref_eval < aux < 1.5 * ref_eval, f"dn aux {aux:.3f} vs NTP {ref_eval:.3f}"
        assert not any("ebm_head" in n for n, _ in leaves), "dn must not create an ebm head"
    if name == "ebm+dn":
        assert any("ebm_head" in n for n, _ in leaves)
        aux = tr - ref_eval  # 0.1*NCE + 0.1*denoise-NTP
        assert 0.1 * 0.5 * 2 * np.log(2) + 0.1 * 0.7 * ref_eval < aux < 0.1 * 1.5 * 2 * np.log(2) + 0.1 * 1.5 * ref_eval, f"ebm+dn aux {aux:.3f}"
    print(f"{name:7s} eval == plain, train {tr:.4f} (NTP {ref_eval:.4f}), flops x{cfg.flops_per_token(V, T) / Qwen3Config(**common).flops_per_token(V, T):.1f}  OK")
# temperature changes the samples: T=2 draws must differ from T=1 draws for the same key on the same states
c1 = ObjectiveQwen3Config(**common, ebm=True, eos_id=EOS, ebm_blocks=7); c2 = ObjectiveQwen3Config(**common, ebm=True, eos_id=EOS, ebm_blocks=7, ebm_temp=2.0)
m1 = c1.model_type.init(Vocab, c1, key=key); m2 = c2.model_type.init(Vocab, c2, key=key)
hh = m1.activations(tokens, mask)
d1 = np.asarray(m1._sample_next(hh, key=jax.random.PRNGKey(5)).array); d2 = np.asarray(m2._sample_next(hh, key=jax.random.PRNGKey(5)).array)
assert (d1 != d2).any(), "T=2 sampler identical to T=1"
print(f"ebm_temp=2 changes {int((d1 != d2).sum())}/{d1.size} draws for the same key  OK")

# mtp: x_{t+k} through a D x D projection and the shared lm_head; target/mask bookkeeping checked against a brute-force CE.
for k in (2, 3):
    cfg = ObjectiveQwen3Config(**common, mtp=True, mtp_k=k, eos_id=EOS)
    model = cfg.model_type.init(Vocab, cfg, key=key)
    ev = float(model.compute_next_token_loss(ex))
    assert abs(ev - ref_eval) < 1e-5, f"mtp k={k}: eval {ev} != plain {ref_eval}"
    tr = float(model.compute_next_token_loss(ex, key=jax.random.PRNGKey(1)))
    # brute force: CE(lm_head(proj(h_t)), x_{t+k}) over positions with t+k in the window, same document, loss weight
    hh = model.activations(tokens, mask)
    pred = model.mtp_proj(hh).rename({"mtp_embed": "embed"})
    logits = np.asarray(hax.dot(pred, model.get_lm_head(), axis="embed").astype(jnp.float32).array)  # (B, T, V)
    lp = logits - logits.max(-1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    num = 0.0; den = 0
    for b in range(B):
        for t in range(T):
            if t + k <= T - 1 and lw[b, t] > 0 and seg[b, t] == seg[b, t + k]:
                num += -lp[b, t, tok[b, t + k]]; den += 1
    brute = num / den
    aux = (tr - ref_eval) / 0.1
    assert abs(aux - brute) < 1e-3, f"mtp k={k}: aux {aux:.5f} vs brute-force {brute:.5f} ({den} positions)"
    grads = eqx.filter_grad(lambda m: m.compute_next_token_loss(ex, key=jax.random.PRNGKey(1)).scalar())(model)
    leaves = [(jax.tree_util.keystr(p), g) for p, g in jax.tree_util.tree_leaves_with_path(grads) if g is not None]
    assert all(bool(jnp.all(jnp.isfinite(g))) for _, g in leaves)
    assert sum(float(jnp.linalg.norm(g.astype(jnp.float32))) for n, g in leaves if "mtp_proj" in n) > 0
    print(f"mtp k={k}: eval == plain, aux CE {aux:.4f} == brute force {brute:.4f} over {den} positions, grads finite, mtp_proj trained  OK")

# aux_gate: the auxiliary weight is scaled by clip((L_ntp - lo)/(hi - lo), 0, 1) computed from the batch's own NTP loss.
for name, kw in [("sr", dict(sr=True)), ("eos", dict(eos=True, eos_id=EOS))]:
    def train_loss(**extra):
        cfg = ObjectiveQwen3Config(**common, **kw, **extra)
        m = cfg.model_type.init(Vocab, cfg, key=key)
        return float(m.compute_next_token_loss(ex, key=jax.random.PRNGKey(1))), float(m.compute_next_token_loss(ex))
    tr0, ev0 = train_loss()
    aux0 = tr0 - ev0
    tr_on, _ = train_loss(aux_gate_hi=1.0, aux_gate_lo=0.5)  # L ~ 4.1 >> hi -> gate 1
    tr_off, _ = train_loss(aux_gate_hi=100.0, aux_gate_lo=99.0)  # L << lo -> gate 0
    tr_half, _ = train_loss(aux_gate_hi=ev0 + 0.05, aux_gate_lo=ev0 - 0.05)  # gate exactly 0.5
    assert abs(tr_on - tr0) < 1e-5, f"{name}: gate=1 should equal ungated ({tr_on} vs {tr0})"
    assert abs(tr_off - ev0) < 1e-5, f"{name}: gate=0 should equal the plain NTP loss ({tr_off} vs {ev0})"
    assert abs(tr_half - (ev0 + 0.5 * aux0)) < 1e-4, f"{name}: gate=0.5 should give NTP + aux/2 ({tr_half} vs {ev0 + 0.5 * aux0})"
    print(f"{name:4s} aux_gate: gate 1 -> {tr_on:.4f} (= ungated), gate 0 -> {tr_off:.4f} (= NTP {ev0:.4f}), gate 0.5 -> {tr_half:.4f} (= NTP + aux/2)  OK")
# the gate must not leak gradient through the NTP loss (stop_gradient): grads with gate 0.5 == 0.5 * aux grads + ntp grads
cfg_g = ObjectiveQwen3Config(**common, eos=True, eos_id=EOS, aux_gate_hi=ref_eval + 0.05, aux_gate_lo=ref_eval - 0.05)
m_g = cfg_g.model_type.init(Vocab, cfg_g, key=key)
g_gate = eqx.filter_grad(lambda m: m.compute_next_token_loss(ex, key=jax.random.PRNGKey(1)).scalar())(m_g)
g_head = [g for p, g in jax.tree_util.tree_leaves_with_path(g_gate) if g is not None and "eos_head" in jax.tree_util.keystr(p) and "weight" in jax.tree_util.keystr(p)][0]
cfg_u = ObjectiveQwen3Config(**common, eos=True, eos_id=EOS)
m_u = cfg_u.model_type.init(Vocab, cfg_u, key=key)
g_un = eqx.filter_grad(lambda m: m.compute_next_token_loss(ex, key=jax.random.PRNGKey(1)).scalar())(m_u)
g_head_u = [g for p, g in jax.tree_util.tree_leaves_with_path(g_un) if g is not None and "eos_head" in jax.tree_util.keystr(p) and "weight" in jax.tree_util.keystr(p)][0]
assert float(jnp.max(jnp.abs(g_head - 0.5 * g_head_u))) < 1e-6, "gate=0.5 head gradient is not half the ungated one (gate leaks gradient?)"
print("aux_gate: head gradient at gate 0.5 is exactly half the ungated gradient (stop_gradient holds)")
print("CPU SMOKE PASSED")
