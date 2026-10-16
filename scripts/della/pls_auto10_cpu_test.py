"""CPU tests for the ten more attempts at the last-layer cost (per_layer_qwen3.PerLayerQwen3Config pls_mix_scale ...
pls_random_layer; MixAuxMuonHConfig), owner popup 2026-10-16.

    WT=<worktree>; PYTHONPATH=$WT:$WT/lib/*/src JAX_PLATFORMS=cpu nice -n 19 <venv>/bin/python scripts/della/pls_auto10_cpu_test.py
0. Validation: two attempts at once, or one without separate heads, raise.
1. Init: with any attempt on, every baseline and separate-head parameter is byte-identical to separate heads alone.
2. Logged per-layer CE: for every attempt but the adapter, pls/L{k} equals separate heads' on the same parameters.
3. Loss formulas against dense references (full logits): conf gate (sum m p ce / sum m), first-n positions (mean over the
   kept positions, segment ids), distillation (soft CE against the final softmax, chunked and unchunked), random layer
   (one layer at pls_weight, the drawn index varies with the key).
4. Gradient routing: mix -> the blocks' Linear weights get the final loss's gradient alone (= the probes model's), the
   ghost gets the readouts' part (separate heads minus probes), everything else separate heads'; MLP-only -> attention
   gets the probes model's gradient, the MLPs separate heads'; random layer -> the other heads get exactly zero.
5. Adapter: the per-layer readout (readout_losses, the eval path) goes through it; the last layer is unchanged.
6. MixAuxMuonHConfig: one update equals a hand-written reference (own clipping, momentum, Newton-Schulz, O + s * O_ghost,
   norm-preserving step); the ghost's update is zero; the other groups' updates equal MuonHConfig's.
7. Training: 5 steps of each attempt (mix with MixAuxMuonHConfig) stay finite; the ghost stays exactly zero.
"""
import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import haliax as hax  # noqa: E402
import haliax.nn as hnn  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402
from levanter.optim.util import norm_preserving_update, zeropower_via_newtonschulz5  # noqa: E402

from experiments.references.per_layer_qwen3 import MixAuxMuonHConfig, PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

B, T, V, L = 4, 16, 64, 4
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
perm = np.random.default_rng(0).permutation(V)
FAIL = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def batch(i, segments=False):
    r = np.random.default_rng(1000 + i)
    tok = np.zeros((B, T), np.int64)
    tok[:, 0] = r.integers(0, V, B)
    for t in range(1, T):
        tok[:, t] = np.where(r.random(B) < 0.2, r.integers(0, V, B), perm[tok[:, t - 1]])
    lw = np.ones((B, T), np.float32)
    lw[:, 5] = 0.5
    mask = AttentionMask.causal()
    seg = None
    if segments:
        seg = np.cumsum(r.random((B, T)) < 0.25, axis=1).astype(np.int32)
        mask = mask.with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos)))
    ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)), attn_mask=mask)
    return ex, tok, lw, seg


def cfg(**kw):
    return PerLayerQwen3Config(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True,
                               use_qk_norm=False, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True, **kw)


ATTEMPTS = dict(mix=dict(pls_mix_scale=0.2), kd=dict(pls_distill=True), conf=dict(pls_conf_gate=True), ad=dict(pls_adapter_dim=16),
                mlpo=dict(pls_aux_mlp_only=True), first=dict(pls_aux_first=5), randl=dict(pls_random_layer=True))
key0 = jrandom.PRNGKey(0)
sep = PerLayerQwen3LMHeadModel.init(Vocab, cfg(), key=key0)
probes = PerLayerQwen3LMHeadModel.init(Vocab, cfg(pls_detach_backbone=True), key=key0)
M = {n: PerLayerQwen3LMHeadModel.init(Vocab, cfg(**kw), key=key0) for n, kw in ATTEMPTS.items()}
KEY = jrandom.PRNGKey(7)


def leaves(t):
    return jax.tree.leaves(eqx.filter(t, eqx.is_array))


def maxdiff(a, b):
    la, lb = leaves(a), leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    return max(float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)))) for x, y in zip(la, lb))


def base(m):
    return (m.embeddings, m.transformer, m.lm_head, m.aux_norms, m.aux_lm_heads)


def stat(v):
    for name in ("value", "item"):
        if hasattr(v, name) and callable(getattr(v, name)):
            try:
                return float(getattr(v, name)())
            except Exception:
                pass
    return float(v)


def run(m, ex, key=KEY):
    loss, stats = m.compute_next_token_loss(ex, key=key, logsumexp_weight=0.0)
    return float(loss.array if isinstance(loss, hax.NamedArray) else loss), {k: stat(v) for k, v in stats.items()}


def grads(m, ex, key=KEY):
    return eqx.filter_grad(lambda mm: mm.compute_next_token_loss(ex, key=key, logsumexp_weight=0.0)[0].array)(m)


# 0. validation
for bad, kw in (("two attempts", dict(pls_distill=True, pls_conf_gate=True)), ("no separate heads", dict(pls_distill=True))):
    try:
        if bad == "no separate heads":
            PerLayerQwen3Config(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True, use_qk_norm=False, pls_weight=1.0, **kw)
        else:
            cfg(**kw)
        check(False, f"0. {bad} raises")
    except ValueError:
        check(True, f"0. {bad} raises")

# 1. init
for n, m in M.items():
    check(maxdiff(base(sep), base(m)) == 0.0, f"1. {n}: baseline and separate-head parameters byte-identical")
check(all(float(jnp.max(jnp.abs(x))) == 0.0 for x in leaves(M["mix"].aux_ghost)) and len(leaves(M["mix"].aux_ghost)) == len(leaves(sep.transformer.layers)), "1. mix: the ghost is all zeros, one leaf per block leaf")
check(M["ad"].aux_adapters is not None and len(M["ad"].aux_adapters) == L - 1 and sep.aux_adapters is None and sep.aux_ghost is None, "1. adapters only when on")


def same_params(m):  # the attempt's model: its shared parameters are separate heads' already (test 1); the model's config
    # lives on its transformer, so swapping modules in would also swap the config
    return m


# 2. logged per-layer CE
for segs in (False, True):
    ex, *_ = batch(0, segs)
    _, s0 = run(sep, ex)
    for n, m in M.items():
        if n == "ad":
            continue
        _, s1 = run(same_params(m), ex)
        d = max(abs(s0[f"pls/L{k}"] - s1[f"pls/L{k}"]) for k in range(L))
        check(d < 2e-5, f"2. segments={segs} {n}: pls/L0..L{L - 1} equal separate heads' (max diff {d:.2e})")


# 3. dense references
def readout_logits(m, ex, k):
    outs = m.layer_outputs(ex.tokens, ex.attn_mask, key=KEY)
    h = m._layer(outs, k).rearrange((Batch, Pos, m.Embed)).array.astype(jnp.float32)
    W = m._head(k).rearrange((m.Vocab, m.Embed)).array.astype(jnp.float32)
    return jnp.einsum("bte,ve->btv", h, W)


def masks(lw):
    m = lw.copy()
    m[:, T - 1] = 0.0  # next_token_loss_weight: the last position has no target
    return m


for segs in (False, True):
    ex, tok, lw, seg = batch(1, segs)
    mm = masks(lw)
    y = np.roll(tok, -1, axis=1)
    lf = readout_logits(sep, ex, L - 1)
    ce_f = -np.take_along_axis(np.asarray(jax.nn.log_softmax(lf, -1)), y[..., None], -1)[..., 0]
    loss_sep, s_sep = run(sep, ex)
    final = s_sep[f"pls/L{L - 1}"]
    ces = [-np.take_along_axis(np.asarray(jax.nn.log_softmax(readout_logits(sep, ex, k), -1)), y[..., None], -1)[..., 0] for k in range(L - 1)]
    # conf
    p = np.exp(-ce_f)
    want = final + sum(float(np.sum(mm * p * c) / np.sum(mm)) for c in ces)
    got, _ = run(same_params(M["conf"]), ex)
    check(abs(got - want) < 1e-4, f"3. segments={segs} conf: loss {got:.5f} = final + sum_k sum m p ce / sum m ({want:.5f})")
    # first n positions of every document
    n = 5
    if seg is None:
        rel = np.tile(np.arange(T), (B, 1))
    else:
        rel = np.zeros((B, T), int)
        for b in range(B):
            for t in range(1, T):
                rel[b, t] = 0 if seg[b, t] != seg[b, t - 1] else rel[b, t - 1] + 1
    keep = (rel < n).astype(np.float32)
    got_keep = np.asarray(M["first"]._first_positions(ex, n).rearrange((Batch, Pos)).array)
    check(np.array_equal(got_keep, keep), f"3. segments={segs} first: kept positions match brute force ({int(keep.sum())} of {B * T})")
    want = final + sum(float(np.sum(mm * keep * c) / np.sum(mm * keep)) for c in ces)
    got, _ = run(same_params(M["first"]), ex)
    check(abs(got - want) < 1e-4, f"3. segments={segs} first: loss {got:.5f} = final + sum_k mean over kept ({want:.5f})")
    # distillation
    pf = np.asarray(jax.nn.softmax(lf, -1))
    soft = [float(np.sum(mm * np.sum(pf * -np.asarray(jax.nn.log_softmax(readout_logits(sep, ex, k), -1)), -1)) / np.sum(mm)) for k in range(L - 1)]
    got, s_kd = run(same_params(M["kd"]), ex)
    check(abs(got - (final + sum(soft))) < 1e-4, f"3. segments={segs} kd: loss {got:.5f} = final + sum_k soft CE ({final + sum(soft):.5f})")
    m_kd = same_params(M["kd"])
    outs = m_kd.layer_outputs(ex.tokens, ex.attn_mask, key=KEY)
    for chunk in (4, 8, 64):
        sc, lc = m_kd._distill_all(outs, outs, ex, logit_soft_cap=None, chunk=chunk)
        check(float(jnp.max(jnp.abs(sc - jnp.asarray(soft)))) < 1e-4 and abs(float(lc[0]) - float(np.sum(mm * ces[0]) / np.sum(mm))) < 1e-4,
              f"3. segments={segs} kd chunk {chunk}: soft and label CE match the dense reference")
    # random layer
    m_rl = same_params(M["randl"])
    drawn = set()
    for kk in range(6):
        key = jrandom.PRNGKey(100 + kk)
        got, st = run(m_rl, ex, key)
        k = int(st["pls/layer"])
        drawn.add(k)
        check(abs(got - (final + st[f"pls/L{k}"])) < 1e-4, f"3. segments={segs} randl key {kk}: loss = final + CE of drawn layer {k}")
    check(len(drawn) > 1, f"3. segments={segs} randl: the drawn layer varies with the key ({sorted(drawn)})")


# 4. gradient routing
ex, *_ = batch(2, True)
g_sep, g_probe = grads(sep, ex), grads(probes, ex)
g_mix = grads(same_params(M["mix"]), ex)


def linear_weights(stk):
    return [l.weight.array for l in jax.tree.leaves(stk, is_leaf=lambda x: isinstance(x, hnn.Linear)) if isinstance(l, hnn.Linear)]


def other_leaves(stk):
    return [x for x in jax.tree.leaves(jax.tree.map(lambda l: None if isinstance(l, hnn.Linear) else l, stk, is_leaf=lambda x: isinstance(x, hnn.Linear))) if eqx.is_array(x)]


def mx(a, b):
    return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a, b))


lw_mix, lw_probe, lw_sep, lw_ghost = (linear_weights(t) for t in (g_mix.transformer.layers, g_probe.transformer.layers, g_sep.transformer.layers, g_mix.aux_ghost))
check(mx(lw_mix, lw_probe) < 1e-5, f"4. mix: block Linear gradients = the probes model's (final loss alone; max diff {mx(lw_mix, lw_probe):.1e})")
check(mx(lw_ghost, [s - p for s, p in zip(lw_sep, lw_probe)]) < 1e-5, f"4. mix: ghost gradients = separate heads minus probes ({mx(lw_ghost, [s - p for s, p in zip(lw_sep, lw_probe)]):.1e})")
check(max(float(jnp.max(jnp.abs(x))) for x in lw_ghost) > 1e-4, "4. mix: the ghost gradient is not zero")
check(mx(other_leaves(g_mix.transformer.layers), other_leaves(g_sep.transformer.layers)) < 1e-5, "4. mix: block norms get separate heads' gradient (both losses)")
check(maxdiff((g_mix.embeddings, g_mix.lm_head, g_mix.aux_lm_heads, g_mix.aux_norms), (g_sep.embeddings, g_sep.lm_head, g_sep.aux_lm_heads, g_sep.aux_norms)) < 1e-5, "4. mix: embeddings and heads get separate heads' gradient")
g_mlpo = grads(same_params(M["mlpo"]), ex)
att = lambda g: leaves(g.transformer.layers.stacked.self_attn)  # noqa: E731
mlp = lambda g: leaves(g.transformer.layers.stacked.mlp)  # noqa: E731
check(mx(att(g_mlpo), att(g_probe)) < 1e-5, f"4. mlpo: attention gradients = the probes model's ({mx(att(g_mlpo), att(g_probe)):.1e})")
check(mx(mlp(g_mlpo), mlp(g_sep)) < 1e-5 and mx(mlp(g_sep), mlp(g_probe)) > 1e-4, "4. mlpo: MLP gradients = separate heads' (and differ from the probes')")
key = jrandom.PRNGKey(100)
_, st = run(same_params(M["randl"]), ex, key)
k = int(st["pls/layer"])
g_rl = grads(same_params(M["randl"]), ex, key)
zero = [j for j in range(L - 1) if j != k and float(jnp.max(jnp.abs(g_rl.aux_lm_heads[j].weight.array))) == 0.0]
check(len(zero) == L - 2 and float(jnp.max(jnp.abs(g_rl.aux_lm_heads[k].weight.array))) > 0, f"4. randl: only the drawn head ({k}) gets a gradient")

# 5. adapter in the eval readout
m_ad = same_params(M["ad"])
ex, tok, lw, seg = batch(3)
per = m_ad.readout_losses(ex, tuple(range(L)))
y = np.roll(tok, -1, 1)
for k in (0, L - 1):
    ref = -np.take_along_axis(np.asarray(jax.nn.log_softmax(readout_logits(m_ad, ex, k), -1)), y[..., None], -1)[..., 0]
    got = np.asarray(per["readout", k].rearrange((Batch, Pos)).array)  # per-position loss times the loss weight
    check(np.max(np.abs(got[:, : T - 1] - (lw * ref)[:, : T - 1])) < 1e-4, f"5. adapter: readout_losses of layer {k} equal the readout through _layer")
h_plain = sep._layer(sep.layer_outputs(ex.tokens, ex.attn_mask), 0)
h_ad = m_ad._layer(m_ad.layer_outputs(ex.tokens, ex.attn_mask), 0)
check(float(jnp.max(jnp.abs((h_plain - h_ad).array))) == 0.0, "5. adapter: down starts at zero, so the readout starts as separate heads'")
from experiments.references.per_layer_qwen3 import AdapterMuonHConfig  # noqa: E402

ad_labels = AdapterMuonHConfig(learning_rate=0.02, adam_lr=1e-3).create_mask(eqx.filter(m_ad, eqx.is_array))
check(set(jax.tree.leaves(ad_labels.aux_adapters)) == {"adam"},
      "5. adapter: AdapterMuonHConfig labels every adapter leaf adam")
base_labels = MuonHConfig(learning_rate=0.02, adam_lr=1e-3).create_mask(eqx.filter(m_ad, eqx.is_array))
same = [a == b for a, b in zip(jax.tree.leaves(eqx.tree_at(lambda x: x.aux_adapters, ad_labels, None)), jax.tree.leaves(eqx.tree_at(lambda x: x.aux_adapters, base_labels, None)))]
check(all(same) and len(same) > 0, "5. adapter: every other leaf keeps MuonHConfig's label")
import draccus  # noqa: E402

for oc_ in (MixAuxMuonHConfig(learning_rate=0.02, adam_lr=1e-3, mix_scale=0.2), AdapterMuonHConfig(learning_rate=0.02, adam_lr=1e-3)):
    try:
        draccus.encode(oc_)
        check(True, f"5. {type(oc_).__name__} is registered (draccus encodes it)")
    except Exception as e:
        check(False, f"5. {type(oc_).__name__} is registered (draccus: {e})")
check(float(jnp.max(jnp.abs((sep._layer(sep.layer_outputs(ex.tokens, ex.attn_mask), L - 1) - m_ad._layer(m_ad.layer_outputs(ex.tokens, ex.attn_mask), L - 1)).array))) == 0.0, "5. adapter: the last layer's readout is unchanged")

# 6. MixAuxMuonHConfig, one update against a hand-written reference
mix_model = M["mix"]
params = eqx.filter(mix_model, eqx.is_array)
g = eqx.filter(grads(mix_model, batch(4)[0]), eqx.is_array)
oc = MixAuxMuonHConfig(learning_rate=0.02, adam_lr=1e-3, max_grad_norm=1.0, mix_scale=0.2, warmup=0, min_lr_ratio=1.0, lr_schedule="constant")
opt = oc.build(10)
upd, _ = opt.update(g, opt.init(params), params)
ref = MuonHConfig(learning_rate=0.02, adam_lr=1e-3, max_grad_norm=1.0, warmup=0, min_lr_ratio=1.0, lr_schedule="constant").build(10)
upd_ref, _ = ref.update(g, ref.init(params), params)
labels = oc.create_mask(params)


def labeled(t, lab):
    return [x for x, l in zip(jax.tree.leaves(t), jax.tree.leaves(labels)) if l == lab]


check(maxdiff([x for x in labeled(upd, "adam")], [x for x in labeled(upd_ref, "adam")]) < 1e-7 and maxdiff(labeled(upd, "adamh"), labeled(upd_ref, "adamh")) < 1e-7,
      "6. mix optimizer: adam and adamh updates equal MuonHConfig's")
check(all(float(jnp.max(jnp.abs(x))) == 0.0 for x in leaves(upd.aux_ghost)), "6. mix optimizer: the ghost's update is zero")
# The blocks' Linear updates, checked against MuonHConfig itself (levanter flattens multi-axis weights for Newton-Schulz):
# (i) s = 0 and a zero ghost gradient: MuonHConfig's update; (ii) ghost gradient = the blocks' own: NS(g) + s NS(g) points
# along NS(g), and each part is clipped by an equal norm, so again MuonHConfig's; (iii) s = 1e6 with the ghost gradient
# taken from another batch: MuonHConfig's update on that batch's block gradient (checks which ghost leaf joins which).
def block_linear(u):
    return linear_weights(u.transformer.layers)


def with_ghost(gr, ghost_src):
    return eqx.tree_at(lambda m: m.aux_ghost, gr, ghost_src)


def ghost_like(gr_layers):  # the blocks' gradient moved onto the ghost's structure
    return gr_layers


plain = MuonHConfig(learning_rate=0.02, adam_lr=1e-3, max_grad_norm=1.0, warmup=0, min_lr_ratio=1.0, lr_schedule="constant").build(10)
zero_ghost = jax.tree.map(jnp.zeros_like, g.aux_ghost)
g0 = with_ghost(g, zero_ghost)
u_plain, _ = plain.update(g0, plain.init(params), params)
for scale, gg, name in ((0.0, g0, "(i) s = 0, zero ghost"), (0.2, with_ghost(g, g.transformer.layers), "(ii) ghost = the blocks' gradient")):
    o = MixAuxMuonHConfig(learning_rate=0.02, adam_lr=1e-3, max_grad_norm=1.0, mix_scale=scale, warmup=0, min_lr_ratio=1.0, lr_schedule="constant").build(10)
    u, _ = o.update(gg, o.init(params), params)
    d = mx(block_linear(u), block_linear(u_plain))
    check(d < 1e-5, f"6. mix optimizer {name}: block Linear updates = MuonHConfig's (max diff {d:.1e})")
g2 = eqx.filter(grads(mix_model, batch(5, True)[0]), eqx.is_array)
o = MixAuxMuonHConfig(learning_rate=0.02, adam_lr=1e-3, max_grad_norm=1.0, mix_scale=1e6, warmup=0, min_lr_ratio=1.0, lr_schedule="constant").build(10)
u, _ = o.update(with_ghost(g, g2.transformer.layers), o.init(params), params)
u2, _ = plain.update(with_ghost(g2, zero_ghost), plain.init(params), params)
d = mx(block_linear(u), block_linear(u2))
check(d < 1e-4, f"6. mix optimizer (iii) s = 1e6: block Linear updates = MuonHConfig's on the ghost's batch (max diff {d:.1e})")

# 7. a few training steps of every attempt
for n, m in M.items():
    oc = (MixAuxMuonHConfig(learning_rate=0.02, adam_lr=1e-3, mix_scale=0.2, warmup=0, min_lr_ratio=1.0, lr_schedule="constant") if n == "mix"
          else AdapterMuonHConfig(learning_rate=0.02, adam_lr=1e-3, warmup=0, min_lr_ratio=1.0, lr_schedule="constant") if n == "ad"
          else MuonHConfig(learning_rate=0.02, adam_lr=1e-3, warmup=0, min_lr_ratio=1.0, lr_schedule="constant"))
    opt = oc.build(10)
    p_ = eqx.filter(m, eqx.is_array)
    st = opt.init(p_)
    ok = True
    for i in range(5):
        gr = eqx.filter(grads(m, batch(10 + i, True)[0], jrandom.PRNGKey(i)), eqx.is_array)
        u, st = opt.update(gr, st, p_)
        m = eqx.apply_updates(m, u)
        p_ = eqx.filter(m, eqx.is_array)
        loss, _ = run(m, batch(10 + i, True)[0])
        ok = ok and np.isfinite(loss)
    ghost_zero = n != "mix" or all(float(jnp.max(jnp.abs(x))) == 0.0 for x in leaves(m.aux_ghost))
    if n == "ad":
        ok = ok and all(float(jnp.max(jnp.abs(d.weight.array))) > 0 for _, d in m.aux_adapters)
    check(ok and ghost_zero, f"7. {n}: 5 steps finite (last loss {loss:.4f}){', ghost still zero' if n == 'mix' else ''}")

print("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
