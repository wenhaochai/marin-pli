"""CPU tests for per_layer_qwen3's multi-token prediction (pls_mtp_weight) and future probes (pls_future_probes).

    WT=<worktree>; PYTHONPATH=$WT:$WT/lib/*/src JAX_PLATFORMS=cpu nice -n 19 <venv>/bin/python scripts/della/pls_mtp_cpu_test.py
1. Init: with the MTP module and future probes on, every other parameter is byte-identical to the model with them off.
2. Additivity: on the same parameters, loss(on) = loss(off) + w * stats[mtp] + sum_k stats[pls2/L{k}], and every
   pls/L{k} statistic is unchanged (the off path is untouched).
3. MTP CE equals a brute-force per-position CE: the module's output at t through the shared lm_head against x_{t+2},
   weight loss_weight[t+1], masked where t+2 leaves the window or crosses a document (segment ids), weighted mean.
4. Future probe CE (one layer) equals the same brute force on its own norm and head over the layer output.
5. Gradients: the future probes send exactly zero gradient into the embeddings, the blocks, the final norm, lm_head and
   the aux heads, and a nonzero one into their own heads; the MTP loss reaches the blocks, the embeddings and lm_head.
6. Optimizer labels (SplitAuxMuonHConfig): fut_lm_heads -> aux_adamh, fut_norms -> aux_adam; the MTP module's Linears
   muonh, its norms adam; every other leaf as MuonHConfig.
7. Passivity end to end: probes + future probes (SplitAux, clip 1.0, 25 steps, L = 6) keep the backbone equal to the
   baseline Qwen3 trained with MuonHConfig, up to float rounding; with MTP on it drifts away (not vacuous).
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
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel, SplitAuxMuonHConfig  # noqa: E402

B, T, V, L = 8, 16, 128, 6
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
perm = np.random.default_rng(0).permutation(V)
FAIL = []
W = 0.3


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
    lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)
    lw[:, 5] = 0.5  # a non-trivial weight, so the t+1 shift of the weight is tested
    mask = AttentionMask.causal()
    seg = None
    if segments:  # documents of random lengths
        seg = np.cumsum(r.random((B, T)) < 0.25, axis=1).astype(np.int32)
        mask = mask.with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos)))
    ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)), attn_mask=mask)
    return ex, tok, lw, seg


def common(**kw):
    return dict(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True, use_qk_norm=False, attn_backend=AttentionBackend.VANILLA, **kw)


PROBES = dict(pls_weight=1.0, pls_separate_heads=True, pls_detach_backbone=True)
key0 = jrandom.PRNGKey(0)
off = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common(**PROBES)), key=key0)
on = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common(**PROBES, pls_mtp_weight=W, pls_future_probes=True)), key=key0)


def maxdiff(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    return max(float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)))) for x, y in zip(la, lb))


def shared(m):
    return eqx.filter((m.embeddings, m.transformer, m.lm_head, m.aux_norms, m.aux_lm_heads), eqx.is_array)


# 1. init
check(maxdiff(shared(off), shared(on)) == 0.0, "1. every parameter outside the MTP module and the future probes is byte-identical")
check(on.mtp is not None and len(on.fut_lm_heads) == L and len(on.fut_norms) == L and off.mtp is None and off.fut_lm_heads is None, f"1. MTP module and {L} future probes present only when on")

KW = dict(key=jrandom.PRNGKey(1), logsumexp_weight=0.0)


def stat(v):
    for name in ("value", "item"):
        if hasattr(v, name) and callable(getattr(v, name)):
            try:
                return float(getattr(v, name)())
            except Exception:
                pass
    for name in ("_value", "_sum"):
        if hasattr(v, name):
            return float(getattr(v, name))
    return float(v)


def losses(m, ex):
    loss, stats = m.compute_next_token_loss(ex, **KW)
    return float(loss.array if isinstance(loss, hax.NamedArray) else loss), {k: stat(v) for k, v in stats.items()}


# 2. additivity
for segs in (False, True):
    ex, *_ = batch(0, segs)
    lo, so = losses(off, ex)
    ln, sn = losses(on, ex)
    rest = W * sn["pls/mtp"] + sum(sn[f"pls2/L{k}"] for k in range(L))
    check(abs(ln - (lo + rest)) < 1e-4 * max(1.0, abs(ln)), f"2. segments={segs}: loss(on) {ln:.5f} = loss(off) {lo:.5f} + w*mtp + sum fut ({rest:.5f})")
    check(all(abs(so[k] - sn[k]) < 1e-6 for k in so), f"2. segments={segs}: every pls/L{{k}} statistic unchanged ({len(so)})")


def brute(pred, head, tok, lw, seg):
    """Weighted-mean CE of pred[b, t] @ head against x_{t+2}, weight lw[b, t+1], valid t+2 <= T-1 and same segment."""
    logits = jnp.einsum("bte,ve->btv", pred.astype(jnp.float32), head.astype(jnp.float32))
    logp = jax.nn.log_softmax(logits, -1)
    num = den = 0.0
    for b in range(B):
        for t in range(T - 2):
            w = lw[b, t + 1] * (1.0 if seg is None or seg[b, t] == seg[b, t + 2] else 0.0)
            num += w * -float(logp[b, t, tok[b, t + 2]])
            den += w
    return num / den, num / (B * T)


def arr(x, axes):
    return x.rearrange(axes).array


# 3. MTP CE brute force
for segs in (False, True):
    ex, tok, lw, seg = batch(1, segs)
    outs = on.layer_outputs(ex.tokens, ex.attn_mask, key=KW["key"])
    h = outs[on.transformer.layers.Block.name, L - 1]
    e_next = on.embeddings.embed(hax.roll(ex.tokens, -1, Pos))
    pred = on.mtp(h, e_next, ex.attn_mask, key=jrandom.fold_in(KW["key"], 0x3A))
    want_mean, want_flat = brute(arr(pred, (Batch, Pos, on.Embed)), arr(on.get_lm_head(), (Vocab, on.Embed)), tok, lw, seg)
    _, st = losses(on, ex)
    got = st["pls/mtp"]
    check(abs(got - want_mean) < 1e-4, f"3. segments={segs}: MTP CE {got:.6f} == brute force weighted mean {want_mean:.6f} (flat mean would be {want_flat:.6f})")

# 4. future probe CE brute force, layer 2 and the last layer
ex, tok, lw, seg = batch(2, True)
outs = on.layer_outputs(ex.tokens, ex.attn_mask, key=KW["key"])
_, st = losses(on, ex)
for k in (2, L - 1):
    hk = on.fut_norms[k](outs[on.transformer.layers.Block.name, k])
    want, _ = brute(arr(hk, (Batch, Pos, on.Embed)), arr(on.fut_lm_heads[k].weight, (Vocab, on.Embed)), tok, lw, seg)
    check(abs(st[f"pls2/L{k}"] - want) < 1e-4, f"4. future probe L{k}: {st[f'pls2/L{k}']:.6f} == brute force {want:.6f}")


# 5. gradients
def _fut_sum(mm):
    outs = mm.layer_outputs(ex.tokens, ex.attn_mask, key=KW["key"])
    tot = 0.0
    for k in range(L):
        ce = mm._future_ce(outs, k, ex, logsumexp_weight=0.0)
        tot = tot + (ce.array if isinstance(ce, hax.NamedArray) else ce)
    return tot


def _mtp(mm):
    outs = mm.layer_outputs(ex.tokens, ex.attn_mask, key=KW["key"])
    ce = mm._mtp_ce(outs, ex, key=jrandom.fold_in(KW["key"], 0x3A), logsumexp_weight=0.0)
    return ce.array if isinstance(ce, hax.NamedArray) else ce


def norm(t):
    leaves = [x for x in jax.tree.leaves(eqx.filter(t, eqx.is_array))]
    return float(sum(jnp.sum(jnp.abs(x.astype(jnp.float32))) for x in leaves)) if leaves else 0.0


gf = eqx.filter_grad(_fut_sum)(on)
check(norm((gf.embeddings, gf.transformer, gf.lm_head, gf.aux_norms, gf.aux_lm_heads, gf.mtp)) == 0.0, "5. future probes: exactly zero gradient into embeddings, blocks, final norm, lm_head, aux heads, MTP module")
check(norm(gf.fut_lm_heads) > 0 and norm(gf.fut_norms) > 0, "5. future probes: nonzero gradient into their own heads and norms")
gm = eqx.filter_grad(_mtp)(on)
check(norm(gm.transformer.layers) > 0 and norm(gm.embeddings) > 0 and norm(gm.lm_head) > 0 and norm(gm.mtp) > 0, "5. MTP loss reaches the blocks, the embeddings, lm_head and its module")
check(norm((gm.fut_lm_heads, gm.aux_lm_heads)) == 0.0, "5. MTP loss: zero gradient into the probes")


# 6. labels
def labels(mask):
    flat = jax.tree_util.tree_leaves_with_path(mask, is_leaf=lambda x: isinstance(x, str))
    return {jax.tree_util.keystr(p): lab for p, lab in flat}


p_on = eqx.filter(on, eqx.is_array)
base_l, split_l = labels(MuonHConfig().create_mask(p_on)), labels(SplitAuxMuonHConfig().create_mask(p_on))
fut = {k: v for k, v in split_l.items() if k.startswith((".fut_lm_heads", ".fut_norms"))}
check(len(fut) == 2 * L and all(v == ("aux_adamh" if k.startswith(".fut_lm_heads") else "aux_adam") for k, v in fut.items()), f"6. {len(fut)} future-probe leaves: heads aux_adamh, norms aux_adam")
mtp_l = {k: v for k, v in split_l.items() if k.startswith(".mtp")}
lin = {k: v for k, v in mtp_l.items() if "proj" in k and k.endswith("weight")}
nrm = {k: v for k, v in mtp_l.items() if "norm" in k}
check(len(lin) >= 8 and all(v == "muonh" for v in lin.values()), f"6. MTP Linear weights muonh ({len(lin)}): {sorted(set(lin.values()))}")
check(len(nrm) >= 3 and all(v == "adam" for v in nrm.values()), f"6. MTP norms adam ({len(nrm)}): {sorted(set(nrm.values()))}")
check(all(split_l[k] == base_l[k] for k in base_l if not k.startswith((".fut_", ".aux_"))), "6. every other leaf keeps MuonHConfig's label")


# 7. passivity end to end
def recipe(cls, clip):
    return cls(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5, muon_epsilon=1e-5,
               max_grad_norm=clip, weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0, coefficient_type="simple")


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
        model, state = step(model, state, batch(i)[0])
        out.append(model)
    return out


def backbone(m):
    return eqx.filter((m.embeddings, m.transformer, m.lm_head), eqx.is_array)


steps = 25
base = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common()), key=key0)
fut_only = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common(**PROBES, pls_future_probes=True)), key=key0)
check(maxdiff(backbone(base), backbone(fut_only)) == 0.0 and maxdiff(backbone(base), backbone(on)) == 0.0, "7. backbone init identical to the baseline Qwen3")
tb = train(base, recipe(MuonHConfig, 1.0), steps)
tf = train(fut_only, recipe(SplitAuxMuonHConfig, 1.0), steps)
tm = train(on, recipe(SplitAuxMuonHConfig, 1.0), steps)
d_fut = max(maxdiff(backbone(a), backbone(b)) for a, b in zip(tb, tf))
d_mtp = max(maxdiff(backbone(a), backbone(b)) for a, b in zip(tb, tm))
check(d_fut < 1e-4, f"7. probes + future probes, SplitAux, clip 1.0, {steps} steps: backbone == baseline (max diff {d_fut:.2e})")
check(d_mtp > 100 * max(d_fut, 1e-7), f"7. with MTP the backbone drifts from the baseline (max diff {d_mtp:.2e}): the check is not vacuous")
check(all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(eqx.filter(tm[-1], eqx.is_array))), "7. MTP run finite after 25 steps")

print("ALL MTP CPU TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILED")
