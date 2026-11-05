"""CPU checks of experiments/references/hnet_qwen3.py (2026-10-28) on a tiny config, against a loop reference written
from goombalab/hnet (hnet/models/hnet.py, hnet/modules/dc.py) that calls the model's own encoder, main network and
decoder as black boxes:
1. activations equal the reference (routing, forced boundaries at document starts, chunking, EMA restarted per document,
   plug-back, STE forward value 1, zero residual), with ample slots and with an overflow of the slots;
2. causality: changing byte t leaves every output before t unchanged;
3. documents: changing a byte of document 1 leaves documents 0 and 2 unchanged;
4. the ratio loss matches the formula, and is absent at evaluation (no key);
5. gradients are finite, and reach q_proj, k_proj (through p) and residual_proj.
route="causal" (2026-11-01):
6. activations equal the reference with the routing pair shifted one byte back (p 0 at a document's second byte);
7. no lookahead: replacing byte t and every later byte changes neither p nor the boundary at any position <= t, nor the
   logits before t; the same check on route="start" fails at position t (so the check can fail);
8. the ratio loss matches the formula on the causal p, and the training forward is finite.
Run on a vis node: JAX_PLATFORMS=cpu python tests/test_hnet_qwen3.py"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import haliax as hax
from haliax import Axis
import equinox as eqx
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import current_train_step
import levanter.trainer as lt
from experiments.references.hnet_qwen3 import HNetQwen3Config, HNetQwen3LMHeadModel, HNetRoute

ok = True


def check(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(("ok " if good else "BAD"), name, extra)


L, B = 256, 2
Vocab = Axis("vocab", 512)


def make(slots, route=HNetRoute.START, **kw):
    cfg = HNetQwen3Config(max_seq_len=L, hidden_dim=64, intermediate_dim=128, num_layers=2, num_heads=4, num_kv_heads=4,
                          hybrid_norm=True, attn_backend=AttentionBackend.VANILLA, enc_layers=1, dec_layers=1,
                          chunk_ratio=4.0, chunk_slots=slots, route=route, **kw)
    model = HNetQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
    # perturb q/k off the identity so that boundaries are not all-or-nothing
    q = model.q_proj.weight.array + 0.3 * jrandom.normal(jrandom.PRNGKey(1), model.q_proj.weight.array.shape)
    model = eqx.tree_at(lambda m: m.q_proj.weight, model, hax.named(q, model.q_proj.weight.axes))
    return cfg, model


Batch = Axis("batch", B)
Pos = Axis("position", L)
rng = np.random.default_rng(0)
ids_np = rng.integers(0, 256, size=(B, L))
seg_np = np.zeros((B, L), np.int32)
seg_np[0, 100:180] = 1
seg_np[0, 180:] = 2
seg_np[1, 60:] = 1
ids = hax.named(jnp.asarray(ids_np), (Batch, Pos))
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg_np), (Batch, Pos)))


def reference(model, cfg, ids, mask, opens=None):
    """Loop reference of the H-Net forward, one row at a time; the routing pair of byte t is (t - 1, t), or (t - 2, t - 1)
    with route="causal". ``opens`` [B, L] (bool) gives p = 1 / 0 directly (the data routes space and bpe); with
    cfg.no_ema, z_j = c_j."""
    E, M = cfg.Embed, cfg.slots
    x = model.embeddings.embed(ids)
    h = model.encoder(x, attn_mask=mask, key=None)
    H = np.asarray(h.rearrange((Batch, Pos, E)).array, np.float64)
    Wq = np.asarray(model.q_proj.weight.rearrange((model.q_proj.Out, E)).array, np.float64)
    Wk = np.asarray(model.k_proj.weight.rearrange((model.k_proj.Out, E)).array, np.float64)
    seg = seg_np
    hc = np.zeros((B, M, E.size))
    segc = np.zeros((B, M), np.int32)
    info = []
    for r in range(B):
        p = np.ones(L)
        shift = 1 if cfg.route == HNetRoute.CAUSAL else 0
        doc = 0
        for t in range(1, L):
            if opens is not None:
                p[t] = float(opens[r, t] or seg[r, t] != seg[r, t - 1])
            elif seg[r, t] != seg[r, t - 1]:
                doc = t
                p[t] = 1.0
            elif cfg.route == HNetRoute.FIXED:
                p[t] = float((t - doc) % round(cfg.chunk_ratio) == 0)
            elif shift and (t == 1 or seg[r, t - 1] != seg[r, t - 2]):
                p[t] = 0.0   # a document's second byte: no pair inside the document
            else:
                q, k = Wq @ H[r, t - 1 - shift], Wk @ H[r, t - shift]
                p[t] = np.clip((1 - q @ k / np.linalg.norm(q) / np.linalg.norm(k)) / 2, 0, 1)
        b = p > 0.5
        starts = [t for t in range(L) if b[t]][:M]
        bb = np.zeros(L, bool)
        bb[starts] = True
        for j, t in enumerate(starts):
            hc[r, j] = H[r, t]
            segc[r, j] = seg[r, t]
        segc[r, len(starts):] = seg.max() + 1
        info.append((p, b, bb, starts))
    Chunk = Pos.resize(M)
    zc = model.transformer(hax.named(jnp.asarray(hc, jnp.float32), (Batch, Chunk, E)),
                           attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(segc), (Batch, Chunk))), key=None)
    Z = np.asarray(zc.rearrange((Batch, Chunk, E)).array, np.float64)
    Y = np.zeros((B, L, E.size))
    for r in range(B):
        p, b, bb, starts = info[r]
        z = np.zeros((len(starts), E.size))
        for j, t in enumerate(starts):
            pj = np.clip(p[t], 1e-4, 1 - 1e-4)
            first = j == 0 or seg[r, t] != seg[r, starts[j - 1]]
            z[j] = Z[r, j] if cfg.no_ema else pj * Z[r, j] + (0 if first else (1 - pj) * z[j - 1])
        j = -1
        for t in range(L):
            if bb[t]:
                j += 1
            Y[r, t] = z[j]   # STE forward value 1, zero residual at init
    y = model.decoder(hax.named(jnp.asarray(Y, jnp.float32), (Batch, Pos, E)), attn_mask=mask, key=None)
    return np.asarray(y.rearrange((Batch, Pos, E)).array), info


infos = {}
for route in (HNetRoute.START, HNetRoute.CAUSAL, HNetRoute.FIXED):
    for slots in (128, 24):
        cfg, model = make(slots, route)
        got = np.asarray(model.activations(ids, mask, key=None).rearrange((Batch, Pos, cfg.Embed)).array)
        want, info = reference(model, cfg, ids, mask)
        infos[route] = info
        nb = [int(i[1].sum()) for i in info]
        err = float(np.max(np.abs(got - want)))
        check(f"{route}: forward equals reference, {slots} slots", err < 2e-4, f"max err {err:.2e}, boundaries per row {nb}, kept {[len(i[3]) for i in info]}")
        if slots == 24:
            check(f"{route}: overflow exercised", max(nb) > slots, f"{nb} > {slots}")

cfg, model = make(128)
base = np.asarray(model.activations(ids, mask, key=None).array)
ids2_np = ids_np.copy()
ids2_np[0, 150] = (ids2_np[0, 150] + 7) % 256
out2 = np.asarray(model.activations(hax.named(jnp.asarray(ids2_np), (Batch, Pos)), mask, key=None).array)
d = np.abs(out2 - base).max(axis=-1)
check("autoregressive: outputs before the changed byte unchanged", d[0, :150].max() < 1e-5, f"max {d[0, :150].max():.1e}; at/after {d[0, 150:180].max():.1e}")
check("documents: doc 2 of row 0 and row 1 unchanged", d[0, 180:].max() < 1e-5 and d[1].max() < 1e-5, f"{d[0, 180:].max():.1e}, {d[1].max():.1e}")
check("the change reaches its own document", d[0, 150:180].max() > 1e-3)

# training path: a traced train step is set (as Trainer._train_step does), so the ratio loss is returned
tok = lt._TRACED_TRAIN_STEP.set(jnp.asarray(5))
try:
    def loss_fn(m):
        y, aux = m.activations(ids, mask, key=jrandom.PRNGKey(7))
        w = jrandom.normal(jrandom.PRNGKey(3), y.array.shape)
        return jnp.sum(y.array.astype(jnp.float32) * w) * 1e-3 + aux, aux

    (val, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
finally:
    lt._TRACED_TRAIN_STEP.reset(tok)
P = np.concatenate([i[0] for i in infos[HNetRoute.START]]); Bm = np.concatenate([i[1] for i in infos[HNetRoute.START]]).astype(float)
want_aux = 0.03 * 4.0 / 3.0 * (3.0 * Bm.mean() * P.mean() + (1 - Bm.mean()) * (1 - P.mean()))
check("ratio loss matches the formula", abs(float(aux) - want_aux) < 1e-4, f"{float(aux):.6f} vs {want_aux:.6f}")
check("no ratio loss at evaluation", not isinstance(model.activations(ids, mask, key=None), tuple))
leaves = jax.tree_util.tree_leaves_with_path(eqx.filter(grads, eqx.is_array))
check("all gradients finite", all(bool(jnp.all(jnp.isfinite(g))) for _, g in leaves))
for name, g in (("q_proj", grads.q_proj.weight.array), ("k_proj", grads.k_proj.weight.array), ("residual_proj", grads.residual_proj.array)):
    n = float(jnp.linalg.norm(g))
    check(f"gradient reaches {name}", n > 0, f"norm {n:.3e}")
x = jnp.asarray(0.3)
check("STE forward 1, backward identity", float(jax.value_and_grad(lambda v: __import__("experiments.references.hnet_qwen3", fromlist=["_ste"])._ste(v) * 2.0)(x)[0]) == 2.0
      and float(jax.grad(lambda v: __import__("experiments.references.hnet_qwen3", fromlist=["_ste"])._ste(v) * 2.0)(x)) == 2.0)


def routing(model, ids_):
    """(p, boundaries) of every byte, raw arrays (batch, position)."""
    h = model.encoder(model.embeddings.embed(ids_), attn_mask=mask, key=None)
    p, b_raw, *_ = model.chunked(h, mask)
    return np.asarray(p), np.asarray(b_raw)


def lookahead(model):
    """Replace byte t and every later byte of row 0 (random bytes) for t in TS; return, over all t, the largest change of p
    and the boundary flips at positions < t, at position t, and the largest change of the logits before t."""
    p0, b0 = routing(model, ids)
    lg0 = np.asarray(model(ids, mask, key=None).rearrange((Batch, Pos, Vocab)).array)
    dp_before = dp_at = d_logit = 0.0
    flips_before = flips_at = 0
    for i, t in enumerate(TS):
        ids2_np = ids_np.copy()
        ids2_np[0, t:] = np.random.default_rng(100 + i).integers(0, 256, size=L - t)
        ids2 = hax.named(jnp.asarray(ids2_np), (Batch, Pos))
        p1, b1 = routing(model, ids2)
        lg1 = np.asarray(model(ids2, mask, key=None).rearrange((Batch, Pos, Vocab)).array)
        dp_before = max(dp_before, float(np.abs(p1[0, :t] - p0[0, :t]).max()))
        dp_at = max(dp_at, float(abs(p1[0, t] - p0[0, t])))
        flips_before += int((b1[0, :t] != b0[0, :t]).sum())
        flips_at += int(b1[0, t] != b0[0, t])
        d_logit = max(d_logit, float(np.abs(lg1[0, :t] - lg0[0, :t]).max()))
    return dp_before, dp_at, flips_before, flips_at, d_logit


TS = [t for t in range(2, L) if seg_np[0, t] == seg_np[0, t - 1] == seg_np[0, t - 2]][::5]   # bytes 3+ of a document
_, cmodel = make(128, HNetRoute.CAUSAL)
dpb, dpa, fb, fa, dl = lookahead(cmodel)
check("causal: no lookahead, p and boundaries at <= t and logits before t unchanged", dpb == 0 and dpa == 0 and fb == 0 and fa == 0 and dl < 1e-5,
      f"{len(TS)} positions; max |dp| before t {dpb:.1e}, at t {dpa:.1e}; flips before t {fb}, at t {fa}; max |dlogit| before t {dl:.1e}")
_, smodel = make(128, HNetRoute.START)
dpb, dpa, fb, fa, dl = lookahead(smodel)
check("start: the same check sees byte t's lookahead (p and boundary at t change), and nothing before t", dpa > 1e-3 and fa > 0 and dpb == 0 and fb == 0 and dl < 1e-5,
      f"{len(TS)} positions; max |dp| before t {dpb:.1e}, at t {dpa:.1e}; flips before t {fb}, at t {fa}; max |dlogit| before t {dl:.1e}")

p_c, b_c = routing(cmodel, ids)
starts = [(r, t) for r in range(B) for t in range(L) if t == 0 or seg_np[r, t] != seg_np[r, t - 1]]
check("causal: p = 1 at a document's first byte, 0 at its second", all(p_c[r, t] == 1 for r, t in starts) and all(p_c[r, t + 1] == 0 for r, t in starts),
      f"{len(starts)} documents")

tok = lt._TRACED_TRAIN_STEP.set(jnp.asarray(5))
try:
    y, aux = cmodel.activations(ids, mask, key=jrandom.PRNGKey(7))
finally:
    lt._TRACED_TRAIN_STEP.reset(tok)
want_aux = 0.03 * 4.0 / 3.0 * (3.0 * b_c.mean() * p_c.mean() + (1 - b_c.mean()) * (1 - p_c.mean()))
check("causal: ratio loss finite and matches the formula on the causal p", bool(jnp.isfinite(aux)) and abs(float(aux) - want_aux) < 1e-4,
      f"{float(aux):.6f} vs {want_aux:.6f}; boundary share {b_c.mean():.3f}")
lg = cmodel(ids, mask, key=None).array
check("causal: training activations and eval logits finite", bool(jnp.all(jnp.isfinite(y.array))) and bool(jnp.all(jnp.isfinite(lg))))
# route="fixed": no learned routing
_, fmodel = make(128, HNetRoute.FIXED)
p_f, b_f = routing(fmodel, ids)
want_f = np.zeros((B, L), bool)
for r in range(B):
    doc = 0
    for t in range(L):
        if t and seg_np[r, t] != seg_np[r, t - 1]:
            doc = t
        want_f[r, t] = (t - doc) % 4 == 0
check("fixed: a boundary every 4 bytes from each document's first byte, p is 0 or 1", np.array_equal(b_f, want_f) and set(np.unique(p_f)) <= {0.0, 1.0},
      f"boundaries per row {b_f.sum(axis=1).tolist()}")
ids2_np = ids_np.copy()
ids2_np[0, 50:] = np.random.default_rng(5).integers(0, 256, size=L - 50)
_, b_f2 = routing(fmodel, hax.named(jnp.asarray(ids2_np), (Batch, Pos)))
check("fixed: boundaries do not depend on the bytes", np.array_equal(b_f, b_f2))


def floss(m):
    t_ = lt._TRACED_TRAIN_STEP.set(jnp.asarray(5))
    try:
        y_, aux_ = m.activations(ids, mask, key=jrandom.PRNGKey(7))
    finally:
        lt._TRACED_TRAIN_STEP.reset(t_)
    return jnp.mean(jnp.square(y_.array)) + aux_


fgrads = eqx.filter_grad(floss)(fmodel)
fnorm = lambda g: float(jnp.linalg.norm(g.weight.array))   # noqa: E731
check("fixed: q and k get no gradient, the encoder does", fnorm(fgrads.q_proj) == 0 and fnorm(fgrads.k_proj) == 0
      and sum(float(jnp.sum(jnp.abs(g))) for g in jax.tree_util.tree_leaves(eqx.filter(fgrads.encoder, eqx.is_array))) > 0,
      f"q {fnorm(fgrads.q_proj):.1e}, k {fnorm(fgrads.k_proj):.1e}")
check("fixed: all gradients finite", all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(eqx.filter(fgrads, eqx.is_array))))
# MuonH on a zero gradient must leave the weight where it is (no division by a zero norm)
from levanter.optim.muonh import scale_with_muonh  # noqa: E402
_w = {"w": fmodel.q_proj}
_opt = scale_with_muonh(learning_rate=0.01)
_st = _opt.init(_w)
_u, _ = _opt.update({"w": fgrads.q_proj}, _st, _w)
_uw = _u["w"].weight.array
check("fixed: a MuonH step on the zero gradient of q is finite and zero", bool(jnp.all(jnp.isfinite(_uw))) and float(jnp.abs(_uw).max()) < 1e-5, f"max |update| {float(jnp.abs(_uw).max()):.1e}")

# ---- ablations (owner popup 2026-11-03) ----
from experiments.references.hnet_qwen3 import BPE_FLAG, WHITESPACE_IDS  # noqa: E402
from experiments.references.hnet_freeze_muonh import CHUNKER, FreezeChunkerMuonHConfig  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.models.loss import maybe_fused_next_token_loss  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402


def train_fwd(m, ids_, mask_=mask):
    t_ = lt._TRACED_TRAIN_STEP.set(jnp.asarray(5))
    try:
        return m.activations(ids_, mask_, key=jrandom.PRNGKey(7))
    finally:
        lt._TRACED_TRAIN_STEP.reset(t_)


def leaves(t):
    return jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array))


# route="space": text with whitespace and special tokens (BOS 256 at each document start, an EOS 257 at row 0's first end)
sp_np = rng.integers(0, 256, size=(B, L))
sp_np[rng.random((B, L)) < 0.18] = 220
sp_np[rng.random((B, L)) < 0.04] = 198
sp_np[0, 99] = 257
for r in range(B):
    for t in range(L):
        if t == 0 or seg_np[r, t] != seg_np[r, t - 1]:
            sp_np[r, t] = 256
sp_ids = hax.named(jnp.asarray(sp_np), (Batch, Pos))
want_sp = np.zeros((B, L), bool)
for r in range(B):
    for t in range(L):
        want_sp[r, t] = (t == 0 or seg_np[r, t] != seg_np[r, t - 1] or sp_np[r, t - 1] in WHITESPACE_IDS or sp_np[r, t - 1] >= 256
                         or sp_np[r, t] >= 256)
for slots in (128, 24):
    scfg, smodel_ = make(slots, HNetRoute.SPACE)
    got = np.asarray(smodel_.activations(sp_ids, mask, key=None).rearrange((Batch, Pos, scfg.Embed)).array)
    want, info = reference(smodel_, scfg, sp_ids, mask, opens=want_sp)
    err = float(np.max(np.abs(got - want)))
    check(f"space: forward equals reference, {slots} slots", err < 2e-4, f"max err {err:.2e}, boundaries per row {[int(i[1].sum()) for i in info]}")
_, smodel_ = make(128, HNetRoute.SPACE)
h_sp = smodel_.encoder(smodel_.embeddings.embed(sp_ids), attn_mask=mask, key=None)
p_sp, b_sp, *_ = smodel_.chunked(h_sp, mask, ids=sp_ids)
check("space: boundaries after whitespace, at specials and document starts; p is 0 or 1", np.array_equal(np.asarray(b_sp), want_sp)
      and set(np.unique(np.asarray(p_sp))) <= {0.0, 1.0}, f"{want_sp.mean() and 1 / want_sp.mean():.2f} bytes per chunk")
sg = eqx.filter_grad(lambda m: jnp.mean(jnp.square(train_fwd(m, sp_ids)[0].array)) + train_fwd(m, sp_ids)[1])(smodel_)
check("space: q and k get no gradient, all gradients finite", fnorm(sg.q_proj) == 0 and fnorm(sg.k_proj) == 0
      and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves(sg)))

# route="word" (owner popup 2026-11-05 "跑"): as space, but whitespace never opens a chunk, so a run of it joins the chunk before
want_wd = np.zeros((B, L), bool)
for r in range(B):
    for t in range(L):
        want_wd[r, t] = (t == 0 or seg_np[r, t] != seg_np[r, t - 1] or sp_np[r, t - 1] >= 256 or sp_np[r, t] >= 256
                         or (sp_np[r, t - 1] in WHITESPACE_IDS and sp_np[r, t] not in WHITESPACE_IDS))
for slots in (128, 24):
    wcfg, wmodel_ = make(slots, HNetRoute.WORD)
    got = np.asarray(wmodel_.activations(sp_ids, mask, key=None).rearrange((Batch, Pos, wcfg.Embed)).array)
    want, info = reference(wmodel_, wcfg, sp_ids, mask, opens=want_wd)
    err = float(np.max(np.abs(got - want)))
    check(f"word: forward equals reference, {slots} slots", err < 2e-4, f"max err {err:.2e}, boundaries per row {[int(i[1].sum()) for i in info]}")
_, wmodel_ = make(128, HNetRoute.WORD)
h_wd = wmodel_.encoder(wmodel_.embeddings.embed(sp_ids), attn_mask=mask, key=None)
p_wd, b_wd, *_ = wmodel_.chunked(h_wd, mask, ids=sp_ids)
check("word: boundaries at a non-whitespace byte after whitespace, at and after specials, at document starts; p is 0 or 1",
      np.array_equal(np.asarray(b_wd), want_wd) and set(np.unique(np.asarray(p_wd))) <= {0.0, 1.0},
      f"{1 / want_wd.mean():.2f} bytes per chunk, space {1 / want_sp.mean():.2f}")
check("word: its boundaries are space's minus those on whitespace after whitespace", bool(np.all(want_wd <= want_sp)) and int((want_sp & ~want_wd).sum()) > 0
      and all(sp_np[r, t] in WHITESPACE_IDS for r, t in zip(*np.nonzero(want_sp & ~want_wd))), f"{int((want_sp & ~want_wd).sum())} fewer")
wg = eqx.filter_grad(lambda m: jnp.mean(jnp.square(train_fwd(m, sp_ids)[0].array)) + train_fwd(m, sp_ids)[1])(wmodel_)
check("word: q and k get no gradient, all gradients finite", fnorm(wg.q_proj) == 0 and fnorm(wg.k_proj) == 0
      and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves(wg)))

# route="bpe": token starts flagged as id + 512; the model must equal the byte model given those boundaries by hand
bp_np = sp_np.copy()
starts_bp = want_sp.copy()                     # any set of starts works; use a random one that differs from space
starts_bp |= rng.random((B, L)) < 0.15
text = bp_np < 256
flag = starts_bp & text
flag[:, 0] = False                              # position 0 is BOS (special, never flagged)
flagged_np = np.where(flag, bp_np + BPE_FLAG, bp_np)
flagged = hax.named(jnp.asarray(flagged_np), (Batch, Pos))
hand_p = np.zeros((B, L), np.float32)
for r in range(B):
    for t in range(L):
        hand_p[r, t] = float(t == 0 or flag[r, t] or bp_np[r, t] >= 256 or seg_np[r, t] != seg_np[r, t - 1])
bcfg, bmodel = make(24, HNetRoute.BPE)
_, hmodel = make(24, HNetRoute.START)          # the same weights (same init key), the learned route replaced by hand
plain = hax.named(jnp.asarray(bp_np), (Batch, Pos))
yb = np.asarray(bmodel.activations(flagged, mask, key=None).array)
yh = np.asarray(hmodel.activations(plain, mask, key=None, route_p=jnp.asarray(hand_p)).array)
check("bpe: forward on flagged ids equals the byte model with the token starts given by hand (overflow included)",
      np.array_equal(yb, yh), f"max |d| {np.abs(yb - yh).max():.1e}; boundaries per row {hand_p.sum(1).tolist()} for {bcfg.slots} slots")
ex_b = LmExample.causal(flagged["batch", 0], segment_ids=hax.named(jnp.asarray(seg_np[0]), (Pos,)))
ex_h = LmExample.causal(plain["batch", 0], segment_ids=hax.named(jnp.asarray(seg_np[0]), (Pos,)))
lb = bmodel.compute_next_token_loss(ex_b, reduction=None, reduction_axis=())
yh0 = hmodel.activations(plain["batch", 0], ex_h.attn_mask, key=None, route_p=jnp.asarray(hand_p[0]))
lh = maybe_fused_next_token_loss(hmodel.Pos, hmodel.Embed, hmodel.Vocab, yh0, hmodel.get_lm_head(), plain["batch", 0],
                                 loss_weight=ex_h.loss_weight, reduction=None, reduction_axis=(), dtype=jnp.float32)
check("bpe: per-byte loss equals the hand run's, whose targets are the plain bytes", np.array_equal(np.asarray(lb.array), np.asarray(lh.array)),
      f"mean {float(jnp.mean(lb.array)):.4f} nats")
ids_s, st_s = bmodel.split_ids(flagged)
check("bpe: the stripped ids are the bytes (< 512), the starts the flags", np.array_equal(np.asarray(ids_s.array), bp_np)
      and np.array_equal(np.asarray(st_s.array), flag) and int(np.asarray(ids_s.array).max()) < Vocab.size)
check("bpe: the flags change the boundaries (the check can fail)", not np.array_equal(
      np.asarray(bmodel.activations(plain, mask, key=None).array), yb))
try:
    HNetQwen3LMHeadModel.init(Axis("vocab", 768), bcfg, key=jrandom.PRNGKey(0))
    check("bpe: a vocabulary other than 512 fails", False)
except ValueError:
    check("bpe: a vocabulary other than 512 fails", True)

# route_p (hnet_eval.py's boundary swap): a model's own p gives back its own forward; another model's p sets the boundaries
cfg_a, ma = make(128)
_, mb = make(128)
mb = eqx.tree_at(lambda m: m.k_proj.weight, mb, hax.named(mb.k_proj.weight.array + 0.3 * jrandom.normal(jrandom.PRNGKey(9), mb.k_proj.weight.array.shape), mb.k_proj.weight.axes))
pa, pb = ma.routing(ids, mask), mb.routing(ids, mask)
ya = np.asarray(ma.activations(ids, mask, key=None).array)
check("route_p: a model's own routing gives back its forward bit for bit", np.array_equal(ya, np.asarray(ma.activations(ids, mask, key=None, route_p=pa).array)))
h_a = ma.encoder(ma.embeddings.embed(ids), attn_mask=mask, key=None)
_, b_ab, _, _, _, hc_ab, _, _ = ma.chunked(h_a, mask, route_p=pb)
_, b_bb, *_ = mb.chunked(mb.encoder(mb.embeddings.embed(ids), attn_mask=mask, key=None), mask)
nb_a, flips = int(np.asarray(pa > 0.5).sum()), int((np.asarray(pa > 0.5) != np.asarray(pb > 0.5)).sum())
check("route_p: the swap takes the other model's boundaries, the chunk vectors stay the model's own encoder states",
      np.array_equal(np.asarray(b_ab), np.asarray(b_bb)) and flips > 0, f"{flips} positions differ; model a has {nb_a} boundaries")

# no_route_grad: the task loss gives q and k no gradient, the ratio loss does; the encoder still learns
_, nmodel = make(128, no_route_grad=True)
w_ = jrandom.normal(jrandom.PRNGKey(3), (B, L, 64))
g_task = eqx.filter_grad(lambda m: jnp.sum(train_fwd(m, ids)[0].rearrange((Batch, Pos, m.config.Embed)).array * w_))(nmodel)
g_ratio = eqx.filter_grad(lambda m: train_fwd(m, ids)[1])(nmodel)
g_task_def = eqx.filter_grad(lambda m: jnp.sum(train_fwd(m, ids)[0].rearrange((Batch, Pos, m.config.Embed)).array * w_))(make(128)[1])
enc_norm = sum(float(jnp.sum(jnp.abs(g))) for g in leaves(g_task.encoder))
check("no_route_grad: task loss gives q and k zero gradient, the encoder a nonzero one", fnorm(g_task.q_proj) == 0 and fnorm(g_task.k_proj) == 0 and enc_norm > 0,
      f"encoder |g| {enc_norm:.2e}; default model's task gradient into q {fnorm(g_task_def.q_proj):.2e}")
check("no_route_grad: the ratio loss gives q and k a nonzero gradient", fnorm(g_ratio.q_proj) > 0 and fnorm(g_ratio.k_proj) > 0,
      f"q {fnorm(g_ratio.q_proj):.2e}, k {fnorm(g_ratio.k_proj):.2e}")
check("no_route_grad: forward equals the default's", np.array_equal(np.asarray(nmodel.activations(ids, mask, key=None).array), base))
try:
    make(128, HNetRoute.FIXED, no_route_grad=True)
    check("no_route_grad with a data route fails", False)
except ValueError:
    check("no_route_grad with a data route fails", True)

# ema_stop_grad: the forward pass does not change, the task gradient into q and k comes through the STE factor alone
_, smodel = make(128, ema_stop_grad=True)
g_stop = eqx.filter_grad(lambda m: jnp.sum(train_fwd(m, ids)[0].rearrange((Batch, Pos, m.config.Embed)).array * w_))(smodel)
check("ema_stop_grad: forward equals the default's", np.array_equal(np.asarray(smodel.activations(ids, mask, key=None).array), base))
check("ema_stop_grad: task gradient into q and k is nonzero and differs from the default's", fnorm(g_stop.q_proj) > 0 and fnorm(g_stop.k_proj) > 0 and fnorm(g_stop.q_proj) != fnorm(g_task_def.q_proj),
      f"q {fnorm(g_stop.q_proj):.2e}, default {fnorm(g_task_def.q_proj):.2e}")
for kw in (dict(no_ema=True), dict(no_route_grad=True)):
    try:
        make(128, ema_stop_grad=True, **kw)
        check(f"ema_stop_grad with {kw} fails", False)
    except ValueError:
        check(f"ema_stop_grad with {kw} fails", True)
try:
    make(128, HNetRoute.FIXED, ema_stop_grad=True)
    check("ema_stop_grad with a data route fails", False)
except ValueError:
    check("ema_stop_grad with a data route fails", True)

# no_residual: R h_t is gone (a nonzero R changes the default model, not this one); R gets a zero gradient and a zero step
R = hax.named(0.05 * jrandom.normal(jrandom.PRNGKey(4), model.residual_proj.array.shape), model.residual_proj.axes)
_, rmodel = make(128, no_residual=True)
_, dmodel = make(128)
y_r0 = np.asarray(rmodel.activations(ids, mask, key=None).array)
y_rR = np.asarray(eqx.tree_at(lambda m: m.residual_proj, rmodel, R).activations(ids, mask, key=None).array)
y_dR = np.asarray(eqx.tree_at(lambda m: m.residual_proj, dmodel, R).activations(ids, mask, key=None).array)
check("no_residual: R does not reach the output; the default model with the same R does", np.array_equal(y_r0, y_rR) and np.array_equal(y_r0, base)
      and np.abs(y_dR - base).max() > 1e-3, f"default with R: max |d| {np.abs(y_dR - base).max():.1e}")
rg = eqx.filter_grad(lambda m: jnp.mean(jnp.square(train_fwd(m, ids)[0].array)) + train_fwd(m, ids)[1])(rmodel)
check("no_residual: R's gradient is zero, all gradients finite", float(jnp.abs(rg.residual_proj.array).max()) == 0
      and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves(rg)))

# no_ema: z_j = c_j, against the reference
for slots in (128, 24):
    ecfg, emodel = make(slots, no_ema=True)
    got = np.asarray(emodel.activations(ids, mask, key=None).rearrange((Batch, Pos, ecfg.Embed)).array)
    want, _ = reference(emodel, ecfg, ids, mask)
    err = float(np.max(np.abs(got - want)))
    check(f"no_ema: forward equals the reference with z_j = c_j, {slots} slots", err < 2e-4 and np.abs(got - base).max() > 1e-3 if slots == 128 else err < 2e-4,
          f"max err {err:.2e}")

# slot_margin
check("slot_margin: slots = window / N x margin rounded up to 128", make(0)[0].slots == 128 and make(0, slot_margin=2.0)[0].slots == 128 * int(np.ceil(256 / 4 * 2 / 128))
      and HNetQwen3Config(max_seq_len=19456, chunk_ratio=4.74, slot_margin=1.5).slots == 6272 and HNetQwen3Config(max_seq_len=19456, chunk_ratio=4.74).slots == 5248)


# optimizer steps: MuonH as the runs build it, on the tiny model (no_residual and the chunker freeze)
def opt_cfg(cls, **kw):
    return cls(learning_rate=0.01, adam_lr=0.002, beta1=0.9, beta2=0.98, epsilon=1e-15, momentum=0.98, nesterov=True,
               backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0, weight_decay=0.1, lr_schedule="cosine", warmup=0,
               min_lr_ratio=0.0, coefficient_type="simple", **kw)


def lm_loss(m, ids_):
    """The trainer's loss: next-byte cross-entropy (compute_next_token_loss) plus the ratio loss, in a traced train step."""
    ex = LmExample(tokens=ids_, loss_weight=hax.named(jnp.broadcast_to((jnp.arange(L) < L - 1).astype(jnp.float32), (B, L)), (Batch, Pos)),
                   attn_mask=mask)
    t_ = lt._TRACED_TRAIN_STEP.set(jnp.asarray(5))
    try:
        return m.compute_next_token_loss(ex, key=jrandom.PRNGKey(7)).scalar()
    finally:
        lt._TRACED_TRAIN_STEP.reset(t_)


def run_steps(m, opt, n, batches):
    params = eqx.filter(m, eqx.is_inexact_array)
    static = eqx.filter(m, eqx.is_inexact_array, inverse=True)
    st = opt.init(params)
    hist = [params]
    for i in range(n):
        g = eqx.filter_grad(lambda pp: lm_loss(eqx.combine(pp, static), batches[i % len(batches)]))(params)
        u, st = opt.update(g, st, params)
        params = eqx.apply_updates(params, u)
        hist.append(params)
    return hist


bat = [ids, hax.named(jnp.asarray(rng.integers(0, 256, size=(B, L))), (Batch, Pos))]
hr_ = run_steps(rmodel, opt_cfg(MuonHConfig).build(10), 2, bat)
check("no_residual: two MuonH steps leave R exactly zero and every parameter finite", float(jnp.abs(hr_[-1].residual_proj.array).max()) == 0
      and all(bool(jnp.all(jnp.isfinite(x))) for x in leaves(hr_[-1])))

S0, NSTEP = 2, 4
hf = run_steps(model, opt_cfg(FreezeChunkerMuonHConfig, freeze_from_step=S0).build(10), NSTEP, bat)
hb = run_steps(model, opt_cfg(MuonHConfig).build(10), NSTEP, bat)
same_before = all(all(bool(jnp.array_equal(a, b)) for a, b in zip(leaves(hf[i]), leaves(hb[i]))) for i in range(S0 + 1))
check(f"chunker freeze: updates 0..{S0 - 1} bit-identical to plain MuonH", same_before)
frozen_ok = all(all(bool(jnp.array_equal(a, b)) for a, b in zip(leaves(getattr(hf[i], n)), leaves(getattr(hf[S0], n))))
                for n in CHUNKER for i in range(S0, NSTEP + 1))
moved = {n: max(float(jnp.abs(a - b).max()) for a, b in zip(leaves(getattr(hf[NSTEP], n)), leaves(getattr(hf[S0], n))))
         for n in ("transformer", "decoder", "lm_head", "residual_proj") if leaves(getattr(hf[NSTEP], n))}
moved_before = {n: max(float(jnp.abs(a - b).max()) for a, b in zip(leaves(getattr(hf[S0], n)), leaves(getattr(hf[0], n)))) for n in CHUNKER}
check(f"chunker freeze: embedding, encoder, q, k exactly equal over updates {S0}..{NSTEP - 1}; they moved before", frozen_ok and min(moved_before.values()) > 0,
      " ".join(f"{k} {v:.1e}" for k, v in moved_before.items()))
check("chunker freeze: main network, decoder, head and R keep moving", all(v > 0 for v in moved.values()),
      " ".join(f"{k} {v:.1e}" for k, v in moved.items()))
print("ALL OK" if ok else "FAILED")
