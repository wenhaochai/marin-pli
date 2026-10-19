"""Follow-up of the AttnRes check-up (attnres_checkup.py) on the 48-layer AttnRes models (owner standing approval for the
AttnRes check-up, 2026-10-19: "越全面越好"). Read-only: loads a finished checkpoint, trains nothing.

Sources, mixes and text as in attnres_checkup.py: slot 0 is the embedding; attnres: slot 1 + s holds the output of sublayer
s = 2 i + j (i 0-based layer, j 0 attention / 1 MLP); attnres_block: slot 1 + b holds block b's running sum (6 layers per
block). Every mix (read) is sum_i w_i v_i, w = softmax_i((v_i . q) / RMS(v_i)): the key is the RMS-normalised value, the
value is raw. Reads: sublayer s's input mix (s = 0 .. 2L-1) and the head's mix (s = 2L). Paloma c4_en validation, the
first 16 x 1,024 tokens; final CE (and top-1 acc) through the model's own final norm + lm_head over valid next-token
targets (attnres_checkup.make_batch). Weight statistics: over all 16 x 1,024 tokens.

A. Giant-source test. A target is a source slot plus the first read it applies to (start_s). Recorded in the baseline
   pass, for each target and every read s >= start_s (each later attention input, MLP input, the head):
     alpha_mean, alpha_median, alpha_p99, alpha_max: the target's softmax weight over tokens;
     frac_ge_0_1, frac_ge_0_01, frac_ge_0_001, frac_ge_1e_4, frac_ge_1e_5: fraction of tokens with alpha >= threshold;
     share_mean, share_median: per token ||alpha_t v_t|| / ||h|| (h the read's output; > 1 means the other sources cancel
     part of it); rms_ratio: per-token RMS(v_t) / median RMS of the other available sources, mean over tokens.
   Variants (final CE each), applied at every read s >= start_s, every other read untouched:
     remove        the target is left out of the softmax (renormalised over the rest; depthmed's skip, one source only);
     rescale_med   the target's value v_t is rescaled per token to the median RMS of the other available sources at that
                   read (direction kept); its key is unchanged (keys are RMS-normalised, so scale-free);
     zero_value    the target keeps its key and so its softmax mass, its value is zero (content gone). Every read here feeds
                   an RMSNorm (the branch pre-norms ln_1/ln_2, the final norm), and zero_value's read is remove's read times
                   the per-token scalar 1 - alpha_t, so the two agree up to the norm's eps;
     remove_at:R   the target left out of read R only (each read R >= start_s in turn), every other read untouched;
     zero_dim      the target's value with its massive dimension (the slot's argmax |activation| dimension) zero, key
                   unchanged.
   Targets: attnres: the sublayer outputs named on the command line (slot 1 + s; start_s = s + 1, its first read).
   attnres_block: "auto" picks the block slot holding the largest |activation| / RMS after layer --after (1-based, 44),
   start_s = 2 * after (reads after that layer); or a slot index given on the command line (the sep twin's control).
B. Embedding split (final CE each): no_embedding_head (the head's mix drops slot 0, sublayer reads keep it);
   no_embedding_reads (every sublayer read with more than one source drops it, i.e. all but layer 1's attention, which
   has only the embedding; the head keeps it); no_embedding (both: attnres_checkup.py's no_embedding).
   Recorded per read: emb_alpha_mean, emb_frac_ge_0.1.
Also recorded: per sublayer write f: RMS, massive = max|f| / RMS(f over all tokens), its dimension; per slot after the last
layer (attnres_block: after every layer): mean per-token RMS, massive and its dimension.

    source scripts/della/pls_env.sh
    JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/attnres_followup.py CKPT CONFIG.json OUT.json MODEL_NAME TARGETS [--after 44]
TARGETS: attnres: comma list of 1-based "LAYER.mlp" / "LAYER.attn" (e.g. 44.mlp,47.mlp); attnres_block: auto or a slot.
"""
import argparse
import dataclasses
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.checkpoint import load_checkpoint  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("config")
ap.add_argument("out")
ap.add_argument("name")
ap.add_argument("targets")
ap.add_argument("--after", type=int, default=44)
ap.add_argument("--b", type=int, default=16)
ap.add_argument("--t", type=int, default=1024)
args = ap.parse_args()
B, T = args.b, args.t
Vocab = Axis("vocab", 128256)
GROUP = 12

wcfg = json.load(open(args.config))["model"]
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
ARCH, L = cfg.depth_arch, cfg.num_layers
BLOCK = ARCH == "attnres_block"
assert ARCH in ("attnres", "attnres_block") and L == 48, (ARCH, L)
if BLOCK:
    assert (2 * L) // cfg.attnres_blocks == GROUP, cfg.attnres_blocks
S = 1 + (2 * L) // GROUP if BLOCK else 2 * L + 1

tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
t0 = time.time()
template = eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0))
m = load_checkpoint(template, args.ckpt, subpath="model")
assert m.aux_adapters is None
print(f"loaded {args.ckpt} ({ARCH}) in {time.time() - t0:.0f}s", flush=True)

import tensorstore as ts  # noqa: E402

VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok_all = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)


def make_batch(tok):
    """attnres_checkup.make_batch: documents isolated by segment ids, next-token targets and their validity."""
    b = tok.shape[0]
    eos = np.roll(tok, 1, axis=1) == 128001
    eos[:, 0] = False
    seg = np.cumsum(eos, axis=1)
    Batch, Pos = Axis("batch", b), Axis("position", T)
    mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg, jnp.int32), (Batch, Pos)))
    tgt = np.roll(tok, -1, axis=1)
    valid = np.ones((b, T), bool)
    valid[:, -1] = False
    valid &= np.roll(seg, -1, axis=1) == seg
    return dict(tok=tok, Batch=Batch, Pos=Pos, mask=mask, tgt=jnp.asarray(tgt), valid=jnp.asarray(valid), n_valid=int(valid.sum()))


tr = m.transformer
Embed, Sub = cfg.Embed, Axis("sub", 2)
blocks = [tr.layer(i) for i in range(L)]
W_main = m.get_lm_head().rearrange((Vocab, Embed)).array.astype(jnp.float32)


def qvec(s):
    if s == 2 * L:
        return tr._final_query().array.astype(jnp.float32)
    lay = blocks[s // 2]
    return (lay.query[Sub, s % 2].astype(jnp.float32) * lay.key_gain[Sub, s % 2].astype(jnp.float32)).array


Q = [qvec(s) for s in range(2 * L + 1)]


def read_name(s):
    return "head" if s == 2 * L else f"L{s // 2 + 1}.{('attn', 'mlp')[s % 2]}"


# ---- jitted pieces --------------------------------------------------------------------------------------------------
@jax.jit
def put(V, R, i, f):
    r = jax.lax.rsqrt(jnp.mean(f * f, -1) + 1e-6)
    return V.at[i].set(f), R.at[i].set(r)


@jax.jit
def add(V, R, i, f):
    v = V[i] + f
    r = jax.lax.rsqrt(jnp.mean(v * v, -1) + 1e-6)
    return V.at[i].set(v), R.at[i].set(r)


@jax.jit
def mix(V, R, q, allowed, tgt, vmode, dim):
    """Softmax over the allowed slots (logits from the original keys). vmode 0: values as they are; 1: the tgt slot's value
    zero; 2: the tgt slot's value rescaled per token to the median RMS of the other allowed slots; 3: the tgt slot's value
    with dimension dim zero. Returns h, w (S, b, p)."""
    lg = jnp.einsum("sbpe,e->sbp", V, q) * R
    lg = jnp.where(allowed[:, None, None], lg, -jnp.inf)
    w = jax.nn.softmax(lg, axis=0)
    rms = 1.0 / jnp.where(R > 0, R, 1.0)
    other = allowed & ~tgt
    med = jnp.nanmedian(jnp.where(other[:, None, None], rms, jnp.nan), axis=0)  # (b, p)
    fac_t = jnp.where(vmode == 1, 0.0, jnp.where(vmode == 2, med / rms, 1.0))  # (S, b, p); only tgt rows used
    fac = jnp.where(tgt[:, None, None], fac_t, 1.0)
    h = jnp.einsum("sbp,sbpe->bpe", w * fac, V)
    drop = jnp.einsum("sbp,sbp->bp", jnp.where(tgt[:, None, None], w, 0.0), V[..., dim])
    h = h.at[..., dim].add(jnp.where(vmode == 3, -drop, 0.0))
    return h, w


@jax.jit
def tstats(V, R, w, h, i, allowed):
    a = w[i]
    c = a[..., None] * V[i]
    share = jnp.linalg.norm(c, axis=-1) / (jnp.linalg.norm(h, axis=-1) + 1e-12)
    rms = 1.0 / jnp.where(R > 0, R, 1.0)
    other = allowed & (jnp.arange(V.shape[0]) != i)
    med = jnp.nanmedian(jnp.where(other[:, None, None], rms, jnp.nan), axis=0)
    af = a.reshape(-1)
    return dict(alpha_mean=af.mean(), alpha_median=jnp.median(af), alpha_p99=jnp.quantile(af, 0.99), alpha_max=af.max(),
                frac_ge_0_1=(af >= 0.1).mean(), frac_ge_0_01=(af >= 0.01).mean(), frac_ge_0_001=(af >= 0.001).mean(),
                frac_ge_1e_4=(af >= 1e-4).mean(), frac_ge_1e_5=(af >= 1e-5).mean(),
                share_mean=share.mean(), share_median=jnp.median(share), rms_ratio=(rms[i] / med).mean())


@jax.jit
def massive(f):
    """mean per-token RMS, max |f| / RMS over all tokens, the dimension holding the max."""
    a = jnp.abs(f)
    rms_all = jnp.sqrt(jnp.mean(f * f))
    return jnp.sqrt(jnp.mean(f * f, -1)).mean(), a.max() / rms_all, jnp.argmax(a.max(axis=tuple(range(f.ndim - 1))))


@eqx.filter_jit
def ce_acc(h, norm, W, tgt, valid):
    Pos1 = Axis("position", h.shape[1])

    def one(xs):
        hb, tb, vb = xs
        z = norm(hax.named(hb, (Pos1, Embed))).array.astype(jnp.float32)
        logits = z @ W.T
        lse = jax.nn.logsumexp(logits, -1)
        ll = jnp.take_along_axis(logits, tb[:, None], -1)[:, 0]
        return jnp.sum((lse - ll) * vb), jnp.sum((jnp.argmax(logits, -1) == tb) * vb)

    ce, ok = jax.lax.map(one, (h, tgt, valid.astype(jnp.float32)))
    n = jnp.sum(valid)
    return jnp.sum(ce) / n, jnp.sum(ok) / n


def make_branch(bt):
    @eqx.filter_jit
    def branch(layer, j, x):
        return layer.branch(j, hax.named(x, (bt["Batch"], bt["Pos"], Embed)), bt["mask"], key=None, pos_ids=None).rearrange((bt["Batch"], bt["Pos"], Embed)).array

    return branch


# ---- policies: (allowed, tgt one-hot, vmode) per read ---------------------------------------------------------------
def policy(variant, target=None, at=None):
    """target: (slot, start_s, dim). variant: baseline | remove | rescale_med | zero_value | zero_dim | no_embedding_head |
    no_embedding_reads | no_embedding."""

    def p(s, n):
        av = np.zeros(S, bool)
        av[:n] = True
        tg = np.zeros(S, bool)
        vmode, dim = 0, 0
        if variant in ("remove", "rescale_med", "zero_value", "zero_dim", "remove_at"):
            slot, start, dim = target
            if variant == "remove_at":
                if s == at and av[slot]:
                    av[slot] = False
            elif s >= start and av[slot]:
                if variant == "remove":
                    av[slot] = False
                else:
                    tg[slot] = True
                    vmode = {"zero_value": 1, "rescale_med": 2, "zero_dim": 3}[variant]
        elif variant in ("no_embedding_head", "no_embedding"):
            if s == 2 * L:
                av[0] = False
        if variant in ("no_embedding_reads", "no_embedding") and s < 2 * L and n > 1:
            av[0] = False
        assert av.any()
        return av, tg, vmode, dim

    return p


def forward(bt, pol, record=None):
    """record: list of targets (slot, start_s) to record at every read, or None (no records)."""
    b = bt["tok"].shape[0]
    branch = make_branch(bt)
    x = m.embeddings.embed(hax.named(jnp.asarray(bt["tok"], jnp.int32), (bt["Batch"], bt["Pos"]))).rearrange((bt["Batch"], bt["Pos"], Embed)).array.astype(jnp.float32)
    V = jnp.zeros((S, b, T, x.shape[-1]), jnp.float32)
    R = jnp.zeros((S, b, T), jnp.float32)
    V, R = put(V, R, 0, x)
    n = 1
    rec = dict(reads=[], writes=[], slots_after_layer=[]) if record is not None else None

    def do_read(s, n, V, R):
        av, tg, vm, dim = pol(s, n)
        h, w = mix(V, R, Q[s], jnp.asarray(av), jnp.asarray(tg), vm, dim)
        if record is not None:
            r = dict(read=read_name(s), s=s, n_sources=n, emb_alpha_mean=float(w[0].mean()), emb_frac_ge_0_1=float((w[0] >= 0.1).mean()))
            for slot, start in record:
                if s >= start and slot < n:
                    r[f"slot{slot}"] = {k: float(v) for k, v in tstats(V, R, w, h, slot, jnp.asarray(av)).items()}
            rec["reads"].append(r)
        return h

    for i in range(L):
        t1 = time.time()
        for j in range(2):
            s = 2 * i + j
            h = do_read(s, n, V, R)
            f = branch(blocks[i], j, h)
            if record is not None:
                rms, mr, md = massive(f)
                rec["writes"].append(dict(write=read_name(s), rms=float(rms), massive=float(mr), massive_dim=int(md), in_rms=float(jnp.sqrt(jnp.mean(h * h, -1)).mean())))
            if BLOCK:
                slot = 1 + s // GROUP
                V, R = put(V, R, slot, f) if s % GROUP == 0 else add(V, R, slot, f)
                n = slot + 1
            else:
                V, R = put(V, R, 1 + s, f)
                n = s + 2
        if record is not None and (BLOCK or i == L - 1):
            st = [massive(V[k]) for k in range(n)]
            rec["slots_after_layer"].append(dict(layer=i + 1, rms=[float(a[0]) for a in st], massive=[float(a[1]) for a in st], massive_dim=[int(a[2]) for a in st]))
        if record is not None and (i % 8 == 7 or i == L - 1):
            print(f"  layer {i + 1} | {time.time() - t1:.1f}s", flush=True)
    h = do_read(2 * L, n, V, R)
    return h, rec


def final_ce(h, bt):
    ce, acc = ce_acc(h, tr.norm, W_main, bt["tgt"], bt["valid"])
    return float(ce), float(acc)


bt = make_batch(tok_all)
out = dict(model=args.name, arch=ARCH, ckpt=args.ckpt, text=f"paloma c4_en validation, first {B} x {T} tokens (documents isolated by segment ids)",
           tokens=bt["n_valid"], n_layers=L, script="scripts/della/attnres_followup.py")
OUT_TMP = args.out + ".tmp"


def dump():
    json.dump(out, open(OUT_TMP, "w"), indent=1)
    os.replace(OUT_TMP, args.out)


# targets
if BLOCK:
    if args.targets == "auto":
        _, rec0 = forward(bt, policy("baseline"), record=[])
        row = rec0["slots_after_layer"][args.after - 1]
        assert row["layer"] == args.after
        slot = 1 + int(np.argmax(row["massive"][1:]))
        out["auto_pick"] = dict(after_layer=args.after, rule="block slot with the largest max|v|/RMS after that layer", slot=slot, row=row)
    else:
        slot = int(args.targets)
    targets = {f"block{slot}": (slot, 2 * args.after)}
else:
    targets = {}
    for t in args.targets.split(","):
        lay, kind = t.split(".")
        s = 2 * (int(lay) - 1) + ("attn", "mlp").index(kind)
        targets[f"L{lay}.{kind}"] = (1 + s, s + 1)
out["targets"] = {k: dict(slot=v[0], start_read=read_name(v[1]), start_s=v[1]) for k, v in targets.items()}
print("targets", out["targets"], flush=True)

t0 = time.time()
h, rec = forward(bt, policy("baseline"), record=list(targets.values()))
out["baseline"] = dict(zip(("ce", "acc"), final_ce(h, bt)))
out["records"] = rec
row = rec["slots_after_layer"][args.after - 1 if BLOCK else -1]  # Full: slots never change once written
for k, (slot, start) in list(targets.items()):
    targets[k] = (slot, start, int(row["massive_dim"][slot]))
    out["targets"][k].update(massive_dim=targets[k][2], massive=row["massive"][slot], rms=row["rms"][slot],
                             median_slot_rms=float(np.median(row["rms"])), at_layer=row["layer"])
print("targets", out["targets"], flush=True)
print(f"baseline ce {out['baseline']['ce']:.4f} | records {time.time() - t0:.0f}s", flush=True)
dump()

out["variants"] = {}
runs = [(f"{k}:{v}", v, tgt, None) for k, tgt in targets.items() for v in ("remove", "rescale_med", "zero_value", "zero_dim")]
runs += [(v, v, None, None) for v in ("no_embedding_head", "no_embedding_reads", "no_embedding")]
runs += [(f"{k}:remove_at:{read_name(r)}", "remove_at", tgt, r) for k, tgt in targets.items() for r in range(tgt[1], 2 * L + 1)]
for key, v, tgt, at in runs:
    t1 = time.time()
    h, _ = forward(bt, policy(v, tgt, at))
    ce, acc = final_ce(h, bt)
    out["variants"][key] = dict(ce=ce, acc=acc)
    dump()
    print(f"{key}: ce {ce:.4f} acc {acc:.4f} | {time.time() - t1:.0f}s", flush=True)
print("wrote", args.out, flush=True)
