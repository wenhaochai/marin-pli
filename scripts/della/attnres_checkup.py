"""Full check-up of our 48-layer AttnRes models (attnres_block and attnres), one-to-one with the Kimi K3 check-up
(owner popup 2026-10-19, "按方案做", "越全面越好"). Read-only: loads a finished checkpoint, trains nothing.

Sources (depth_arch_qwen3._attnres_outputs / _attnres_scan): slot 0 is the embedding. attnres: slot 1 + s holds the
output of sublayer s (s = 2 layer + {0 attention, 1 MLP}), 2L + 1 = 97 slots. attnres_block: slot 1 + b holds block b's
running sum (12 sublayers = 6 layers per block), so a block's slot holds its running sum while it fills and its finished
sum after. Every mix is sum_i w_i v_i with w = softmax_i((v_i . q) / RMS(v_i)), q = query * key_gain (_mix_buf's form;
_mix's _rms(v) . q is the same number up to float32 rounding).

Per layer k (0-based, after the layer):
  readings through the final norm + lm_head (CE, top-1 acc over next-token targets, document boundaries left out):
    next  the mix the next layer's attention reads (after the last layer the head's mix, so next = exit there)
    exit  the head's mix (final_query * final_key_gain) over the sources after layer k
    sum   the plain sum of the sources after layer k
    own   (models with separate heads / probes) layer k's own aux norm + aux head on the next state, as the training
          readout reads it (_layer, _head); layer 47 is the model's own head, so own = exit there
  weights (mean over all tokens): attn_in_probs, mlp_in_probs (layer k's two input mixes), exit_probs; per-token
    entropies (nats); the exit weight on the embedding by position bin (sequence position and position in document);
    attnres also the weights summed per group of 12 sublayers (*_grouped: [embedding, group 1, ..., current group])
  writes: per-token RMS of the attention and MLP outputs; cos(write, output query); per-token RMS of each source
Ablations (one full forward per variant, final CE / acc through the head):
  baseline; uniform_all (equal weights over the available sources at every mix incl. the head); sum_all (plain sum);
  no_embedding (embedding weight 0, softmax renormalised over the rest; layer 0's attention keeps it: it is the only
  source there); recent_only (block: the current block's running sum + the most recently finished block; attnres: the
  outputs of the current 12-sublayer group and the previous one; in group 1 the embedding stands in as the previous
  block, as in the block model; softmax renormalised over those); uniform_block_b, b = 1..8 (uniform weights only at
  the 12 sublayers of layers 6(b-1)..6b-1).
Static queries: per sublayer and the head, q = query * key_gain: norm, cos with the head's q, cos with the next q.

    source scripts/della/pls_env.sh
    JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/attnres_checkup.py CKPT CONFIG.json OUT.json MODEL_NAME [B T [B_ABL]]
CONFIG.json: {"model": W&B model config}. Text: paloma c4_en validation, the first B*T tokens (default 16 x 1024);
B_ABL: sequences used by the ablations (default B).
"""
import dataclasses
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.checkpoint import load_checkpoint  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

CKPT, CFG, OUT, NAME = sys.argv[1:5]
B = int(sys.argv[5]) if len(sys.argv) > 5 else 16
T = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
B_ABL = int(sys.argv[7]) if len(sys.argv) > 7 else B
Vocab = Axis("vocab", 128256)
GROUP = 12  # sublayers per block (attnres_block) / per group (attnres): 6 layers
POS_BINS = ((0, 16), (16, 64), (64, 256), (256, 1 << 30))

wcfg = json.load(open(CFG))["model"]
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
ARCH, L = cfg.depth_arch, cfg.num_layers
BLOCK = ARCH == "attnres_block"
assert ARCH in ("attnres", "attnres_block") and L == 48, (ARCH, L)
if BLOCK:
    assert (2 * L) // cfg.attnres_blocks == GROUP, cfg.attnres_blocks
S = 1 + (2 * L) // GROUP if BLOCK else 2 * L + 1  # source slots

from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
t0 = time.time()
template = eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0))
m = load_checkpoint(template, CKPT, subpath="model")
assert m.aux_adapters is None
print(f"loaded {CKPT} ({ARCH}) in {time.time() - t0:.0f}s", flush=True)

import tensorstore as ts  # noqa: E402

VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok_all = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)


def make_batch(tok):
    """Mask (documents isolated by segment ids), next-token targets and their validity, position in document."""
    b = tok.shape[0]
    eos = np.roll(tok, 1, axis=1) == 128001
    eos[:, 0] = False
    seg = np.cumsum(eos, axis=1)
    Batch, Pos = Axis("batch", b), Axis("position", T)
    mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg, jnp.int32), (Batch, Pos)))
    tgt = np.roll(tok, -1, axis=1)
    valid = np.ones((b, T), bool)
    valid[:, -1] = False
    valid &= np.roll(seg, -1, axis=1) == seg  # a position whose next token starts a new document is left out
    pos = np.broadcast_to(np.arange(T), (b, T))
    start = np.zeros((b, T), np.int64)
    for r in range(b):
        s0 = 0
        for p in range(T):
            if p > 0 and seg[r, p] != seg[r, p - 1]:
                s0 = p
            start[r, p] = s0
    return dict(tok=tok, Batch=Batch, Pos=Pos, mask=mask, tgt=jnp.asarray(tgt), valid=jnp.asarray(valid), n_valid=int(valid.sum()),
                pos=jnp.asarray(pos), docpos=jnp.asarray(pos - start))


tr = m.transformer
Embed, Sub = cfg.Embed, Axis("sub", 2)
blocks = [tr.layer(i) for i in range(L)]
W_main = m.get_lm_head().rearrange((Vocab, Embed)).array.astype(jnp.float32)
HAS_OWN = m.aux_lm_heads is not None


def qvec(s):
    """Effective query (query * key_gain) of consumer s: sublayer s = 2i + j, or the head (s = 2L)."""
    if s == 2 * L:
        return tr._final_query().array.astype(jnp.float32)
    lay = blocks[s // 2]
    return (lay.query[Sub, s % 2].astype(jnp.float32) * lay.key_gain[Sub, s % 2].astype(jnp.float32)).array


Q = [qvec(s) for s in range(2 * L + 1)]
QOUT = Q[2 * L]


# ---- jitted pieces --------------------------------------------------------------------------------------------------
@jax.jit
def put(V, R, i, f):
    """Slot i := f (with its 1 / RMS)."""
    r = jax.lax.rsqrt(jnp.mean(f * f, -1) + 1e-6)
    return V.at[i].set(f), R.at[i].set(r)


@jax.jit
def add(V, R, i, f):
    """Slot i += f (a block's running sum), 1 / RMS recomputed."""
    v = V[i] + f
    r = jax.lax.rsqrt(jnp.mean(v * v, -1) + 1e-6)
    return V.at[i].set(v), R.at[i].set(r)


@jax.jit
def mix(V, R, q, allowed, kind):
    """kind 0: softmax over the allowed slots; 1: uniform over them; 2: weight 1 each (plain sum). Returns h, w (S, b, p)."""
    lg = jnp.einsum("sbpe,e->sbp", V, q) * R
    lg = jnp.where(allowed[:, None, None], lg, -jnp.inf)
    sm = jax.nn.softmax(lg, axis=0)
    a = jnp.broadcast_to(allowed.astype(jnp.float32)[:, None, None], lg.shape)
    w = jnp.where(kind == 0, sm, jnp.where(kind == 1, a / jnp.sum(a, 0), a))
    return jnp.einsum("sbp,sbpe->bpe", w, V), w


@jax.jit
def wstats(w, pos, docpos):
    """Mean weight per slot, mean per-token entropy, embedding weight by position bin (sequence position, doc position)."""
    ent = -jnp.sum(jnp.where(w > 0, w * jnp.log(jnp.where(w > 0, w, 1.0)), 0.0), 0)
    e = w[0]
    by = jnp.stack([jnp.sum(e * ((pos >= lo) & (pos < hi))) / jnp.sum((pos >= lo) & (pos < hi)) for lo, hi in POS_BINS])  # empty bin: nan
    byd = jnp.stack([jnp.sum(e * ((docpos >= lo) & (docpos < hi))) / jnp.sum((docpos >= lo) & (docpos < hi)) for lo, hi in POS_BINS])
    return w.mean((1, 2)), ent.mean(), by, byd


@jax.jit
def write_stats(f):
    rms = jnp.sqrt(jnp.mean(f * f, -1))
    cos = jnp.einsum("bpe,e->bp", f, QOUT) / (jnp.linalg.norm(f, axis=-1) * jnp.linalg.norm(QOUT) + 1e-12)
    return rms.mean(), cos.mean()


@jax.jit
def source_rms(R):
    """Mean per-token RMS of each slot (1 / R; empty slots 0)."""
    return jnp.mean(jnp.where(R > 0, 1.0 / jnp.where(R > 0, R, 1.0), 0.0), (1, 2))


@eqx.filter_jit
def ce_acc(h, norm, W, tgt, valid):
    """Mean next-token CE and top-1 accuracy of h (b, p, E) through norm and the head W (V, E); one sequence at a time."""
    Pos1 = Axis("position", h.shape[1])

    def one(args):
        hb, tb, vb = args
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


def read(h, bt, norm=None, W=None):
    ce, acc = ce_acc(h, tr.norm if norm is None else norm, W_main if W is None else W, bt["tgt"], bt["valid"])
    return float(ce), float(acc)


# ---- which sources each mix may use ---------------------------------------------------------------------------------
def available(n):
    a = np.zeros(S, bool)
    a[:n] = True
    return a


def recent(s, n):
    """recent_only's slots for consumer s (sublayer s, or the head at s = 2L) with n slots filled."""
    a = np.zeros(S, bool)
    c = s // GROUP
    if BLOCK:
        partial = s % GROUP != 0 and s < 2 * L  # the current block's running sum exists
        a[n - 1] = True  # the running sum, or (no running sum) the most recently finished block / the embedding
        if partial:
            a[n - 2] = True  # the most recently finished block, or the embedding in block 1
    else:
        lo = max(0, GROUP * (c - 1))
        a[1 + lo: 1 + s] = True
        if c == 0:
            a[0] = True
    assert a[:n].sum() == a.sum() > 0
    return a


def policy(variant):
    def p(s, n):
        av = available(n)
        if variant == "baseline":
            return av, 0
        if variant == "uniform_all":
            return av, 1
        if variant == "sum_all":
            return av, 2
        if variant == "no_embedding":
            if n > 1:
                av[0] = False
            return av, 0
        if variant == "recent_only":
            return recent(s, n), 0
        if variant.startswith("uniform_block_"):
            b = int(variant.rsplit("_", 1)[1])
            return av, 1 if s < 2 * L and s // GROUP == b - 1 else 0
        raise ValueError(variant)

    return p


def grouped(probs):
    """attnres: weights per slot -> [embedding, group 1, ..., current group]."""
    out = [probs[0]]
    for i, p in enumerate(probs[1:]):
        g = i // GROUP
        if g + 1 >= len(out):
            out.append(0.0)
        out[g + 1] += p
    return out


# ---- the forward pass -----------------------------------------------------------------------------------------------
def forward(bt, pol, record=False):
    b = bt["tok"].shape[0]
    branch = make_branch(bt)
    x = m.embeddings.embed(hax.named(jnp.asarray(bt["tok"], jnp.int32), (bt["Batch"], bt["Pos"]))).rearrange((bt["Batch"], bt["Pos"], Embed)).array.astype(jnp.float32)
    V = jnp.zeros((S, b, T, x.shape[-1]), jnp.float32)
    R = jnp.zeros((S, b, T), jnp.float32)
    V, R = put(V, R, 0, x)
    n, rows = 1, []
    for i in range(L):
        t1 = time.time()
        rec = dict(layer=i)
        for j in range(2):
            s = 2 * i + j
            allowed, kind = pol(s, n)
            h, w = mix(V, R, Q[s], jnp.asarray(allowed), kind)
            f = branch(blocks[i], j, h)
            if record:
                name = ("attn", "mlp")[j]
                mw, ent, _, _ = wstats(w, bt["pos"], bt["docpos"])
                rec[f"{name}_in_probs"] = np.asarray(mw)[:n].tolist()
                rec[f"{name}_in_entropy"] = float(ent)
                rms, cos = write_stats(f)
                rec[f"{name}_write_rms"], rec[f"{name}_write_cos_out"] = float(rms), float(cos)
            if BLOCK:
                slot = 1 + s // GROUP
                V, R = put(V, R, slot, f) if s % GROUP == 0 else add(V, R, slot, f)
                n = slot + 1
            else:
                V, R = put(V, R, 1 + s, f)
                n = s + 2
        if record:
            av = jnp.asarray(available(n))
            rec["n_sources"] = n
            he, we = mix(V, R, QOUT, av, 0)
            hn = mix(V, R, Q[2 * i + 2], av, 0)[0] if i + 1 < L else he
            hs = mix(V, R, QOUT, av, 2)[0]
            mw, ent, by, byd = wstats(we, bt["pos"], bt["docpos"])
            rec["exit_probs"], rec["exit_entropy"] = np.asarray(mw)[:n].tolist(), float(ent)
            rec["exit_emb_by_pos"], rec["exit_emb_by_docpos"] = [[None if np.isnan(v) else float(v) for v in np.asarray(a)] for a in (by, byd)]  # empty bin: null
            rec["source_rms"] = np.asarray(source_rms(R))[:n].tolist()
            for name, h in (("next", hn), ("exit", he), ("sum", hs)):
                rec[f"{name}_ce"], rec[f"{name}_acc"] = read(h, bt)
            if HAS_OWN:
                if i + 1 < L:
                    rec["own_ce"], rec["own_acc"] = read(hn, bt, m.aux_norms[i], m.aux_lm_heads[i].weight.rearrange((Vocab, Embed)).array.astype(jnp.float32))
                else:
                    rec["own_ce"], rec["own_acc"] = rec["exit_ce"], rec["exit_acc"]
            if not BLOCK:
                for k in ("attn_in", "mlp_in", "exit"):
                    rec[f"{k}_probs_grouped"] = grouped(rec[f"{k}_probs"])
            rows.append(rec)
            own = f" own {rec['own_ce']:.3f}" if HAS_OWN else ""
            print(f"layer {i + 1}: next {rec['next_ce']:.3f} exit {rec['exit_ce']:.3f} sum {rec['sum_ce']:.3f}{own} | n {n} | "
                  f"exit emb {rec['exit_probs'][0]:.3f} | {time.time() - t1:.0f}s", flush=True)
    allowed, kind = pol(2 * L, n)
    h, _ = mix(V, R, QOUT, jnp.asarray(allowed), kind)
    return h, rows


def queries():
    names = [f"L{s // 2}.{('attn', 'mlp')[s % 2]}" for s in range(2 * L)] + ["head"]
    qs = np.stack([np.asarray(q, np.float64) for q in Q])
    nrm = np.linalg.norm(qs, axis=1)
    unit = qs / np.maximum(nrm, 1e-12)[:, None]
    return dict(names=names, norm=nrm.tolist(), cos_out=(unit @ unit[-1]).tolist(), cos_next=[float(unit[s] @ unit[s + 1]) for s in range(2 * L)])


bt = make_batch(tok_all)
out = dict(model=NAME, arch=ARCH, ckpt=CKPT, text=f"paloma c4_en validation, first {B} x {T} tokens (documents isolated by segment ids)",
           tokens=bt["n_valid"], n_layers=L, block_layers=GROUP // 2 if BLOCK else None, queries=queries())
t0 = time.time()
h, rows = forward(bt, policy("baseline"), record=True)
out["layers"] = rows
print(f"records done in {time.time() - t0:.0f}s", flush=True)
json.dump(out, open(OUT, "w"), indent=1)

bta = bt if B_ABL == B else make_batch(tok_all[:B_ABL])
out["ablation_text"] = f"paloma c4_en validation, first {B_ABL} x {T} tokens"
out["ablation_tokens"] = bta["n_valid"]
abl = {}
variants = ["baseline", "uniform_all", "sum_all", "no_embedding", "recent_only"] + [f"uniform_block_{b}" for b in range(1, 9)]
for v in variants:
    t1 = time.time()
    h, _ = forward(bta, policy(v))
    ce, acc = read(h, bta)
    abl[v] = dict(ce=ce, acc=acc)
    out["ablations"] = abl
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"ablation {v}: ce {ce:.4f} acc {acc:.4f} | {time.time() - t1:.0f}s", flush=True)
print("wrote", OUT, "| exit CE at layer 48", round(rows[-1]["exit_ce"], 4), "| baseline ablation", round(abl["baseline"]["ce"], 4))
