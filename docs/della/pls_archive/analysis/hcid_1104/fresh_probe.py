"""Is a layer's next-token information absent, masked or hidden? Fresh linear probes on one checkpoint's state at chosen layers. Targets: the K most frequent next tokens of the training windows plus one class for the rest. Train on N windows of FineWeb-Edu 10B, test on 16 other windows. Inputs, each normalised per token as a readout does (RMS over the input):
  sum: the stream sum, what the trained probes read.
  streams: all streams side by side (hc designs); what a probe could read if nothing were hidden.
  sum-res: the sum minus the mean of its token type, with a bigram prior added to the logits; tells whether the token-identity part masks the context part.
Adam, full-rank linear map, minibatch 8,192, weight decay 1e-4, best test CE over the epochs is reported with the epoch count (the test set only stops training; differences under 0.02 nats are not read).
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY fresh_probe.py TAG"""
import dataclasses, glob, json, os, sys, time
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, jmp, numpy as np, optax
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
from experiments.references.depth_arch_qwen3 import _hc_constants, _rms, _hc_res, _static
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load
tag = sys.argv[1]
LAYERS = [int(v) for v in os.environ.get("PROBE_LAYERS", "12").split(",")]
N_TR, N_TE, KV = int(os.environ.get("WS_N", 128)), 16, int(os.environ.get("TOP_K", 4096))
EPOCHS = int(os.environ.get("EPOCHS", 16))
B, T = 1, 4096
ids_tr, ids_te = load(B, T, N_TR, seed=21), load(B, T, N_TE, seed=22)
Batch, Pos = Axis("batch", B), Axis("position", T)
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
T_ = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + S_
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
mp = jmp.get_policy("p=f32,c=bfloat16")
f32 = lambda a: a.astype(jnp.float32)
ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
model = mp.cast_to_compute(load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model"))
is_hc = cfg.depth_arch == "hc"
Embed = cfg.Embed
if is_hc:
    tr, m_, rm = model.transformer, cfg.hc_streams, _hc_res(cfg)
    static = _static(cfg)
    @eqx.filter_jit
    def step(lay, H, attn, li, j):
        q = lay.hyper[j]
        c_read, c_res, c_write = _hc_constants(m_, 2 * li + j)
        Hn = f32(_rms(H, Embed)); sc = f32(q.scale)
        if static:
            read = c_read + f32(q.read); res = c_res + f32(q.res); write = c_write + f32(q.write)
        else:
            read = c_read + f32(q.read) + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(q.w_read), axis=Embed))
            write = c_write + f32(q.write) + sc["hp_scale", 1] * hax.tanh(hax.dot(Hn, f32(q.w_write), axis=Embed).rename({"stream": "stream_out"}))
            res = c_res + f32(q.res) + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(q.w_res), axis=Embed))
        if rm == "identity":
            res = c_res
        elif rm == "scalar":
            res = (hax.sum(c_res * (c_res + f32(q.res))) / m_) * c_res
        else:
            assert rm == "learned", rm
        xr = hax.dot(read.astype(H.dtype), H, axis="stream")
        f = lay.branch(j, xr, attn, key=None, pos_ids=None)
        upd = write.astype(H.dtype) * f.broadcast_axis(write.resolve_axis("stream_out"))
        return (hax.dot(res.astype(H.dtype), H, axis="stream") + upd).rename({"stream_out": "stream"})
    def states(tok, attn):
        H = hax.stack(Axis("stream", m_), [model.embeddings.embed(tok)] * m_); out = {}
        for li in range(max(LAYERS)):
            lay = tr.layer(li)
            for j in range(2):
                H = step(lay, H, attn, jnp.asarray(li), j)
            if li + 1 in LAYERS:
                out[li + 1] = np.asarray(H.rearrange(("stream", Batch, Pos, Embed)).array.astype(jnp.float32), np.float32)[:, 0]   # (stream, tokens, embed)
        return out
else:
    @eqx.filter_jit
    def fwd(m, tok, attn):
        outs = m.layer_outputs(tok, attn, key=None)
        return outs.rearrange((m.transformer.layers.Block.name, Batch, Pos, Embed)).array.astype(jnp.float32)
    def states(tok, attn):
        s = np.asarray(fwd(model, tok, attn), np.float32)[:, 0]
        return {l: s[l - 1][None] for l in LAYERS}
def collect(ids, n):
    F = {l: [] for l in LAYERS}; TK = []
    for b in range(n):
        tok = ids[b * B * T:][: B * T].reshape(B, T)
        seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
        st = states(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), AttentionMask.causal().with_segment_ids(seg))
        for l in LAYERS:
            F[l].append(st[l][:, :-1].copy())   # the last position has no target
        TK.append(tok[0])
    return {l: np.concatenate(v, 1) for l, v in F.items()}, np.stack(TK)
t0 = time.time()
Ftr, Ttr = collect(ids_tr, N_TR); Fte, Tte = collect(ids_te, N_TE)
print(f"{tag} {ck.rsplit('/', 1)[1]}: features in {time.time() - t0:.0f} s, train tokens {Ftr[LAYERS[0]].shape[1]}", flush=True)
cur_tr, nxt_tr = Ttr[:, :-1].ravel(), Ttr[:, 1:].ravel(); cur_te, nxt_te = Tte[:, :-1].ravel(), Tte[:, 1:].ravel()
vals, cnt = np.unique(nxt_tr, return_counts=True)
top = vals[np.argsort(-cnt)[:KV]]
remap = np.full(128256, KV, np.int32); remap[top] = np.arange(KV)
y_tr, y_te = remap[nxt_tr], remap[nxt_te]
print(f"  top {KV} next tokens cover {(y_te < KV).mean():.3f} of the test targets")
def rmsnorm(a):
    s = np.abs(a).max(-1, keepdims=True); a = a / np.maximum(s, 1e-30)
    return a / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-12)
def fit(xtr, ytr, xte, yte, ptr=None, pte=None, name=""):
    """Linear softmax, K + 1 classes; ptr / pte: fixed prior logits per token or None."""
    d, n = xtr.shape[1], len(ytr)
    W = jnp.zeros((d, KV + 1)); b = jnp.zeros(KV + 1)
    sched = optax.cosine_decay_schedule(3e-3, EPOCHS * (n // 8192))
    opt = optax.adamw(sched, weight_decay=1e-4); st = opt.init((W, b))
    @jax.jit
    def upd(W, b, st, x, y, p):
        def loss(wb):
            lg = x @ wb[0] + wb[1] + (p if p is not None else 0.0)
            return optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()
        l, g = jax.value_and_grad(loss)((W, b))
        u, st = opt.update(g, st, (W, b)); (W, b) = optax.apply_updates((W, b), u)
        return W, b, st, l
    @jax.jit
    def ev(W, b, x, y, p):
        lg = x @ W + b + (p if p is not None else 0.0)
        return optax.softmax_cross_entropy_with_integer_labels(lg, y).sum()
    rng = np.random.default_rng(0); best = (9e9, -1)
    for ep in range(EPOCHS):
        perm = rng.permutation(n)
        for i in range(0, n - 8191, 8192):
            k = perm[i:i + 8192]
            W, b, st, _ = upd(W, b, st, jnp.asarray(xtr[k]), jnp.asarray(ytr[k]), None if ptr is None else jnp.asarray(ptr[k]))
        tot = 0.0
        for i in range(0, len(yte), 16384):
            tot += float(ev(W, b, jnp.asarray(xte[i:i + 16384]), jnp.asarray(yte[i:i + 16384]), None if pte is None else jnp.asarray(pte[i:i + 16384])))
        ce = tot / len(yte)
        if ce < best[0]:
            best = (ce, ep)
    return best
# bigram prior on the K + 1 classes, add-0.5 smoothing, only for the token types seen in training
Ncur = 128256; types, inv = np.unique(cur_tr, return_inverse=True)
ty_tr = inv; lut = np.full(Ncur, len(types), np.int64); lut[types] = np.arange(len(types)); ty_te = lut[cur_te]
keep_types = np.argsort(-np.bincount(ty_tr))[:6000]; slot = np.full(len(types) + 1, 6000, np.int64); slot[keep_types] = np.arange(6000)
Ct = np.zeros((6001, KV + 1), np.float32); np.add.at(Ct, (slot[ty_tr], y_tr), 1.0)
Pr = np.log(Ct + 0.5 / (KV + 1) * 50) ; Pr -= np.log((Ct + 0.5 / (KV + 1) * 50).sum(1, keepdims=True))
ptr, pte = Pr[slot[ty_tr]], Pr[slot[ty_te]]
print(f"  bigram prior alone: test CE {float((-pte[np.arange(len(y_te)), y_te]).mean()):.3f}", flush=True)
for l in LAYERS:
    fs, ft = Ftr[l], Fte[l]
    S_tr, S_te = rmsnorm(fs.sum(0)), rmsnorm(ft.sum(0))
    r = {}
    r["sum"] = fit(S_tr, y_tr, S_te, y_te)
    # the sum minus the mean of its token type, plus the bigram prior
    sums = np.zeros((6001, S_tr.shape[1]), np.float32); np.add.at(sums, slot[ty_tr], S_tr); nn = np.bincount(slot[ty_tr], minlength=6001)
    mu = sums / np.maximum(nn, 1)[:, None]
    r["sum-res"] = fit(S_tr - mu[slot[ty_tr]], y_tr, S_te - mu[slot[ty_te]], y_te, ptr, pte)
    r["sum+prior"] = fit(S_tr, y_tr, S_te, y_te, ptr, pte)
    if fs.shape[0] > 1:
        c_tr = rmsnorm(np.moveaxis(fs, 0, 1).reshape(fs.shape[1], -1)); c_te = rmsnorm(np.moveaxis(ft, 0, 1).reshape(ft.shape[1], -1))
        r["streams"] = fit(c_tr, y_tr, c_te, y_te)
        r["streams+prior"] = fit(c_tr, y_tr, c_te, y_te, ptr, pte)
    print(f"  L{l}: " + "  ".join(f"{k} {v[0]:.3f} (epoch {v[1]})" for k, v in r.items()), flush=True)
print("time", round(time.time() - t0), "s", flush=True)
