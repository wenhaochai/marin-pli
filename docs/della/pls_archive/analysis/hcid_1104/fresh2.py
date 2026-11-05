"""Does the old part of the sum mask the recent part? Fresh linear probes (as fresh_probe.py) at layer L on parts of the exact source split of the sum (kernel.py): old = the embedding and the first layer's two writes, recent = everything after. Inputs, RMS-normalised per token: sum; recent alone; old alone; old and recent side by side, each normalised alone. Also the same split for the streams: streams side by side (is the information hidden in the differences?).
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY fresh2.py TAG"""
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
from experiments.references.depth_arch_qwen3 import _hc_constants, _rms, _hc_res, _static, _sum_gain, HC_MOVE_BETA, HC_MOVE_SHIFT
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load
tag = sys.argv[1]
LAYERS = [int(v) for v in os.environ.get("PROBE_LAYERS", "12").split(",")]
N_TR, N_TE, KV = int(os.environ.get("WS_N", 192)), 16, int(os.environ.get("TOP_K", 1024))
EPOCHS = int(os.environ.get("EPOCHS", 12))
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
tr, m_, Embed, rm = model.transformer, cfg.hc_streams, cfg.Embed, _hc_res(cfg)
static = _static(cfg)
S0, So0 = Axis("stream", m_), Axis("stream_out", m_)
STATE_F32 = bool(getattr(cfg, "depth_hc_state_f32", False))
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
    elif rm == "sumshrink":
        res = c_res - ((1.0 - _sum_gain(cfg, 2 * li + j)) / m_) * hax.named(jnp.ones((m_, m_)), (S0, So0))
    elif rm == "move":
        pos = hax.named(jax.nn.softplus(HC_MOVE_BETA * (res.array - HC_MOVE_SHIFT)), res.axes) / HC_MOVE_BETA + 1e-6 * c_res
        res = pos / hax.sum(pos, axis=So0).broadcast_axis(So0)
    elif rm == "sum":
        res = res - ((hax.sum(res, axis=So0) - 1.0) / m_).broadcast_axis(So0)
    else:
        assert rm == "learned", rm
    xr = hax.dot(read.astype(H.dtype), H, axis="stream").astype(jnp.bfloat16)
    f = lay.branch(j, xr, attn, key=None, pos_ids=None)
    upd = write.astype(H.dtype) * f.astype(H.dtype).broadcast_axis(So0)
    H2 = (hax.dot(res.astype(H.dtype), H, axis="stream") + upd).rename({"stream_out": "stream"})
    A = res.broadcast_axis((Batch, Pos)).rearrange((Batch, Pos, S0, So0)).array if res.ndim == 2 else res.rearrange((Batch, Pos, S0, So0)).array
    W = write.broadcast_axis((Batch, Pos)).rearrange((Batch, Pos, So0)).array if write.ndim == 1 else write.rearrange((Batch, Pos, So0)).array
    return H2, A, W, f.rearrange((Batch, Pos, Embed)).array.astype(jnp.float32)
def collect(ids, n):
    F = {l: dict(s=[], old=[], st=[]) for l in LAYERS}; TK = []
    for b in range(n):
        tok = ids[b * B * T:][: B * T].reshape(B, T)
        seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
        attn = AttentionMask.causal().with_segment_ids(seg)
        toks = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))
        e = model.embeddings.embed(toks)
        H = hax.stack(S0, [e.astype(jnp.float32 if STATE_F32 else e.dtype)] * m_)
        e_arr = e.rearrange((Batch, Pos, Embed)).array.astype(jnp.float32)
        c = [jnp.ones((B, T, m_), jnp.float32)]; Fs = [e_arr]
        for li in range(max(LAYERS)):
            lay = tr.layer(li)
            for j in range(2):
                H, A, W, f = step(lay, H, attn, jnp.asarray(li), j)
                c = [jnp.einsum("btio,bti->bto", A, ci) for ci in c]
                if li == 0:
                    c = c + [W]; Fs.append(f)   # only the old sources are tracked: the embedding and the first layer's two writes
            if li + 1 in LAYERS:
                old = sum(ci.sum(-1)[..., None] * Fi for ci, Fi in zip(c, Fs))
                Hs = H.rearrange((S0, Batch, Pos, Embed)).array.astype(jnp.float32)
                F[li + 1]["s"].append(np.asarray(Hs.sum(0), np.float32)[0, :-1]); F[li + 1]["old"].append(np.asarray(old, np.float32)[0, :-1])
                F[li + 1]["st"].append(np.asarray(Hs, np.float32)[:, 0, :-1])
        TK.append(tok[0])
    return {l: {k: np.concatenate(v, 1 if k == "st" else 0) for k, v in d.items()} for l, d in F.items()}, np.stack(TK)
t0 = time.time()
Ftr, Ttr = collect(ids_tr, N_TR); Fte, Tte = collect(ids_te, N_TE)
print(f"{tag} {ck.rsplit('/', 1)[1]}: features in {time.time() - t0:.0f} s, train tokens {Ttr[:, :-1].size}", flush=True)
nxt_tr, nxt_te = Ttr[:, 1:].ravel(), Tte[:, 1:].ravel()
vals, cnt = np.unique(nxt_tr, return_counts=True)
top = vals[np.argsort(-cnt)[:KV]]
remap = np.full(128256, KV, np.int32); remap[top] = np.arange(KV)
y_tr, y_te = remap[nxt_tr], remap[nxt_te]
print(f"  top {KV} next tokens cover {(y_te < KV).mean():.3f} of the test targets; class prior CE {float(-np.log(np.bincount(y_tr, minlength=KV + 1)[y_te] / len(y_tr)).mean()):.3f}", flush=True)
def rmsnorm(a):
    s = np.abs(a).max(-1, keepdims=True); a = a / np.maximum(s, 1e-30)
    return a / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-12)
def fit(xtr, ytr, xte, yte):
    d, n = xtr.shape[1], len(ytr)
    W = jnp.zeros((d, KV + 1), jnp.float32); b = jnp.zeros(KV + 1, jnp.float32)
    sched = optax.cosine_decay_schedule(float(os.environ.get("LR", 1e-3)), EPOCHS * (n // 8192))
    opt = optax.adamw(sched, weight_decay=float(os.environ.get("WD", 1e-3))); st = opt.init((W, b))
    @jax.jit
    def upd(W, b, st, x, y):
        def loss(wb):
            return optax.softmax_cross_entropy_with_integer_labels(x @ wb[0] + wb[1], y).mean()
        l, g = jax.value_and_grad(loss)((W, b))
        u, st = opt.update(g, st, (W, b)); (W, b) = optax.apply_updates((W, b), u)
        return W, b, st, l
    @jax.jit
    def ev(W, b, x, y):
        return optax.softmax_cross_entropy_with_integer_labels(x @ W + b, y).sum()
    rng = np.random.default_rng(0); best = (9e9, -1)
    for ep in range(EPOCHS):
        perm = rng.permutation(n)
        for i in range(0, n - 8191, 8192):
            k = perm[i:i + 8192]
            W, b, st, _ = upd(W, b, st, jnp.asarray(xtr[k]), jnp.asarray(ytr[k]))
        tot = sum(float(ev(W, b, jnp.asarray(xte[i:i + 16384]), jnp.asarray(yte[i:i + 16384]))) for i in range(0, len(yte), 16384))
        ce = tot / len(yte)
        if ce < best[0]:
            best = (ce, ep)
    return best
for l in LAYERS:
    a, c = Ftr[l], Fte[l]
    s_tr, s_te, o_tr, o_te = a["s"], c["s"], a["old"], c["old"]
    r_tr, r_te = s_tr - o_tr, s_te - o_te
    r = {}
    r["sum"] = fit(rmsnorm(s_tr), y_tr, rmsnorm(s_te), y_te)
    r["recent"] = fit(rmsnorm(r_tr), y_tr, rmsnorm(r_te), y_te)
    r["old"] = fit(rmsnorm(o_tr), y_tr, rmsnorm(o_te), y_te)
    r["old|recent"] = fit(np.concatenate([rmsnorm(o_tr), rmsnorm(r_tr)], 1), y_tr, np.concatenate([rmsnorm(o_te), rmsnorm(r_te)], 1), y_te)
    if m_ > 1:
        st_tr, st_te = np.moveaxis(a["st"], 0, 1), np.moveaxis(c["st"], 0, 1)
        r["streams"] = fit(rmsnorm(st_tr.reshape(st_tr.shape[0], -1)), y_tr, rmsnorm(st_te.reshape(st_te.shape[0], -1)), y_te)
    nr = lambda x: np.sqrt(((x / np.maximum(np.abs(x).max(-1, keepdims=True), 1e-30)) ** 2).sum(-1)) * np.abs(x).max(-1)
    print(f"  L{l}: |old| / |sum| median {np.median(nr(o_te) / nr(s_te)):.3f}, |recent| / |sum| median {np.median(nr(r_te) / nr(s_te)):.3f}; fresh probes: " + "  ".join(f"{k} {v[0]:.3f} ({v[1]})" for k, v in r.items()), flush=True)
print("time", round(time.time() - t0), "s", flush=True)
