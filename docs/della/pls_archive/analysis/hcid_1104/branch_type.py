"""How much of what a branch reads and writes is the current token's type? For each sublayer: x = the branch input (read-weighted streams), F = the branch output, S = the new stream sum, D = the stream differences (streams minus their mean, concatenated). Type share = share of the variance of the RMS-normalised vector explained by the token type (types with at least 8 tokens, unbiased within-type variance). 4 windows of FineWeb-Edu, bfloat16 as training.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY branch_type.py TAG"""
import dataclasses, glob, json, os, sys
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, jmp, numpy as np
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
B, T, N = 1, 4096, int(os.environ.get("WS_N", 4))
LAYERS = [2, 4, 8, 12, 16, 24, 36, 47]
ids = load(B, T, N, seed=7)
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
    Hs = H2.rearrange((S0, Batch, Pos, Embed)).array.astype(jnp.float32)
    return H2, xr.rearrange((Batch, Pos, Embed)).array.astype(jnp.float32), f.rearrange((Batch, Pos, Embed)).array.astype(jnp.float32), Hs
store = {}
TOK = []
for b in range(N):
    tok = ids[b * B * T:][: B * T].reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    attn = AttentionMask.causal().with_segment_ids(seg)
    e = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
    H = hax.stack(S0, [e.astype(jnp.float32 if STATE_F32 else e.dtype)] * m_)
    for li in range(max(LAYERS)):
        lay = tr.layer(li)
        for j in range(2):
            H, x, f, Hs = step(lay, H, attn, jnp.asarray(li), j)
            if li + 1 in LAYERS:
                d = store.setdefault((li + 1, j), dict(x=[], f=[], s=[], d=[]))
                d["x"].append(np.asarray(x, np.float32)[0]); d["f"].append(np.asarray(f, np.float32)[0])
                hs = np.asarray(Hs, np.float32)[:, 0]
                d["s"].append(hs.sum(0)); d["d"].append(np.moveaxis(hs - hs.mean(0, keepdims=True), 0, 1).reshape(T, -1))
    TOK.append(tok[0])
TOK = np.concatenate(TOK)
vals, inv, cnt = np.unique(TOK, return_inverse=True, return_counts=True)
keep = cnt[inv] >= 8; idx = inv[keep]
def rn(a):
    a = np.asarray(a, np.float64); s = np.abs(a).max(-1, keepdims=True); a = a / np.maximum(s, 1e-300)
    return a / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-24)
def type_share(a):
    u = rn(a)[keep]
    mu = u.mean(0); tot = ((u - mu) ** 2).sum(-1).mean()
    if tot < 1e-12:
        return float("nan")
    sums = np.zeros((len(vals), u.shape[1])); np.add.at(sums, idx, u); n = np.bincount(idx, minlength=len(vals))
    means = sums / np.maximum(n, 1)[:, None]
    within = ((u - means[idx]) ** 2).sum(-1).sum() / max(len(u) - (n > 0).sum(), 1)
    return 1 - within / tot
print(f"\n{tag} {ck.rsplit('/', 1)[1]} streams {m_} res {rm}; type share: x (attn, mlp), F (attn, mlp), sum after the layer, differences after the layer; median |F| (attn, mlp), median |x| (attn, mlp)")
for l in LAYERS:
    a, m = store[(l, 0)], store[(l, 1)]
    get = lambda d, k: np.concatenate(d[k], 0)
    nr = lambda v: np.median(np.sqrt(((v / np.maximum(np.abs(v).max(-1, keepdims=True), 1e-30)) ** 2).sum(-1)) * np.abs(v).max(-1))
    dd = get(m, "d")
    print(f"  L{l:<3d} x {type_share(get(a, 'x')):5.2f} {type_share(get(m, 'x')):5.2f}  F {type_share(get(a, 'f')):5.2f} {type_share(get(m, 'f')):5.2f}  S {type_share(get(m, 's')):5.2f}  D {type_share(dd) if m_ > 1 else float('nan'):5.2f}  |F| {nr(get(a, 'f')):8.2f} {nr(get(m, 'f')):8.2f}  |x| {nr(get(a, 'x')):9.2f} {nr(get(m, 'x')):9.2f}", flush=True)
