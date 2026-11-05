"""Which part of the state does a probe read? New sum after a sublayer = kept + surfaced + write, with g the mean row sum of the mixing per token: kept = g S_old (the old sum, scaled), surfaced = (sum of the mixed streams) - g S_old (what moves from the stream differences into the sum), write = Sigma_o w_o F. Same checkpoint, FineWeb-Edu windows, bfloat16; the trained probe of that layer is applied to variants of the state after the layer's last sublayer (MLP): all parts, without each part, each part alone, and "write of both sublayers of this layer alone". A trained probe is tuned to the mixture, so a variant's CE is what this probe can read from that part, not what a probe trained on it could.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY s12_parts.py TAG"""
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
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel, maybe_fused_next_token_loss
from experiments.references.depth_arch_qwen3 import _hc_constants, _rms, _hc_res, _static
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load
tag = sys.argv[1]
B, T, N = 1, 4096, int(os.environ.get("WS_N", 4))
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
LAYERS = [4, 8, 12, 24, 36]
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
    mixed = hax.dot(res.astype(H.dtype), H, axis="stream")
    H2 = (mixed + upd).rename({"stream_out": "stream"})
    S_old, S_mix, W = hax.sum(f32(H), axis="stream"), hax.sum(f32(mixed), axis="stream_out"), hax.sum(f32(upd), axis="stream_out")
    g = hax.mean(hax.sum(res, axis="stream_out"), axis="stream")
    kept = g * S_old
    surf = S_mix - kept
    w2 = hax.sum(f32(write) ** 2, axis="stream_out"); w1 = hax.sum(f32(write), axis="stream_out")
    into_sum = (w1 * w1) / (m_ * w2 + 1e-12)        # 1: the whole write goes into the sum; 0: all of it into the stream differences
    return H2, dict(kept=kept, surf=surf, write=W, into_sum=into_sum)
@eqx.filter_jit
def ce(mod, k, s, tok_ex, w):
    norm = mod.aux_norms[k] if mod._separate(k) else mod.transformer.norm
    x = norm(s.astype(mod.embeddings.token_embeddings.weight.dtype) if False else s.astype(jnp.bfloat16))
    lm_head = mod._head(k)
    return maybe_fused_next_token_loss(mod.Pos, mod.Embed, mod.Vocab, x, lm_head, tok_ex, loss_weight=w, reduction=None, reduction_axis=(), dtype=jnp.float32, logsumexp_weight=None, logit_soft_cap=None).rearrange((Batch, Pos)).array
def nrm(a):
    a = np.asarray(a, np.float64); s = np.abs(a).max(-1, keepdims=True)
    return (s[..., 0] * np.sqrt(((a / np.maximum(s, 1e-300)) ** 2).sum(-1)))
res = {k: {} for k in LAYERS}
for b in range(N):
    tok = ids[b * B * T:][: B * T].reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    attn = AttentionMask.causal().with_segment_ids(seg)
    toks = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)); w = hax.named(jnp.ones((B, T), jnp.float32), (Batch, Pos))
    H = hax.stack(Axis("stream", m_), [model.embeddings.embed(toks)] * m_)
    for li in range(max(LAYERS)):
        lay = tr.layer(li); P = []
        for j in range(2):
            H, p = step(lay, H, attn, jnp.asarray(li), j); P.append(p)
        if li + 1 in LAYERS:
            p0, p1 = P
            a, s, wr = p1["kept"], p1["surf"], p1["write"]
            V = {"all": a + s + wr, "no write": a + s, "no surfaced": a + wr, "no kept": s + wr, "write alone": wr, "surfaced alone": s, "kept alone": a}
            for n, x in V.items():
                res[li + 1].setdefault(n, []).append(np.asarray(ce(model, li, x, toks, w)).ravel())
            for n, x in (("kept", a), ("surf", s), ("write", wr)):
                res[li + 1].setdefault("|" + n + "|", []).append(nrm(x.rearrange((Batch, Pos, Embed)).array).ravel())
            res[li + 1].setdefault("|new|", []).append(nrm((a + s + wr).rearrange((Batch, Pos, Embed)).array).ravel())
            res[li + 1].setdefault("into", []).append(np.asarray(jnp.broadcast_to(p1["into_sum"].array, p1["into_sum"].array.shape if p1["into_sum"].ndim else (B, T))).ravel())
print(f"\n{tag} {ck.rsplit('/', 1)[1]} streams {m_} res {rm} static {static}")
print(" layer | CE: all, no write, no surfaced, no kept, write alone, surfaced alone, kept alone | median norm / |new|: kept, surfaced, write | write into the sum")
for k in LAYERS:
    r = {n: np.concatenate(v) for n, v in res[k].items()}
    print(f"  L{k:<3d} " + " ".join(f"{r[n].mean():6.3f}" for n in ("all", "no write", "no surfaced", "no kept", "write alone", "surfaced alone", "kept alone")) + " | " + " ".join(f"{np.median(r['|' + n + '|'] / r['|new|']):6.3f}" for n in ("kept", "surf", "write")) + f" | {np.median(r['into']):6.3f}", flush=True)
