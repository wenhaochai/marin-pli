"""Exact sources of the sum a probe reads. Each stream is linear in the embedding and the branch outputs: H_o(L) = c_e,o e + Sigma_i c_i,o F_i, so the sum at layer L is S_L = T_e e + Sigma_i T_i F_i with per-token coefficients T = Sigma_o c_o, tracked through the mixing (per token where the mixing is dynamic). In bfloat16 state, coefficients in float32. Prints for layer L: the share of |S_L| that comes from each source group by distance in sublayers (|T_i F_i| / |S_L|, median over tokens), and the trained probe's CE when S_L keeps only the sources of the last k sublayers, without the embedding part, and the embedding part alone.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY kernel.py TAG"""
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
from experiments.references.depth_arch_qwen3 import _hc_constants, _rms, _hc_res, _static, _sum_gain, HC_MOVE_BETA, HC_MOVE_SHIFT
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load
tag = sys.argv[1]
B, T, N = 1, 4096, int(os.environ.get("WS_N", 4))
LAYERS = [int(v) for v in os.environ.get("KERNEL_LAYERS", "12").split(",")]
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
@eqx.filter_jit
def step(lay, H, attn, li, j):
    """One sublayer: the new state, the mixing A(i, o) per token as a float32 array (batch, position, m, m), the write w(o), the branch output."""
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
    xr = hax.dot(read.astype(p_dtype(H)), H, axis="stream").astype(jnp.bfloat16)
    f = lay.branch(j, xr, attn, key=None, pos_ids=None)
    upd = write.astype(H.dtype) * f.astype(H.dtype).broadcast_axis(So0)
    H2 = (hax.dot(res.astype(H.dtype), H, axis="stream") + upd).rename({"stream_out": "stream"})
    A = res.broadcast_axis((Batch, Pos)).rearrange((Batch, Pos, S0, So0)).array if res.ndim == 2 else res.rearrange((Batch, Pos, S0, So0)).array
    W = write.broadcast_axis((Batch, Pos)).rearrange((Batch, Pos, So0)).array if write.ndim == 1 else write.rearrange((Batch, Pos, So0)).array
    return H2, A, W, f.rearrange((Batch, Pos, Embed)).array.astype(jnp.float32)
def p_dtype(H):
    return H.dtype
@eqx.filter_jit
def ce(mod, k, x_arr, toks, w):
    s = hax.named(x_arr, (Batch, Pos, Embed)).astype(jnp.bfloat16)
    norm = mod.aux_norms[k] if mod._separate(k) else mod.transformer.norm
    return maybe_fused_next_token_loss(mod.Pos, mod.Embed, mod.Vocab, norm(s), mod._head(k), toks, loss_weight=w, reduction=None, reduction_axis=(), dtype=jnp.float32, logsumexp_weight=None, logit_soft_cap=None).rearrange((Batch, Pos)).array
def nrm(a):
    s = jnp.max(jnp.abs(a), axis=-1, keepdims=True) + 1e-30
    return s[..., 0] * jnp.sqrt(jnp.sum((a / s) ** 2, axis=-1))
STATE_F32 = bool(getattr(cfg, "depth_hc_state_f32", False))
res = {l: {} for l in LAYERS}
for b in range(N):
    tok = ids[b * B * T:][: B * T].reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    attn = AttentionMask.causal().with_segment_ids(seg)
    toks = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)); w = hax.named(jnp.ones((B, T), jnp.float32), (Batch, Pos))
    e = model.embeddings.embed(toks)
    H = hax.stack(S0, [e.astype(jnp.float32 if STATE_F32 else e.dtype)] * m_)
    e_arr = e.rearrange((Batch, Pos, Embed)).array.astype(jnp.float32)
    c = [jnp.ones((B, T, m_), jnp.float32)]   # coefficient per source per stream; c[0]: the embedding
    Fs = [e_arr]
    for li in range(max(LAYERS)):
        lay = tr.layer(li)
        for j in range(2):
            H, A, W, f = step(lay, H, attn, jnp.asarray(li), j)
            c = [jnp.einsum("btio,bti->bto", A, ci) for ci in c] + [W]
            Fs.append(f)
        if li + 1 in LAYERS:
            n = len(Fs)                       # sources: 0 embedding, 1 .. n-1 the sublayers in order
            T_all = [ci.sum(-1) for ci in c]  # (b, t) per source
            parts = [Ti[..., None] * Fi for Ti, Fi in zip(T_all, Fs)]
            S_true = hax.sum(f32(H), axis="stream").rearrange((Batch, Pos, Embed)).array
            S_sum = sum(parts)
            r = res[li + 1]
            r.setdefault("recon", []).append(np.asarray(nrm(S_sum - S_true) / nrm(S_true)).ravel())
            base = nrm(S_true)
            r.setdefault("share", []).append(np.stack([np.asarray(nrm(p) / base).ravel() for p in parts]))
            r.setdefault("T", []).append(np.stack([np.asarray(Ti).ravel() for Ti in T_all]))
            r.setdefault("C", []).append(np.stack([np.asarray(jnp.sqrt((ci * ci).sum(-1))).ravel() for ci in c]))   # size of a source's coefficient over the streams, sum and differences together
            r.setdefault("Fn", []).append(np.stack([np.asarray(nrm(Fi)).ravel() for Fi in Fs]))
            r.setdefault("all", []).append(np.asarray(ce(model, li, S_true, toks, w)).ravel())
            r.setdefault("no embedding", []).append(np.asarray(ce(model, li, S_true - parts[0], toks, w)).ravel())
            r.setdefault("embedding", []).append(np.asarray(ce(model, li, parts[0], toks, w)).ravel())
            for k in (1, 2, 4, 8, 16, 32):
                if k < n - 1:
                    r.setdefault(f"last {k}", []).append(np.asarray(ce(model, li, sum(parts[-k:]), toks, w)).ravel())
print(f"\n{tag} {ck.rsplit('/', 1)[1]} streams {m_} res {rm} static {static}")
out = {}
for l in LAYERS:
    r = {k: np.concatenate(v, -1) for k, v in res[l].items()}
    share, Ts = r["share"], r["T"]   # (sources, tokens)
    print(f" L{l}: reconstruction error median {np.median(r['recon']):.1e}; probe CE all {r['all'].mean():.3f}, without the embedding part {r['no embedding'].mean():.3f}, embedding part alone {r['embedding'].mean():.3f}; " + ", ".join(f"{k} {r[k].mean():.3f}" for k in r if k.startswith("last ")))
    print(f"   median |T F| / |S|: embedding {np.median(share[0]):.3f}; by sublayer back from the last: " + " ".join(f"{np.median(share[-k]):.3f}" for k in range(1, min(25, share.shape[0]))))
    print(f"   median |T|: embedding {np.median(np.abs(Ts[0])):.2e}; by sublayer back from the last: " + " ".join(f"{np.median(np.abs(Ts[-k])):.2f}" for k in range(1, min(25, Ts.shape[0]))))
    Cs, Fn = r["C"], r["Fn"]
    print(f"   median |c| over the streams: embedding {np.median(Cs[0]):.2e}; by sublayer back from the last: " + " ".join(f"{np.median(Cs[-k]):.2f}" for k in range(1, min(25, Cs.shape[0]))))
    print(f"   median |F|: embedding {np.median(Fn[0]):.2f}; by sublayer back from the last: " + " ".join(f"{np.median(Fn[-k]):.1f}" for k in range(1, min(25, Fn.shape[0]))))
    print(f"   p10, p90 of log10 |T| for the sources 4, 8, 16, 24 sublayers back: " + " ".join(f"{np.percentile(np.log10(np.abs(Ts[-k]) + 1e-12), 10):.2f},{np.percentile(np.log10(np.abs(Ts[-k]) + 1e-12), 90):.2f}" for k in (4, 8, 16, 24) if k < Ts.shape[0]))
    out[f"{l}/C"] = np.median(Cs, 1); out[f"{l}/Fn"] = np.median(Fn, 1)
    out[f"{l}/share"] = np.median(share, 1); out[f"{l}/T"] = np.median(np.abs(Ts), 1)
    out[f"{l}/ce"] = np.array([r[k].mean() for k in ("all", "no embedding", "embedding")])
np.savez(f"/scratch/gpfs/GROUP/USER/tmp/pls_hcid/kernel{tag}.npz", **out)
