"""What gain does the learned mixing give the stream sum, per token, in real data? Hand-written hc step (bfloat16, as training) on FineWeb-Edu windows, final checkpoint. For every sublayer: g = mean over output streams of the row sums of res (static + dynamic), per token; |S| before the mix, |Sigma w F| the write into the sum. Prints the static gain, the median token gain, the cumulative gain from sublayer 0, and the share of the sum's squared norm that the current write carries.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY res_gain.py TAG [TAG ...]"""
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
from experiments.references.depth_arch_qwen3 import _hc_constants, _rms, _hc_res, _static
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load
B, T, N = 1, 4096, int(os.environ.get("WS_N", 2))
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
out = {}
for tag in sys.argv[1:]:
    ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
    cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
    model = mp.cast_to_compute(load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model"))
    tr, m_, Embed, rm = model.transformer, cfg.hc_streams, cfg.Embed, _hc_res(cfg)
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
        xr = hax.dot(read.astype(H.dtype), H, axis="stream")
        f = lay.branch(j, xr, attn, key=None, pos_ids=None)
        upd = write.astype(H.dtype) * f.broadcast_axis(write.resolve_axis("stream_out"))
        if rm == "identity":
            res = c_res
        elif rm == "scalar":
            res = (hax.sum(c_res * (c_res + f32(q.res))) / m_) * c_res
        else:
            assert rm == "learned", rm
        H2 = (hax.dot(res.astype(H.dtype), H, axis="stream") + upd).rename({"stream_out": "stream"})
        # gain of the sum: mean over streams of the sum over the output axis of res (for m = 1 the scalar itself), with equal streams; and the actual ratio for the sum itself
        S0, wF = hax.sum(f32(H), axis="stream"), hax.sum(f32(upd), axis="stream_out")
        kept = hax.sum(f32(hax.dot(res.astype(H.dtype), H, axis="stream")), axis="stream_out")
        def nrm(v):
            a = jnp.moveaxis(v.rearrange((Batch, Pos, Embed)).array, -1, -1)
            s = jnp.max(jnp.abs(a), axis=-1, keepdims=True) + 1e-30
            return (s[..., 0] * jnp.sqrt(jnp.sum((a / s) ** 2, axis=-1)))
        g_r = hax.mean(hax.sum(res, axis="stream_out"), axis="stream")
        g_row = g_r.rearrange((Batch, Pos)).array if g_r.ndim else jnp.broadcast_to(g_r.array, (B, T))   # the same for every token in static designs
        return H2, g_row, nrm(S0), nrm(wF), nrm(kept), nrm(hax.sum(f32(H2), axis="stream"))
    keep = []
    for b in range(N):
        tok = ids[b * B * T:][: B * T].reshape(B, T)
        seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
        attn = AttentionMask.causal().with_segment_ids(seg)
        x0 = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
        H = hax.stack(Axis("stream", m_), [x0] * m_)
        rows = []
        for li in range(cfg.num_layers):
            lay = tr.layer(li)
            for j in range(2):
                H, *v = step(lay, H, attn, jnp.asarray(li), j)
                rows.append([np.asarray(a, np.float64).ravel() for a in v])
        keep.append(np.array(rows))   # (sublayers, 5, tokens)
    A = np.concatenate(keep, 2)
    g, s0, wf, kept, s1 = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]
    def static_gain(li, j):
        c = _hc_constants(m_, 2 * li + j)[1]; r = f32(tr.layer(li).hyper[j].res)
        if rm == "identity":
            return 1.0
        if rm == "scalar":
            return float(hax.sum(c * (c + r)).array) / m_
        return float(hax.mean(hax.sum(c + r, axis="stream_out"), axis="stream").array)
    gs = np.array([static_gain(li, j) for li in range(cfg.num_layers) for j in range(2)])
    ratio = kept / np.maximum(s0, 1e-30)            # actual gain of the sum norm through the mixing
    share = wf ** 2 / np.maximum(s1 ** 2, 1e-30)    # write's squared norm over the new sum's
    out[tag] = dict(g=np.median(g, 1), gs=gs, ratio=np.median(ratio, 1), share=np.median(share, 1), s=np.median(s1, 1), wf=np.median(wf, 1))
    print(f"\n{tag} {ck.rsplit('/', 1)[1]} streams {m_} res {rm} static {static}: {A.shape[2]} tokens")
    print(" layer: static row-sum gain (mean of the two sublayers) | median token row-sum gain | median |sum| through the mix, ratio | log10 cumulative static | log10 cumulative median ratio | median |write| / |new sum| | median |sum|")
    cs = np.cumsum(np.log10(np.abs(gs))); cr = np.cumsum(np.log10(np.maximum(np.median(ratio, 1), 1e-30)))
    for l in (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 36, 42, 44, 46, 47, 48):
        i = 2 * (l - 1)
        print(f"  L{l:<3d} {gs[i:i + 2].mean():7.3f} {np.median(g[i:i + 2]):7.3f} {np.median(ratio[i:i + 2]):7.3f} {cs[i + 1]:8.2f} {cr[i + 1]:8.2f}   {np.sqrt(np.median(share[i + 1])):8.3f}   {np.median(s1[i + 1]):10.3e}", flush=True)
np.savez(os.environ.get("OUT", "/scratch/gpfs/GROUP/USER/tmp/pls_hcid/res_gain.npz"), **{f"{t}/{k}": v for t, d in out.items() for k, v in d.items()})
