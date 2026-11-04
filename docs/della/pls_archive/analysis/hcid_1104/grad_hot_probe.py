"""Sublayer probe of hot tokens (grad_hot2.py) in the first layers, in bfloat16 like training, following hyper_step. Fresh process: the gradient program of grad_hot2.py fills the A40.
For each sublayer, the token against the median token of the batch: |x|, its rank; peak / RMS of x; cancellation |x| / sum_s |r_s| |H_s| (near 0: the signed read cancels), its rank; branch output |F|, its rank; sum of read and write weights; |stream sum|.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY grad_hot_probe.py TAG BATCH SEQ POS [BATCH SEQ POS ...]"""
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
from experiments.references.depth_arch_qwen3 import _hc_constants, _rms, _hc_res
import tensorstore as ts
tag = sys.argv[1]
trip = [tuple(int(v) for v in sys.argv[i:i + 3]) for i in range(2, len(sys.argv), 3)]
B, T, N = int(os.environ.get("WS_B", 2)), int(os.environ.get("WS_T", 1024)), int(os.environ.get("WS_N", 256))
PROBE = int(os.environ.get("PROBE_LAYERS", 8))
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
T_ = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + S_
TOK = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/nemotron_cc/hq_actual-5af4cc/train/part-00000-of-01018/input_ids/data"
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load; ids = load(B, T, N)
Batch, Pos = Axis("batch", B), Axis("position", T)
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
mp = jmp.get_policy("p=f32,c=bfloat16")
ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
print("checkpoint", ck, flush=True)
wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
mc = mp.cast_to_compute(model)
tr = mc.transformer
m_ = cfg.hc_streams
Embed = cfg.Embed
f32 = lambda a: a.astype(jnp.float32)
rm = _hc_res(cfg)
A3 = lambda a: np.asarray(a.astype(jnp.float32).rearrange((Batch, Pos, Embed)).array, np.float64)
for b, seq, p in trip:
    tok = ids[b * B * T:][: B * T].reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    attn = AttentionMask.causal().with_segment_ids(seg)
    toks = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))
    x0 = mc.embeddings.embed(toks)
    H = hax.stack(Axis("stream", m_), [x0] * m_)
    idx = (seq, p)
    nE = np.linalg.norm(A3(x0), axis=-1)
    print(f"\n{tag} batch {b} seq {seq} position {p}, token ids {tok[seq, max(0, p - 3): p + 2].tolist()}: |e| {nE[idx]:.3g}, median {np.median(nE):.3g}")
    print("sublayer: |x| token / median (rank); peak/rms token / median; cancellation token / median (rank); |F| token / median (rank); F peak/rms; sum r token (median); sum w token (median); |stream sum| after, token / median (rank)")
    for li in range(PROBE):
        lay = tr.layer(li)
        for j in range(2):
            q = lay.hyper[j]
            c_read, c_res, c_write = _hc_constants(m_, 2 * li + j)
            Hn = f32(_rms(H, Embed))
            sc = f32(q.scale)
            read = c_read + f32(q.read) + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(q.w_read), axis=Embed))
            write = c_write + f32(q.write) + sc["hp_scale", 1] * hax.tanh(hax.dot(Hn, f32(q.w_write), axis=Embed).rename({"stream": "stream_out"}))
            xr = hax.dot(read.astype(H.dtype), H, axis="stream")
            f = lay.branch(j, xr, attn, key=None, pos_ids=None)
            upd = write.astype(H.dtype) * f.broadcast_axis(write.resolve_axis("stream_out"))
            xa, fa = A3(xr), A3(f)
            nx, nf = np.linalg.norm(xa, axis=-1), np.linalg.norm(fa, axis=-1)
            pk = np.abs(xa).max(-1) / (np.sqrt((xa ** 2).mean(-1)) + 1e-30)
            pf = np.abs(fa).max(-1) / (np.sqrt((fa ** 2).mean(-1)) + 1e-30)
            Ha = np.asarray(f32(H).rearrange((Batch, Pos, "stream", Embed)).array, np.float64)
            ra = np.asarray(read.rearrange((Batch, Pos, "stream")).array, np.float64)
            wa = np.asarray(write.rearrange((Batch, Pos, "stream_out")).array, np.float64)
            canc = nx / (np.sum(np.abs(ra) * np.linalg.norm(Ha, axis=-1), axis=-1) + 1e-30)
            if rm == "identity":
                H = H + upd.rename({"stream_out": "stream"})
            else:
                res = c_res + f32(q.res) + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(q.w_res), axis=Embed))
                H = (hax.dot(res.astype(H.dtype), H, axis="stream") + upd).rename({"stream_out": "stream"})
            ns = np.linalg.norm(np.asarray(hax.sum(f32(H), axis="stream").rearrange((Batch, Pos, Embed)).array, np.float64), axis=-1)
            rk = lambda v: float((v < v[idx]).mean())
            print(f"  L{li + 1}.{j}: |x| {nx[idx]:.3g} / {np.median(nx):.3g} ({rk(nx):.3f}); pk {pk[idx]:.1f} / {np.median(pk):.1f}; canc {canc[idx]:.3f} / {np.median(canc):.3f} ({rk(canc):.3f}); |F| {nf[idx]:.3g} / {np.median(nf):.3g} ({rk(nf):.3f}); Fpk {pf[idx]:.1f}; r {ra[idx].sum():.2f} ({np.median(ra.sum(-1)):.2f}); w {wa[idx].sum():.2f} ({np.median(wa.sum(-1)):.2f}); |sum| {ns[idx]:.3g} / {np.median(ns):.3g} ({rk(ns):.3f})", flush=True)
