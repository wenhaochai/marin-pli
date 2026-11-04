"""Per-token norm of the branch output F at every sublayer, in bfloat16 like training, on real training windows: what bound on |F| leaves healthy runs untouched? Quantiles over the tokens per sublayer; the table gives the largest of each quantile over the sublayers, and the sublayers above some bounds.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY branch_norm.py TAG [TAG ...]"""
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
for tag in sys.argv[1:]:
    ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
    cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
    model = mp.cast_to_compute(load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model"))
    tr, m_, Embed, rm = model.transformer, cfg.hc_streams, cfg.Embed, _hc_res(cfg)
    @eqx.filter_jit
    def step(lay, H, attn, li, j):
        q = lay.hyper[j]
        c_read, c_res, c_write = _hc_constants(m_, 2 * li + j)
        Hn = f32(_rms(H, Embed)); sc = f32(q.scale)
        read = c_read + f32(q.read) + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(q.w_read), axis=Embed))
        write = c_write + f32(q.write) + sc["hp_scale", 1] * hax.tanh(hax.dot(Hn, f32(q.w_write), axis=Embed).rename({"stream": "stream_out"}))
        xr = hax.dot(read.astype(H.dtype), H, axis="stream")
        f = lay.branch(j, xr, attn, key=None, pos_ids=None)
        upd = write.astype(H.dtype) * f.broadcast_axis(write.resolve_axis("stream_out"))
        if rm == "identity":
            H2 = H + upd.rename({"stream_out": "stream"})
        else:
            res = c_res + f32(q.res) + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(q.w_res), axis=Embed))
            H2 = (hax.dot(res.astype(H.dtype), H, axis="stream") + upd).rename({"stream_out": "stream"})
        ff = f32(f)
        return H2, hax.sqrt(hax.sum(ff * ff, axis=Embed)).rearrange((Batch, Pos)).array
    Q = []
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
                H, n = step(lay, H, attn, jnp.asarray(li), j)
                rows.append(np.asarray(n, np.float64).ravel())
        Q.append(np.stack(rows))
    A = np.concatenate(Q, 1)   # (sublayers, tokens)
    q = np.percentile(A, [50, 99, 99.9], axis=1); mx = A.max(1)
    print(f"{tag} {ck.rsplit('/', 1)[1]}: {A.shape[1]} tokens; over the {A.shape[0]} sublayers: largest median {q[0].max():.1f}, largest p99 {q[1].max():.1f}, largest p99.9 {q[2].max():.1f}, largest max {mx.max():.3g}")
    for c in (50, 64, 100, 200):
        print(f"   bound {c}: share of token-sublayer pairs above {(A > c).mean():.2e}; sublayers with any above {int((mx > c).sum())}; largest share in one sublayer {(A > c).mean(1).max():.4f} at sublayer {int((A > c).mean(1).argmax())}", flush=True)
    top = np.argsort(-q[0])[:5]
    print("   sublayers (layer.j) with the largest median:", [(f"L{i // 2 + 1}.{i % 2}", round(float(q[0][i]), 1), round(float(q[1][i]), 1), round(float(mx[i]), 1)) for i in top], flush=True)
