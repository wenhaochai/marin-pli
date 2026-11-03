"""Why does w959 (HC, mixing keeps the stream sum, probes; L48 CE 11.76 since about step 500) not learn? From its newest checkpoint on 4,096 Paloma tokens, next to w960
(HC, mixing = a scalar, same stack and step range) and, at final, HC probes and w811: per layer the median, p99 and max of the stream-state norms relative to the embedding,
read weights, the static mixing's singular values and row sums, the dynamic mixing's size.
    JAX_PLATFORMS=cpu nice -n 19 $PY w959_state.py"""
import dataclasses, glob, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
import experiments.references.depth_arch_qwen3 as dq
import tensorstore as ts, wandb
B, T = 4, 1024
SP = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
TP = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + SP
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001; eos[:, 0] = False
Batch, Pos = Axis("batch", B), Axis("position", T)
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(np.cumsum(eos, axis=1), jnp.int32), (Batch, Pos)))
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
api = wandb.Api(timeout=300)
Lays = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 15, 23, 35, 47]
for tag in sys.argv[1:]:
    cks = sorted(glob.glob(f"{TP}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit('-', 1)[1])) + sorted(glob.glob(f"{SP}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit('-', 1)[1]))
    ck = cks[-1]
    wcfg = dict(api.run("reself/marin-della/muonh-qwen3-130m-della4xh100-pls1" + tag).config)["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
    model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
    print(f"== {tag}  {ck.rsplit('/', 1)[1]}  hc_res {getattr(cfg, 'depth_hc_res', None)}", flush=True)
    tr = model.transformer
    hy = tr.layers.stacked.hyper
    m = cfg.hc_streams
    # static mixing per sublayer; the constants depend on the sublayer index
    for j in (0, 1):
        p = hy[j]
        res = np.asarray(p.res.rearrange(("layer", "stream", "stream_out")).array, np.float64)
        read = np.asarray(p.read.rearrange(("layer", "stream")).array, np.float64)
        write = np.asarray(p.write.rearrange(("layer", "stream_out")).array, np.float64)
        sc = np.asarray(p.scale.rearrange(("layer", "hp_scale")).array, np.float64)
        def lrms(q):  # RMS per layer of a named array with a layer axis
            arr = np.asarray(q.rearrange(("layer", ...)).array, np.float64)
            return np.sqrt((arr ** 2).reshape(arr.shape[0], -1).mean(1))
        wres, wrd, wwr = lrms(p.w_res), lrms(p.w_read), lrms(p.w_write)
        print(f" sublayer {j}: layer |res|max s_rr s_w |read|max |write|max rows-unprojected  sv-projected  eig-projected  |w_res|,|w_read|,|w_write| RMS", flush=True)
        for l in Lays:
            cr, c, cw = dq._hc_constants(m, 2 * l + j)
            c = np.asarray(c.rearrange(("stream", "stream_out")).array, np.float64)
            A0 = c + res[l]
            P = A0 - ((A0.sum(1, keepdims=True) - 1) / m)
            print(f"  L{l+1:<3d} {np.abs(res[l]).max():7.2f} {sc[l,0]:7.2f} {sc[l,1]:7.2f} {np.abs(read[l]).max():7.2f} {np.abs(write[l]).max():8.2f}  rows {np.round(A0.sum(1), 2)}  sv {np.round(np.linalg.svd(P, compute_uv=False), 2)}  eig {np.round(np.sort(np.abs(np.linalg.eigvals(P)))[::-1], 2)}  rms {wres[l]:.3f} {wrd[l]:.3f} {wwr[l]:.3f}", flush=True)
    x = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
    outs = eqx.filter_jit(lambda tr_, x_: tr_.outputs(x_, mask))(tr, x)
    s = np.asarray(outs.rearrange((cfg.Layers.name, ..., cfg.Embed.name)).array, np.float32).reshape(cfg.num_layers, B * T, -1).astype(np.float64)
    e = np.asarray(x.array, np.float32).reshape(B * T, -1).astype(np.float64)
    r = np.linalg.norm(s, axis=2) / np.linalg.norm(e, axis=1)[None]
    print(f" readout state |s|/|e|, nonfinite {int((~np.isfinite(s)).sum())}\n layer median p99 max")
    for l in Lays:
        v = r[l]; print(f"  L{l+1:<3d} {np.nanmedian(v):10.2f} {np.nanpercentile(v, 99):10.2f} {np.nanmax(v):12.1f}")
    print(flush=True)
