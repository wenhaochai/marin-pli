"""w810's state norm jumps to about 8,000 times the embedding by layer 8 (mean over tokens, pls_write_size.py). Is that all tokens or
a few? Per layer: median, 90th, 99th percentile and maximum of |s_l| / |e| over tokens, the share of tokens above 100 x the layer's
median, and the same relative write. Top tokens at layer 8 with their position and id. Same text as pls_write_size.py.
    JAX_PLATFORMS=cpu nice -n 19 $PY state_norm_tokens.py TAG [TAG ...]"""
import dataclasses, json, os, sys, time, glob
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax.numpy as jnp, jax.random as jrandom, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
import tensorstore as ts
B, T = int(os.environ.get("WS_B", 4)), 1024
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001
eos[:, 0] = False
Batch, Pos = Axis("batch", B), Axis("position", T)
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(np.cumsum(eos, axis=1), jnp.int32), (Batch, Pos)))
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
L = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 23, 35, 47]
for tag in sys.argv[1:]:
    ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-4958"))[0]
    wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
    model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
    tr = model.transformer
    x = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
    outs = eqx.filter_jit(lambda tr_, x_: tr_.outputs(x_, mask))(tr, x)
    s = np.asarray(outs.rearrange((cfg.Layers.name, ..., cfg.Embed.name)).array, np.float32).reshape(cfg.num_layers, B * T, -1).astype(np.float64)
    e = np.asarray(x.array, np.float32).reshape(B * T, -1).astype(np.float64)
    r = np.linalg.norm(s, axis=2) / np.linalg.norm(e, axis=1)[None]
    print(tag, "dtype", outs.array.dtype)
    print("layer   median     p90      p99      max     mean  share>100xmedian  rel-write median  rel-write p99")
    prev = np.concatenate([e[None] * (cfg.hc_streams if cfg.depth_arch in ("hc", "mhc") else 1), s[:-1]], 0)
    rw = np.linalg.norm(s - prev, axis=2) / np.linalg.norm(prev, axis=1 + 1)
    for l in L:
        v = r[l]; m = np.median(v)
        print(f"L{l+1:<4d} {m:8.2f} {np.percentile(v,90):8.2f} {np.percentile(v,99):8.2f} {v.max():10.1f} {v.mean():8.1f}  {np.mean(v > 100*m):8.4f}       {np.median(rw[l]):8.4f}     {np.percentile(rw[l],99):8.3f}")
    top = np.argsort(-r[7])[:6]
    print("top tokens at L8 (flat index, batch, pos, id, ratio):", [(int(i), int(i // T), int(i % T), int(tok.reshape(-1)[i]), round(float(r[7][i]), 1)) for i in top])
