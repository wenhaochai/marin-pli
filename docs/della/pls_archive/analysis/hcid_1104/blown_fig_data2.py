"""Per-layer tail of the state norm for the blown-token figure: |s_l| / |e| over 4 x 1024 Paloma c4_en validation tokens at the final checkpoint,
median, 99th percentile and maximum, for w810, HC separate heads and w811. Output: blown_state.json. Same text as pls_write_size.py."""
import dataclasses, json, os, sys, glob
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
B, T = 4, 1024
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
T_ = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + S_
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
out = {}
for tag in sys.argv[2:]:
    ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]; print("checkpoint", ck, flush=True)
    wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
    model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
    x = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
    outs = eqx.filter_jit(lambda tr_, x_: tr_.outputs(x_, mask))(model.transformer, x)
    s = np.asarray(outs.rearrange((cfg.Layers.name, ..., cfg.Embed.name)).array, np.float32).reshape(cfg.num_layers, B * T, -1).astype(np.float64)
    e = np.asarray(x.array, np.float32).reshape(B * T, -1).astype(np.float64)
    r = np.linalg.norm(s, axis=2) / np.linalg.norm(e, axis=1)[None]
    out[tag] = dict(p50=np.median(r, 1).tolist(), p99=np.percentile(r, 99, axis=1).tolist(), max=r.max(1).tolist(), share=(r > 100 * np.median(r, 1, keepdims=True)).mean(1).tolist())
    print(tag, "L5 p50 p99 max", round(out[tag]["p50"][4], 1), round(out[tag]["p99"][4], 1), round(out[tag]["max"][4], 1), flush=True)
json.dump(out, open(sys.argv[1], "w"))
