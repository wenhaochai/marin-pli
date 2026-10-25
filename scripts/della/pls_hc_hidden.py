"""Hidden share of Hyper-Connections' residual state, on finished checkpoints (read-only, CPU). HC carries m streams
H = (h_1 .. h_m); every readout (and the next layers' attention and MLP inputs, through learned weights) sees the stream
sum s = sum_i h_i. The part of H a readout cannot see is H minus its stream mean, h_i - s / m. Per layer and per token:
    hidden share = sum_i |h_i - s/m|^2 / sum_i |h_i|^2      (0: all streams equal; (m-1)/m: s = 0)
averaged over tokens. Hypothesis (2026-10-25): separate heads cost HC less than Pre-LN because HC can keep features its
readouts never see in this hidden part; then a separate-heads HC model holds a larger hidden share than the ordinary
model (probes) trained with the same seed and data.
Text: Paloma c4_en validation tokens [0, B T), causal mask with document segments.
    JAX_PLATFORMS=cpu nice -n 19 $PY hc_hidden.py CKPT CONFIG.json OUT.json
CONFIG.json: the W&B model config of any run of the same design and depth (the stream count and shapes).
"""
import dataclasses, json, os, sys, time
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

ckpt, cfgp, outp = sys.argv[1:4]
B, T = int(os.environ.get("HID_B", 4)), 1024
wcfg = json.load(open(cfgp))["model"]
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
assert cfg.depth_arch in ("hc", "mhc"), cfg.depth_arch
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
t0 = time.time()
m = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ckpt, subpath="model")
print(f"loaded {ckpt} in {time.time() - t0:.0f}s", flush=True)

import tensorstore as ts
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001
eos[:, 0] = False
Batch, Pos = Axis("batch", B), Axis("position", T)
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(np.cumsum(eos, axis=1), jnp.int32), (Batch, Pos)))
tr, S = m.transformer, cfg.hc_streams
x = m.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))


def hstep(layer, H, i, *, key):   # depth_arch_qwen3.DepthArchTransformer.outputs for hc / mhc, emitting the streams
    H = layer.hyper_step(H, mask, key=key, pos_ids=None, layer=i).rearrange(H.axes)
    a = H.rearrange(("stream", ..., cfg.Embed.name)).array.astype(jnp.float32).reshape(S, -1, cfg.Embed.size)
    mean = a.mean(0, keepdims=True)
    tot = (a * a).sum((0, 2))                       # per token: sum_i |h_i|^2
    hid = ((a - mean) ** 2).sum((0, 2))             # per token: sum_i |h_i - s/m|^2
    return H, hax.named(jnp.stack([jnp.mean(hid / tot), jnp.mean(jnp.sqrt(tot / S)), jnp.mean(jnp.sqrt((mean * mean).sum((0, 2))))]), Axis("stat", 3))


t0 = time.time()
_, st = eqx.filter_jit(lambda tr_, x_: tr_.layers.scan_via(hstep)(hax.stack(Axis("stream", S), [x_] * S), hax.arange(cfg.Layers), key=None))(tr, x)
st = np.asarray(st.rearrange((cfg.Layers.name, "stat")).array, np.float64)
print(f"forward in {time.time() - t0:.0f}s", flush=True)
out = dict(ckpt=ckpt, streams=S, tokens=B * T, hidden_share=st[:, 0].tolist(), stream_rms=st[:, 1].tolist(), mean_norm=st[:, 2].tolist())
json.dump(out, open(outp, "w"), indent=1)
print("hidden share by layer", [round(v, 3) for v in st[:, 0]])
