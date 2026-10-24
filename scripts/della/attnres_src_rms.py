"""Source sizes of a 48-layer Full AttnRes checkpoint through the model's own loop path (depth_arch_qwen3._attnres_outputs),
for static and dynamic routing alike: every source (slot 0 the embedding, slot 1 + s the output of sublayer s) passes through
_rms when it is written, so a wrapped _rms records each source's mean per-token RMS. Same text and metric as A2
(attnres_massive.py: Paloma c4_en validation tokens [0, B T), largest slot RMS over the median slot RMS). Read-only.
    JAX_PLATFORMS=cpu nice -n 19 $PY src_rms.py CKPT CONFIG.json ROUTE OUT.json
"""
import dataclasses, json, sys, time
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
import experiments.references.depth_arch_qwen3 as D
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel

ckpt, cfgp, route, outp = sys.argv[1:5]
B, T = 16, 1024
wcfg = json.load(open(cfgp))["model"]
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
kw["depth_route"] = route
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
assert cfg.depth_arch == "attnres" and cfg.num_layers == 48
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
template = eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0))
t0 = time.time()
m = load_checkpoint(template, ckpt, subpath="model")
print(f"loaded {ckpt} in {time.time() - t0:.0f}s", flush=True)

import tensorstore as ts
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001
eos[:, 0] = False
seg = np.cumsum(eos, axis=1)
Batch, Pos = Axis("batch", B), Axis("position", T)
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg, jnp.int32), (Batch, Pos)))
x = m.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))

rec = []
_rms0 = D._rms
def _rms_rec(v, axis):
    r = _rms0(v, axis)
    a = v.array.astype(jnp.float32)
    rec.append(float(jnp.mean(jnp.sqrt(jnp.mean(a * a, axis=-1)))))
    return r
D._rms = _rms_rec
with jax.disable_jit():
    outs = m.transformer._attnres_outputs(x, mask, [None] * cfg.num_layers, None)
D._rms = _rms0
S = 2 * cfg.num_layers + 1
assert len(rec) == S, (len(rec), S)   # one _rms call per written source, nothing else
med = float(np.median(rec))
order = np.argsort(rec)[::-1]
name = lambda s: "embed" if s == 0 else f"L{(s - 1) // 2 + 1}.{('attn', 'mlp')[(s - 1) % 2]}"
top = [dict(slot=int(s), write=name(int(s)), rms=rec[s], ratio_to_median=rec[s] / med) for s in order[:5]]
out = dict(ckpt=ckpt, route=route, median_rms=med, largest=top, rms=rec)
json.dump(out, open(outp, "w"), indent=1)
print(json.dumps(top, indent=0))
