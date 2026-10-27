"""How much each layer changes what the readouts see, on finished 48-layer checkpoints (read-only, CPU). Owner popup
2026-10-27, after the exact path split of mHC with fixed mixing (pls_branch_paths.py): there the separate-heads model
writes most in its early layers and little late, the ordinary model the reverse. This measures the same for any design.
s_l = the state readout l reads (DepthArchTransformer.outputs: Pre-LN's residual stream, HC / mHC's stream sum, AttnRes's
mix for the head), s_0 = the embedding as the readouts would see it (m copies summed for HC / mHC). Per token,
    write(l) = |s_l - s_{l-1}| / |s_0|,     relative write(l) = |s_l - s_{l-1}| / |s_{l-1}|,
averaged over tokens.
Text: Paloma c4_en validation tokens [0, B T), causal mask with document segments.
    JAX_PLATFORMS=cpu nice -n 19 $PY pls_write_size.py CKPT CONFIG.json OUT.json
"""
import dataclasses, json, os, sys, time
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

ckpt, cfgp, outp = sys.argv[1:4]
B, T = int(os.environ.get("WS_B", 4)), 1024
wcfg = json.load(open(cfgp))["model"]
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
t0 = time.time()
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ckpt, subpath="model")
print(f"loaded {ckpt} ({cfg.depth_arch}) in {time.time() - t0:.0f}s", flush=True)

import tensorstore as ts
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001
eos[:, 0] = False
Batch, Pos = Axis("batch", B), Axis("position", T)
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(np.cumsum(eos, axis=1), jnp.int32), (Batch, Pos)))
tr = model.transformer
x = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
t0 = time.time()
outs = eqx.filter_jit(lambda tr_, x_: tr_.outputs(x_, mask))(tr, x)
print(f"forward in {time.time() - t0:.0f}s", flush=True)
s = np.asarray(outs.rearrange((cfg.Layers.name, ..., cfg.Embed.name)).array, np.float32).reshape(cfg.num_layers, B * T, -1)
s0 = np.asarray(x.array, np.float32).reshape(B * T, -1) * (cfg.hc_streams if cfg.depth_arch in ("hc", "mhc") else 1)
prev = np.concatenate([s0[None], s[:-1]], 0)
d = np.linalg.norm(s - prev, axis=2)
out = dict(ckpt=ckpt, arch=cfg.depth_arch, detach=cfg.pls_detach_backbone, tokens=B * T,
           write=(d / np.linalg.norm(s0, axis=1)[None]).mean(1).tolist(), rel_write=(d / np.linalg.norm(prev, axis=2)).mean(1).tolist(),
           norm=np.linalg.norm(s, axis=2).mean(1).tolist(), norm0=float(np.linalg.norm(s0, axis=1).mean()))
json.dump(out, open(outp, "w"), indent=1)
print("write by layer", [round(v, 3) for v in out["write"]])
print("relative write by layer", [round(v, 3) for v in out["rel_write"]])
