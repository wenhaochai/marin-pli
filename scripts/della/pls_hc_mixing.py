"""The learned residual mixing of Hyper-Connections, on finished 48-layer checkpoints (read-only, CPU). Owner popup
2026-10-30 (offline mechanism analysis): mHC pays about twice HC's separate-heads cost; neither its read/write maps nor
Sinkhorn-vs-identity mixing explain most of the gap, and HC with one stream pays more than Pre-LN. HC's mixing A is free
(identity + learned static offset + s tanh(.) per token); mHC's is doubly stochastic, which keeps the stream sum, the
state every readout reads. This records, per sublayer v = 2 l + j, the static A_v = I + res (in, out), the dynamic scales
(read/mixing, write), and how A_v acts on the stream sum and on the directions orthogonal to it:
    sum_gain(v)  = 1^T A_v 1 / m             (A applied to equal streams, read back by the sum)
    leak(v)      = |P_perp A_v^T 1| / |1|    (how much of the summed state A sends into stream differences)
    diff_gain(v) = spectral norm of P_perp A_v P_perp   (largest growth inside the difference space)
with P_perp the projector onto vectors whose entries sum to 0. Also the static read r_v and write w_v (+ offsets).
    JAX_PLATFORMS=cpu nice -n 19 $PY pls_hc_mixing.py CKPT CONFIG.json OUT.json
"""
import dataclasses, json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax.random as jrandom, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel

ckpt, cfgp, outp = sys.argv[1:4]
wcfg = json.load(open(cfgp))["model"]
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
cfg = PerLayerQwen3Config(max_seq_len=1024, attn_backend=AttentionBackend.VANILLA, **kw)
assert cfg.depth_arch == "hc", cfg.depth_arch
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
t0 = time.time()
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Axis("vocab", 128256), cfg, key=jrandom.PRNGKey(0)), ckpt, subpath="model")
print(f"loaded {ckpt} in {time.time() - t0:.0f}s", flush=True)
hyper = model.transformer.layers.stacked.hyper   # (HyperParams, HyperParams), each stacked over layers
m, L = cfg.hc_streams, cfg.num_layers
one = np.ones(m) / np.sqrt(m)
P = np.eye(m) - np.outer(one, one)
rows = []
for l in range(L):
    for j in range(2):
        p = hyper[j]
        res = np.asarray(p.res.rearrange((cfg.Layers.name, "stream", "stream_out")).array[l], np.float64)
        A = np.eye(m) + res                                   # (in, out): H_out[o] = sum_i A[i, o] H_in[i]
        sc = np.asarray(p.scale.array[l], np.float64)
        rd = np.asarray(p.read.array[l], np.float64) + (np.arange(m) == (2 * l + j) % m)
        wr = np.asarray(p.write.array[l], np.float64) + 1.0
        rows.append(dict(v=2 * l + j, sum_gain=float(np.ones(m) @ A @ np.ones(m) / m),
                         leak=float(np.linalg.norm(P @ (A @ np.ones(m))) / np.sqrt(m)) if m > 1 else 0.0,
                         diff_gain=float(np.linalg.norm(P @ A.T @ P, 2)) if m > 1 else 0.0,
                         scale_rr=float(sc[0]), scale_w=float(sc[1]), read=rd.tolist(), write=wr.tolist(),
                         read_sum=float(rd.sum()), write_sum=float(wr.sum())))
json.dump(dict(ckpt=ckpt, m=m, detach=cfg.pls_detach_backbone, rows=rows), open(outp, "w"), indent=1)
g = np.array([r["sum_gain"] for r in rows])
print("log10 prod sum_gain", round(float(np.log10(np.abs(g)).sum()), 2), "| late (v >= 80) mean sum_gain", round(float(g[80:].mean()), 3),
      "| mean leak", round(float(np.mean([r["leak"] for r in rows])), 3), "| mean diff_gain", round(float(np.mean([r["diff_gain"] for r in rows])), 3))
