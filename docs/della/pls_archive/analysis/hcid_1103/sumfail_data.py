"""Data for hcid_sum_fail.py, one JSON. (1) Train CE at L48 over the first 2,200 steps (every 10 steps) for w959, w960, HC probes, w811. (2) CE per layer, mean of steps 2,100-2,197, same four runs. (3) Static mixing per sublayer: mean row sum of A0 = c_res + res and largest |eig| of the mixing the run applies, for HC probes at step 4,958 and w959 at step 2,130; g per sublayer for w960 at step 2,130.
    JAX_PLATFORMS=cpu WANDB_BASE_URL=https://api.wandb.ai PYTHONPATH=. nice -n 19 $PY sumfail_data.py OUT.json"""
import dataclasses, glob, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax.numpy as jnp, jax.random as jrandom, numpy as np, wandb
from haliax import Axis
import haliax as hax
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
import experiments.references.depth_arch_qwen3 as dq
from haliax.partitioning import ResourceAxis
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
SP = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
TP = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + SP
R = {"w959": "-sep-bbfrozen-ahc-hcsum-fa4thd-vs-d48", "w960": "-sep-bbfrozen-ahc-hcscalar-fa4thd-vs-d48", "HC": "-sep-bbfrozen-ahc-fa4thd-vs-d48", "w811": "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48"}
api = wandb.Api(timeout=300)
out = {"l48": {}, "ce": {}, "mix": {}}
K = [f"train/pls/L{l}" for l in range(48)]
for n, t in R.items():
    r = api.run(P + t)
    rows = {}
    for x in r.scan_history(keys=["_step", "train/pls/L47"], min_step=0, max_step=2200, page_size=2000):
        rows[int(x["_step"])] = x["train/pls/L47"]
    out["l48"][n] = sorted(rows.items())
    w = {}
    for x in r.scan_history(keys=["_step", *K], min_step=2100, max_step=2197, page_size=500):
        w[x["_step"]] = [x[k] for k in K]
    out["ce"][n] = np.mean(list(w.values()), axis=0).tolist()
    print(n, len(rows), len(w), flush=True)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
Vocab = Axis("vocab", 128256)
for n in ("HC", "w959", "w960"):
    t = R[n]
    cks = sorted(glob.glob(f"{TP}{t}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit('-', 1)[1])) + sorted(glob.glob(f"{SP}{t}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit('-', 1)[1]))
    wcfg = dict(api.run(P + t).config)["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=64, attn_backend=AttentionBackend.VANILLA, **kw)
    model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), cks[-1], subpath="model")
    m = cfg.hc_streams
    hy = model.transformer.layers.stacked.hyper
    rows = []
    for l in range(cfg.num_layers):
        for j in (0, 1):
            res = np.asarray(hy[j].res.rearrange(("layer", "stream", "stream_out")).array, np.float64)[l]
            c = np.asarray(dq._hc_constants(m, 2 * l + j)[1].rearrange(("stream", "stream_out")).array, np.float64)
            A0 = c + res
            if n == "w959":
                A = A0 - (A0.sum(1, keepdims=True) - 1) / m
            elif n == "w960":
                A = np.eye(m) * (c * A0).sum() / m
            else:
                A = A0
            rows.append({"layer": l + 1, "sub": j, "rowsum": float(A0.sum(1).mean()), "maxeig": float(np.abs(np.linalg.eigvals(A)).max()), "g": float((c * A0).sum() / m)})
    out["mix"][n] = {"step": int(cks[-1].rsplit('-', 1)[1]), "rows": rows}
    print(n, cks[-1].rsplit('/', 1)[1], flush=True)
json.dump(out, open(sys.argv[1], "w"))
