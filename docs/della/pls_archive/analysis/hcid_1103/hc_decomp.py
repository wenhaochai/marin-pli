"""How much of HC's static mixing is the plain shrink of the stream sum? A0 = I + res per sublayer, final checkpoints. T = I - (1 - g) / m 11' with g the mean row sum of A0, what depth_hc_res="sumshrink" applies. R = A0 - T, the rest. |A0 - I|, |T - I|, |R| in Frobenius norm, the share of the squared size of A0 - I that T explains, and the off-diagonal mass of A0 split by sign. By groups of layers, for HC's ordinary model and separate heads.
    JAX_PLATFORMS=cpu PYTHONPATH=. WANDB_BASE_URL=https://api.wandb.ai nice -n 19 $PY hc_decomp.py"""
import dataclasses, glob, os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax.numpy as jnp, jax.random as jrandom, numpy as np, wandb
from haliax import Axis
import haliax as hax
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
from haliax.partitioning import ResourceAxis
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"; SP = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
Vocab = Axis("vocab", 128256); fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}; api = wandb.Api(timeout=300)
for t, name in (("-sep-bbfrozen-ahc-fa4thd-vs-d48", "HC ordinary"), ("-sep-ahc-fa4thd-vs-d48", "HC separate heads"), ("-sep-bbfrozen-ahc-rstatic-vs-d48", "HC static routing, ordinary")):
    cks = sorted(glob.glob(f"{SP}{t}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit('-', 1)[1]))
    if not cks:
        print(name, "no checkpoint"); continue
    wcfg = dict(api.run(P + t).config)["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=64, attn_backend=AttentionBackend.VANILLA, **kw)
    model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), cks[-1], subpath="model")
    m = cfg.hc_streams; hy = model.transformer.layers.stacked.hyper
    res = np.stack([np.asarray(hy[j].res.rearrange(("layer", "stream", "stream_out")).array, np.float64) for j in (0, 1)], 1)   # (layer, sublayer, s, so)
    print(f"== {name} {cks[-1].rsplit('/', 1)[1]}")
    print(" layers  |A0-I|  |T-I|  |R|  T share of |A0-I|^2  offdiag positive  offdiag negative  mean g")
    for lo, hi in ((1, 1), (2, 6), (7, 11), (12, 12), (13, 24), (25, 36), (37, 48)):
        rows = []
        for l in range(lo - 1, hi):
            for j in (0, 1):
                E = res[l, j]; A0 = np.eye(m) + E; g = A0.sum(1).mean()
                T = np.eye(m) - (1 - g) / m; R = A0 - T
                off = A0 - np.diag(np.diag(A0))
                rows.append((np.linalg.norm(E), np.linalg.norm(T - np.eye(m)), np.linalg.norm(R), np.maximum(off, 0).sum() / m, -np.minimum(off, 0).sum() / m, g, np.sum((T - np.eye(m)) ** 2), np.sum(E ** 2)))
        r = np.array(rows)
        print(f"  L{lo}-{hi}: {r[:,0].mean():6.3f} {r[:,1].mean():6.3f} {r[:,2].mean():6.3f}   {r[:,6].sum() / r[:,7].sum():5.2f}     {r[:,3].mean():6.3f}   {r[:,4].mean():6.3f}   {r[:,5].mean():6.3f}", flush=True)
