"""Does mHC's Sinkhorn mixing move content between streams? It keeps the stream sum and cannot expand (doubly stochastic, non-negative). Static part only: A = sinkhorn(constant + bias), the gain-scaled dynamic term set to 0.
Per sublayer: off-diagonal mass per row (share of an input stream's outflow that leaves it), largest |eig| after the top one, row and column sums. HC's numbers for the same layers come from mix_spectrum.py.
    JAX_PLATFORMS=cpu nice -n 19 $PY mhc_mix.py TAG [TAG ...]"""
import dataclasses, glob, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax.numpy as jnp, jax.random as jrandom, numpy as np
from haliax import Axis
import haliax as hax, wandb
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
import experiments.references.depth_arch_qwen3 as dq
from haliax.partitioning import ResourceAxis
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
SP = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
api = wandb.Api(timeout=300)
for tag in sys.argv[1:]:
    cks = sorted(glob.glob(f"{SP}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit('-', 1)[1]))
    ck = cks[-1]
    wcfg = dict(api.run("reself/marin-della/muonh-qwen3-130m-della4xh100-pls1" + tag).config)["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=64, attn_backend=AttentionBackend.VANILLA, **kw)
    model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
    m = cfg.hc_streams
    hy = model.transformer.layers.stacked.hyper
    print(f"== {tag} {ck.rsplit('/', 1)[1]}", flush=True)
    print(" layer sub  offdiag mass  diag mean  second|eig|  gain_res")
    S, So = Axis("stream", m), Axis("stream_out", m)
    for l in [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 15, 23, 35, 47]:
        for j in (0, 1):
            p = hy[j]
            bias = p.bias["layer", l]
            c = dq._mhc_constants(m, 2 * l + j) + bias.astype(jnp.float32)
            A = dq._sinkhorn_liger(c["mhc", 2 * m:].unflatten_axis("mhc", (So, S)))
            a = np.asarray(A.rearrange(("stream_out", "stream")).array, np.float64)
            eig = np.sort(np.abs(np.linalg.eigvals(a)))[::-1]
            g = float(np.abs(np.asarray(p.gain["layer", l].array, np.float64)[2]))
            print(f"  L{l+1:<3d} {j}  {(a.sum() - np.trace(a)) / m:6.3f}  {np.trace(a) / m:6.3f}  {eig[1]:6.3f}  {g:6.3f}", flush=True)
