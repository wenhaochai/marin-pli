"""Is w959's blow-up a property of the design? Final HC probes run (and HC separate heads): static mixing A0 = c_res + res per sublayer. Largest |eigenvalue| and singular value of A0, of A0 with the
row-sum correction that hcsum applies (P), the row sums, the share of negative entries in P, the off-diagonal mass per row. Cumulative product of the largest |eig| as a bound on how fast a mode can grow.
Then hcsum at step 2130 and mHC Sinkhorn probes final (its A from 512 Paloma tokens, averaged over tokens): off-diagonal mass and row and column sums, to see whether Sinkhorn moves content between streams.
    JAX_PLATFORMS=cpu nice -n 19 $PY mix_spectrum.py"""
import dataclasses, glob, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax.numpy as jnp, jax.random as jrandom, numpy as np
from haliax import Axis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
import experiments.references.depth_arch_qwen3 as dq
import haliax as hax, wandb
from haliax.partitioning import ResourceAxis
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
SP = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
TP = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + SP
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
api = wandb.Api(timeout=300)
def load(tag):
    cks = sorted(glob.glob(f"{TP}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit('-', 1)[1])) + sorted(glob.glob(f"{SP}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit('-', 1)[1]))
    ck = cks[-1]
    wcfg = dict(api.run("reself/marin-della/muonh-qwen3-130m-della4xh100-pls1" + tag).config)["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=64, attn_backend=AttentionBackend.VANILLA, **kw)
    return load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model"), cfg, ck.rsplit("/", 1)[1]
for tag in sys.argv[1:]:
    model, cfg, st = load(tag)
    m = cfg.hc_streams
    hy = model.transformer.layers.stacked.hyper
    print(f"== {tag} {st}", flush=True)
    print(" layer sub  rowsum(mean)  max|eig A0|  max sv A0  | projected: max|eig P| max sv P share<0  offdiag mass A0 per row")
    grow_a, grow_p = 0.0, 0.0
    for l in range(cfg.num_layers):
        for j in (0, 1):
            p = hy[j]
            res = np.asarray(p.res.rearrange(("layer", "stream", "stream_out")).array, np.float64)[l]
            c = np.asarray(dq._hc_constants(m, 2 * l + j)[1].rearrange(("stream", "stream_out")).array, np.float64)
            A0 = c + res
            P = A0 - (A0.sum(1, keepdims=True) - 1) / m
            ea, ep = np.abs(np.linalg.eigvals(A0)).max(), np.abs(np.linalg.eigvals(P)).max()
            grow_a += np.log10(ea); grow_p += np.log10(ep)
            if l in (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 15, 23, 35, 47):
                off = (np.abs(A0).sum() - np.abs(np.diag(A0)).sum()) / m
                print(f"  L{l+1:<3d} {j}  {A0.sum(1).mean():6.3f}  {ea:6.3f} {np.linalg.svd(A0, compute_uv=False)[0]:6.3f} | {ep:6.3f} {np.linalg.svd(P, compute_uv=False)[0]:6.3f} {(P < 0).mean():5.2f}  {off:6.3f}", flush=True)
    print(f"  log10 of the product of the largest |eig| over 96 sublayers: A0 {grow_a:+.1f}, P {grow_p:+.1f}", flush=True)
