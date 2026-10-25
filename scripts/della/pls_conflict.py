"""Gradient conflict between the intermediate layer losses and the last-layer loss, on a finished separate-heads checkpoint
(read-only, CPU). The same weights are run under two configs that differ only in pls_detach_backbone:
  separate heads (False): the backbone gradient is g_fin + g_int, g_int the gradient of sum_k CE_k over the intermediate
                          readouts (pls_weight 1, uniform);
  probes (True):          the backbone gradient is g_fin alone, the gradient of the last-layer CE.
So g_int = g_sep - g_probe. Backbone = the transformer's parameters (layers stacked on the layer axis, plus the design's
non-stacked ones); the embedding, lm_head and the intermediate heads and norms are held fixed. For every layer (and in
total): |g_fin|, |g_int|, cos(g_fin, g_int). If the separate-heads cost is the conflict between the two, designs and
depths that pay less should show a smaller |g_int| / |g_fin| or a larger cos.
Text: Paloma c4_en validation tokens [0, B T), causal mask, next-token targets.
    JAX_PLATFORMS=cpu nice -n 19 $PY conflict.py RUN_ID OUT.json [--b 4 --t 1024]
RUN_ID without the muonh-qwen3- prefix; the model config comes from the run's W&B config (cached next to OUT).
"""
import argparse, dataclasses, json, os, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel

ap = argparse.ArgumentParser()
ap.add_argument("rid"); ap.add_argument("out")
ap.add_argument("--b", type=int, default=4); ap.add_argument("--t", type=int, default=1024)
args = ap.parse_args()
B, T = args.b, args.t
K = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun"
ckpt = f"{K}/muonh-qwen3-{args.rid}/2026.09.13/checkpoints/step-4958"
cpath = os.path.join(os.path.dirname(args.out), f"cfg-{args.rid}.json")
if not os.path.exists(cpath):
    import wandb
    r = wandb.Api(timeout=300).run(f"reself/marin-della/muonh-qwen3-{args.rid}")
    json.dump({"model": r.config["model"]}, open(cpath, "w"))
wcfg = json.load(open(cpath))["model"]
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
cfg_sep = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
assert cfg_sep.pls_separate_heads and not cfg_sep.pls_detach_backbone and cfg_sep.pls_weight == 1.0 and cfg_sep.pls_weight_shape == "uniform", "a separate-heads checkpoint"
cfg_pr = dataclasses.replace(cfg_sep, pls_detach_backbone=True)

tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)


def load(cfg):
    """The checkpoint under cfg, split into the backbone (arrays under .transformer) and the rest held fixed."""
    t0 = time.time()
    m = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ckpt, subpath="model")
    print(f"loaded {ckpt} in {time.time() - t0:.0f}s", flush=True)
    spec = jax.tree_util.tree_map_with_path(lambda p, x: eqx.is_array(x) and any(getattr(k, "name", None) == "transformer" for k in p), m)
    return eqx.partition(m, spec)


import tensorstore as ts
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
Batch, Pos = Axis("batch", B), Axis("position", T)
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)),
               loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=AttentionMask.causal())


def grad_for(cfg):
    bb, rest = load(cfg)
    print(f"backbone {sum(x.size for x in jax.tree.leaves(bb)) / 1e6:.1f}M parameters", flush=True)

    def lf(bb_):
        out = eqx.combine(bb_, rest).compute_next_token_loss(ex, key=jrandom.PRNGKey(3))   # a key selects the training loss (no key: the eval loss, last layer only)
        loss = out[0] if isinstance(out, tuple) else out
        return loss.array if isinstance(loss, hax.NamedArray) else loss
    t = time.time()
    v, g = eqx.filter_jit(eqx.filter_value_and_grad(lf))(bb)
    print(f"loss {float(v):.4f}, grad in {time.time() - t:.0f}s", flush=True)
    return float(v), g


l_sep, g_sep = grad_for(cfg_sep)
l_pr, g_fin = grad_for(cfg_pr)
assert abs(l_sep - l_pr) < 1e-5 * max(1.0, abs(l_sep)), (l_sep, l_pr)   # the same forward pass; only the backward differs
isN = lambda x: isinstance(x, hax.NamedArray)


def real(x):
    """A gradient leaf of a backbone parameter (the fixed parameters leave NamedArrays holding a scalar)."""
    if x is None:
        return False
    if isN(x):
        try:
            return jnp.ndim(x.array) == len(x.axes)
        except ValueError:
            return False
    return jnp.ndim(x) > 0


G_sep = [x for x in jax.tree.leaves(g_sep, is_leaf=isN) if real(x)]   # the two trees differ only in the static config
G_fin = [x for x in jax.tree.leaves(g_fin, is_leaf=isN) if real(x)]
print(f"{len(G_fin)} gradient leaves, {sum(int(np.prod(np.shape(x.array if isN(x) else x))) for x in G_fin) / 1e6:.1f}M values", flush=True)
assert len(G_sep) == len(G_fin) and all(getattr(a, 'axes', a.shape) == getattr(b, 'axes', b.shape) for a, b in zip(G_sep, G_fin))
G_int = [a - b for a, b in zip(G_sep, G_fin)]

Lname, L = cfg_sep.Layers.name, cfg_sep.num_layers
per = np.zeros((L, 3))   # sum fin^2, int^2, fin.int per layer
tot = np.zeros(3)
for f, i in zip(G_fin, G_int):
    if isinstance(f, hax.NamedArray) and Lname in f.axis_names:
        fa = np.asarray(f.rearrange((Lname, ...)).array, np.float64).reshape(L, -1)
        ia = np.asarray(i.rearrange((Lname, ...)).array, np.float64).reshape(L, -1)
        per += np.stack([(fa * fa).sum(1), (ia * ia).sum(1), (fa * ia).sum(1)], 1)
    fa = np.asarray(f.array if isinstance(f, hax.NamedArray) else f, np.float64).ravel()
    ia = np.asarray(i.array if isinstance(i, hax.NamedArray) else i, np.float64).ravel()
    tot += [(fa * fa).sum(), (ia * ia).sum(), (fa * ia).sum()]
row = lambda s: dict(fin=float(np.sqrt(s[0])), int=float(np.sqrt(s[1])), ratio=float(np.sqrt(s[1] / s[0])), cos=float(s[2] / np.sqrt(s[0] * s[1])))
out = dict(rid=args.rid, ckpt=ckpt, tokens=B * T, loss_sep=l_sep, loss_final=l_pr, total=row(tot), layers=[row(s) for s in per])
json.dump(out, open(args.out, "w"), indent=1)
print(json.dumps(out["total"]))
print("cos by layer", [round(r["cos"], 3) for r in out["layers"]])
print("ratio by layer", [round(r["ratio"], 2) for r in out["layers"]])
