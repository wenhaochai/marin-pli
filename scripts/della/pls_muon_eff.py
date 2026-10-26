"""MuonH update efficiency on the last-layer loss, on a finished separate-heads checkpoint (read-only, CPU).
MuonH moves every linear weight W (flattened to 2D, per layer of the scan) by lr |W| P / |P|, P = polar(G) = U V^T of
the gradient (momentum buffer in training; one batch here). The step size of each matrix does not depend on the
gradient, only its direction does. To first order the last-layer loss falls by
    lr sum_W |W| <g_fin,W, P_W> / |P_W|,
largest when P = polar(g_fin) (then <g_fin, P> = the nuclear norm of g_fin). Separate heads step along P = polar(g_fin +
g_int). Efficiency eta = sum_W |W| <g_fin, polar(g_sep)>/|.| / sum_W |W| <g_fin, polar(g_fin)>/|.| : the share of the
best last-layer decrease that the separate-heads update still gets (1: no loss; 0: none; < 0: it raises the last-layer
loss). Per layer and in total over the Linear weights (the parameters MuonH updates; Adam's vectors are left out).
Question (2026-10-25): HC's gradient angle between g_sep and g_fin is no smaller than Pre-LN's (21-25 deg against
16-18 deg), yet it pays less. If the cost is the last-layer decrease lost per step, eta should order with the cost:
HC 48 > HC 24 > HC 12, Pre-LN full > 1/2 > 1/4, and HC 48 > Pre-LN.
Gradients as in pls_conflict/conflict.py (the same weights under pls_detach_backbone False / True).
    JAX_PLATFORMS=cpu nice -n 19 $PY muon_eff.py RUN_ID OUT.json [--b 4 --t 1024]
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
from levanter.optim.util import map_flattened_linear_layers
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
    return float(v), g, bb


l_sep, g_sep, bb = grad_for(cfg_sep)
l_pr, g_fin, _ = grad_for(cfg_pr)
assert abs(l_sep - l_pr) < 1e-5 * max(1.0, abs(l_sep)), (l_sep, l_pr)   # the same forward pass; only the backward differs
isN = lambda x: isinstance(x, hax.NamedArray)


def unit_polar(layer):
    """polar(G) / |polar(G)|_F of one flattened 2D weight gradient (exact SVD; Newton-Schulz in training approximates it)."""
    if layer.weight is None or getattr(layer.weight, "array", None) is None:
        return layer
    a = layer.weight.array.astype(jnp.float32)
    u, _, vt = jnp.linalg.svd(a, full_matrices=False)
    p = u @ vt
    return dataclasses.replace(layer, weight=dataclasses.replace(layer.weight, array=p / (jnp.linalg.norm(p) + 1e-12)))


def wnorm(layer):
    """|W|_F of one flattened 2D weight, spread over its entries (so an elementwise sum gives |W| <a, b>)."""
    if layer.weight is None or getattr(layer.weight, "array", None) is None:
        return layer
    a = layer.weight.array.astype(jnp.float32)
    return dataclasses.replace(layer, weight=dataclasses.replace(layer.weight, array=jnp.full_like(a, jnp.linalg.norm(a))))


zero = lambda x: (x * 0 if eqx.is_array(x) or (isN(x) and getattr(x, 'array', None) is not None) else x)
P_sep = eqx.filter_jit(lambda g: map_flattened_linear_layers(unit_polar, g))(g_sep)
P_fin = eqx.filter_jit(lambda g: map_flattened_linear_layers(unit_polar, g))(g_fin)
Wn = eqx.filter_jit(lambda p: map_flattened_linear_layers(wnorm, p, or_else=zero))(bb)


def real(x):
    if x is None:
        return False
    if isN(x):
        try:
            return jnp.ndim(x.array) == len(x.axes)
        except ValueError:
            return False
    return jnp.ndim(x) > 0


leaves = lambda t: [x for x in jax.tree.leaves(t, is_leaf=isN) if real(x)]
T4 = [leaves(t) for t in (g_fin, P_sep, P_fin, Wn)]
assert len({len(t) for t in T4}) == 1, [len(t) for t in T4]
Lname, L = cfg_sep.Layers.name, cfg_sep.num_layers
per = np.zeros((L, 2)); tot = np.zeros(2); nmat = 0
for f, ps, pf, w in zip(*T4):
    arr = lambda x: np.asarray((x.rearrange((Lname, ...)) if isN(x) and Lname in x.axis_names else x).array if isN(x) else x, np.float64)
    if not np.any(arr(w)):
        continue   # not a Linear weight (Adam)
    nmat += 1
    fa, sa, qa, wa = (arr(x) for x in (f, ps, pf, w))
    if isN(f) and Lname in f.axis_names:
        fa, sa, qa, wa = (x.reshape(L, -1) for x in (fa, sa, qa, wa))
        per += np.stack([(wa * fa * sa).sum(1), (wa * fa * qa).sum(1)], 1)
    tot += [(wa * fa * sa).sum(), (wa * fa * qa).sum()]
out = dict(rid=args.rid, ckpt=ckpt, tokens=B * T, linear_leaves=nmat, loss_final=l_pr, eta=float(tot[0] / tot[1]),
           best=float(tot[1]), eta_layers=(per[:, 0] / per[:, 1]).tolist(), best_layers=per[:, 1].tolist())
json.dump(out, open(args.out, "w"), indent=1)
print("eta", round(out["eta"], 4), "linear leaves", nmat)
print("eta by layer", [round(v, 3) for v in out["eta_layers"]])
