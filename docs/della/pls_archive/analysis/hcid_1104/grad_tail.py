"""The first layers of HC mixing = I with separate heads show large logged gradient norms late in training: at 524,288 tokens per step, w810 has 18.8 at layer 1, HC 2.8, while at 4,092 tokens they are close (47 vs 31, docs item 13).
A mean over many tokens shrinks noise by 1 / sqrt(tokens), and leaves a coherent part or a rare very large batch in place. Which is it?
Final checkpoint, read-only, in the production regime (compute bfloat16, gradients cast back to float32, all heads, the training loss). Text: a training shard, N batches of B x T tokens with no overlap; B x T x N = 524,288 by default, one logged step's worth of tokens.
Per batch: gradient norm per layer of the branch weights (down_proj, o_proj; the same quantity as W&B grad/norm/...) and of the hyper parameters, the per-head CE. Over the batches: the norm of the mean gradient per layer, what noise alone would give (sqrt(sum |g_b|^2) / N),
the cosine between the means over odd and even batches (near 0: noise; near 1: coherent), the largest batch's share of sum |g_b|^2 (near 1: one batch rules).
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY grad_tail.py OUT.npz TAG"""
import dataclasses, glob, json, os, sys, time
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, jmp, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
import tensorstore as ts
outp, tag = sys.argv[1:3]
B, T, N = int(os.environ.get("WS_B", 2)), int(os.environ.get("WS_T", 1024)), int(os.environ.get("WS_N", 256))
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
T_ = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + S_
TOK = os.environ.get("TOK_PATH", "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/nemotron_cc/hq_actual-5af4cc/train/part-00000-of-01018/input_ids/data")
ids = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": TOK}}).result()[: B * T * N].read().result(), np.int64)
assert ids.size == B * T * N and (ids == 0).mean() < 0.01, (ids.size, float((ids == 0).mean()))
print("devices", jax.devices(), "tokens", ids.size, flush=True)
Batch, Pos = Axis("batch", B), Axis("position", T)
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
mp = jmp.get_policy("p=f32,c=bfloat16")
ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
print("checkpoint", ck, flush=True)
wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
print({k: kw.get(k) for k in ("depth_arch", "depth_hc_res", "pls_weight", "pls_separate", "pls_detach_backbone", "pls_serial_readouts", "pls_fold_norm", "gradient_checkpointing", "hc_streams", "num_layers")}, flush=True)
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
Ln = cfg.Layers.name
def lf(m_, ex, key):
    loss, stats = m_.compute_next_token_loss(ex, key=key, logsumexp_weight=0.0)
    return (loss.array if isinstance(loss, hax.NamedArray) else loss), {k: (v.value() if hasattr(v, "value") else v) for k, v in stats.items()}
def layer_norm2(a):
    a = a.rearrange((Ln, ...)).array
    return jnp.sum(jnp.square(a.astype(jnp.float32)).reshape(a.shape[0], -1), axis=1)
@eqx.filter_jit
def step(m_, ex, key):
    (loss, stats), g = eqx.filter_value_and_grad(lf, has_aux=True)(mp.cast_to_compute(m_), ex, key)
    g = mp.cast_to_param(g)
    st = g.transformer.layers.stacked
    dp, op = st.mlp.down_proj.weight, st.self_attn.o_proj.weight
    hy = {}
    for j in (0, 1):
        for n in ("read", "write", "scale", "w_read", "w_write"):
            v = getattr(st.hyper[j], n, None)
            if v is not None:
                hy[f"{n}{j}"] = layer_norm2(v)
    return loss, stats, layer_norm2(dp) + layer_norm2(op), hy, dp.rearrange((Ln, ...)).array, op.rearrange((Ln, ...)).array
add = eqx.filter_jit(lambda a, b: jax.tree_util.tree_map(lambda x, y: x + y, a, b))
key = jrandom.PRNGKey(3)
A = [None, None]
norm2, hyper, losses, stat = [], {}, [], {}
t0 = time.time()
for b in range(N):
    tok = ids[b * B * T:][: B * T].reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=AttentionMask.causal().with_segment_ids(seg))
    loss, st, n2, hy, dp, op = step(model, ex, key)
    A[b % 2] = (dp, op) if A[b % 2] is None else add(A[b % 2], (dp, op))
    norm2.append(np.asarray(n2, np.float64)); losses.append(float(loss))
    for k, v in hy.items():
        hyper.setdefault(k, []).append(np.asarray(v, np.float64))
    for k, v in st.items():
        if "pls" in k or "loss" in k:
            stat.setdefault(k, []).append(float(v))
    if b < 3 or (b + 1) % 16 == 0:
        print(f"batch {b + 1}/{N}: loss {losses[-1]:.3f}, |g| L1 {np.sqrt(norm2[-1][0]):.2f}, L2 {np.sqrt(norm2[-1][1]):.2f}, {time.time() - t0:.0f} s", flush=True)
norm2 = np.array(norm2)
def per_layer(x):
    return np.asarray(jnp.sum(jnp.square(x).reshape(x.shape[0], -1), axis=1), np.float64)
def inner(x, y):
    return np.asarray(jnp.sum((x * y).reshape(x.shape[0], -1), axis=1), np.float64)
a2 = per_layer(A[0][0]) + per_layer(A[0][1]); b2 = per_layer(A[1][0]) + per_layer(A[1][1]); ab = inner(A[0][0], A[1][0]) + inner(A[0][1], A[1][1])
mean_norm = np.sqrt(a2 + b2 + 2 * ab) / N
noise = np.sqrt(norm2.sum(0)) / N
cos = ab / np.sqrt(a2 * b2 + 1e-60)
top = norm2.max(0) / norm2.sum(0)
pb = np.sqrt(norm2)
np.savez(outp, norm2=norm2, losses=np.array(losses), mean_norm=mean_norm, noise=noise, cos=cos, top=top, tag=tag, ck=ck, B=B, T=T, N=N, **{f"hyper_{k}": np.array(v) for k, v in hyper.items()}, **{f"stat_{k.replace('/', '_')}": np.array(v) for k, v in stat.items()})
print(f"{tag} {ck.rsplit('/', 1)[1]}: {N} batches of {B} x {T}; mean loss {np.mean(losses):.3f}, max {np.max(losses):.3f}")
print("layer  |mean g|  noise alone  cos(odd, even)  per-batch median  max  largest batch share of sum |g_b|^2")
for l in [0, 1, 2, 3, 4, 5, 7, 11, 12, 15, 23, 35, 47]:
    print(f"L{l + 1:<3d} {mean_norm[l]:8.3f} {noise[l]:8.3f}   {cos[l]:+.3f}   {np.median(pb[:, l]):8.2f} {pb[:, l].max():9.2f}   {top[l]:.3f}", flush=True)
print("hyper parameters, sum over layers of |g|^2, sqrt, per-batch median and max; largest batch and layer")
for k, v in hyper.items():
    v = np.array(v); s = np.sqrt(v.sum(1)); i = int(s.argmax()); print(f"  {k}: median {np.median(s):.2f}, max {s.max():.2f} at batch {i}, layer {int(v[i].argmax()) + 1}")
