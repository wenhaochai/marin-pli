"""Does mHC's Sinkhorn mixing move on real tokens, the token-dependent part included? Hand-written _mhc_step on FineWeb-Edu windows (compute bfloat16, routing float32, as training), final checkpoint. Per sublayer: A = sinkhorn(residual logits), off-diagonal mass per row (1 - diagonal), mean and 99th percentile over tokens, by layer group. Then the last-layer loss with A as trained and with A replaced by I in every sublayer at evaluation.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY mhc_mix_full.py TAG"""
import dataclasses, glob, json, os, sys
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, jmp, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel, maybe_fused_next_token_loss
from experiments.references.depth_arch_qwen3 import _mhc_constants, _sinkhorn_liger
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load
tag = sys.argv[1]
B, T, N = 1, 4096, int(os.environ.get("WS_N", 4))
ids = load(B, T, N, seed=7)
Batch, Pos = Axis("batch", B), Axis("position", T)
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
mp = jmp.get_policy("p=f32,c=bfloat16")
f32 = lambda a: a.astype(jnp.float32)
ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
model = mp.cast_to_compute(load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model"))
tr, m, Embed = model.transformer, cfg.hc_streams, cfg.Embed
S0, So = Axis("stream", m), Axis("stream_out", m)
print(tag, ck.rsplit("/", 1)[1], "streams", m, "arch", cfg.depth_arch, "res", getattr(cfg, "depth_mhc_res", None), "rw", getattr(cfg, "depth_mhc_rw", None), flush=True)
@eqx.filter_jit
def step(lay, H, attn, li, j, identity):
    p = lay.hyper[j]
    Hf = f32(H)
    inv = hax.rsqrt(hax.mean(Hf * Hf, axis=(S0, Embed)) + 1e-6)
    mix = hax.dot(H, p.phi.astype(H.dtype), axis=(S0, Embed), preferred_element_type=jnp.float32) * inv
    g = f32(p.gain)
    c = _mhc_constants(m, 2 * li + j) + f32(p.bias)
    pre = mix["mhc", 0:m] * g["hp_scale", 0] + c["mhc", 0:m]
    post = mix["mhc", m:2 * m] * g["hp_scale", 1] + c["mhc", m:2 * m]
    res = mix["mhc", 2 * m:] * g["hp_scale", 2] + c["mhc", 2 * m:]
    read = hax.nn.sigmoid(pre.rename({"mhc": "stream"}))
    write = 2.0 * hax.nn.sigmoid(post.rename({"mhc": "stream_out"}))
    x = hax.dot(read, Hf, axis=S0)
    f = lay.branch(j, x.astype(H.dtype), attn, key=None, pos_ids=None)
    A = _sinkhorn_liger(res.unflatten_axis("mhc", (So, S0)))
    arr = A.rearrange((Batch, Pos, So, S0)).array
    diag = jnp.diagonal(arr, axis1=-2, axis2=-1)
    off = 1.0 - diag.mean(-1)
    if identity:
        H2 = (Hf + (write * f32(f).broadcast_axis(So)).rename({"stream_out": "stream"})).astype(H.dtype)
    else:
        H2 = (hax.dot(A, Hf, axis=S0) + write * f32(f).broadcast_axis(So)).rename({"stream_out": "stream"}).astype(H.dtype)
    return H2, off
@eqx.filter_jit
def ce(mod, s, toks, w):
    return maybe_fused_next_token_loss(mod.Pos, mod.Embed, mod.Vocab, mod.transformer.norm(s), mod.get_lm_head(), toks, loss_weight=w, reduction=None, reduction_axis=(), dtype=jnp.float32, logsumexp_weight=None, logit_soft_cap=None).rearrange((Batch, Pos)).array
off_all, ces = [], {False: [], True: []}
for b in range(N):
    tok = ids[b * B * T:][: B * T].reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    attn = AttentionMask.causal().with_segment_ids(seg)
    toks = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)); w = hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos))
    x0 = model.embeddings.embed(toks)
    for identity in (False, True):
        H = hax.stack(S0, [x0] * m)
        offs = []
        for li in range(cfg.num_layers):
            lay = tr.layer(li)
            for j in range(2):
                H, off = step(lay, H, attn, jnp.asarray(li), j, identity)
                if not identity:
                    offs.append(np.asarray(off, np.float64).ravel())
        c = np.asarray(ce(model, hax.sum(H, axis="stream").astype(jnp.bfloat16), toks, w)).ravel()[:-1]
        ces[identity].append(c)
        if not identity:
            off_all.append(np.array(offs))
O = np.concatenate(off_all, 1)   # (sublayers, tokens)
print(" group: mean off-diagonal mass per row, p99 over tokens, largest token")
for lo, hi in ((1, 1), (2, 6), (7, 11), (12, 12), (13, 24), (25, 36), (37, 48)):
    a = O[2 * (lo - 1):2 * hi]
    print(f"  L{lo}-{hi:<3d} {a.mean():.5f} {np.percentile(a, 99):.5f} {a.max():.5f}")
print(f" all sublayers: mean {O.mean():.5f}, p99 {np.percentile(O, 99):.5f}, max {O.max():.5f}")
a, b = np.concatenate(ces[False]), np.concatenate(ces[True])
print(f" last-layer loss on {len(a)} tokens: mixing as trained {a.mean():.4f}, every A replaced by I {b.mean():.4f}, difference {b.mean() - a.mean():+.4f}, mean |difference| per token {np.abs(b - a).mean():.4f}", flush=True)
