"""Exact per-path decomposition of mHC with the residual mixing fixed to the identity (depth_mhc_res="identity"), on
finished checkpoints (read-only, CPU). Owner popup 2026-10-27, after the Qwen3.8-Next report's Gated Residual analysis
(sec. 2.2, Eq. 35-37): with no mixing, stream c before sublayer v is H_c = e + sum_{u<v} w_c(u) f(u), where e is the
embedding (copied to every stream), f(u) sublayer u's output and w_c(u) = 2 sigmoid(.) its write to stream c. Sublayer v
reads x(v) = sum_c r_c(v) H_c with r_c = sigmoid(.), so writer u's part of x(v) is (sum_c r_c(v) w_c(u)) f(u): one
positive scalar per stream times f(u). Per token,
    a(u->v, c) = r_c(v) w_c(u) |f(u)|,    pi(u->v) = sum_c a(u->v, c) / sum_{u'<v, c} a(u'->v, c)
(the embedding is writer -1 with w_c = 1). The RMSNorm inside the sublayer scales every writer alike, so it leaves pi
unchanged. Readers: the 96 sublayers (2 l: attention of layer l, 2 l + 1: its MLP) and the 48 readouts, which see the
stream sum after layer l (r_c = 1). pi is averaged over tokens. Check: the stream sum rebuilt from the recorded writes
equals the model's stream sum after the last layer.
Text: Paloma c4_en validation tokens [0, B T), causal mask with document segments.
    JAX_PLATFORMS=cpu nice -n 19 $PY pls_branch_paths.py CKPT CONFIG.json OUT.npz
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
from experiments.references.depth_arch_qwen3 import MhcParams, _mhc_constants

ckpt, cfgp, outp = sys.argv[1:4]
B, T = int(os.environ.get("PATHS_B", 8)), 1024
wcfg = json.load(open(cfgp))["model"]
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
assert cfg.depth_arch == "mhc" and cfg.depth_mhc_res == "identity", (cfg.depth_arch, cfg.depth_mhc_res)
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
t0 = time.time()
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ckpt, subpath="model")
print(f"loaded {ckpt} in {time.time() - t0:.0f}s", flush=True)

import tensorstore as ts
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001
eos[:, 0] = False
Batch, Pos = Axis("batch", B), Axis("position", T)
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(np.cumsum(eos, axis=1), jnp.int32), (Batch, Pos)))
tr, m, Embed = model.transformer, cfg.hc_streams, cfg.Embed
S, So = Axis("stream", m), Axis("stream_out", m)
x0 = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
f32 = lambda a: a.astype(jnp.float32)  # noqa: E731


def step(layer, H, li, *, key):
    """depth_arch_qwen3.DepthArchLayer._mhc_step, identity branch, recording read, write and |f| per sublayer."""
    rec = []
    for j in range(2):
        p = layer.hyper[j]
        assert isinstance(p, MhcParams)
        Hf = f32(H)
        inv = hax.rsqrt(hax.mean(Hf * Hf, axis=(S, Embed)) + 1e-6)
        mix = hax.dot(H, p.phi.astype(H.dtype), axis=(S, Embed), preferred_element_type=jnp.float32) * inv
        g = f32(p.gain)
        c = _mhc_constants(m, 2 * li.array + j) + f32(p.bias)
        read = hax.nn.sigmoid((mix["mhc", 0:m] * g["hp_scale", 0] + c["mhc", 0:m]).rename({"mhc": "stream"}))
        write = 2.0 * hax.nn.sigmoid((mix["mhc", m : 2 * m] * g["hp_scale", 1] + c["mhc", m : 2 * m]).rename({"mhc": "stream_out"}))
        f = layer.branch(j, hax.dot(read, Hf, axis=S).astype(H.dtype), mask, key=None, pos_ids=None)
        H = (Hf + (write * f32(f).broadcast_axis(So)).rename({"stream_out": "stream"})).astype(H.dtype)
        fn = hax.sqrt(hax.sum(f32(f) * f32(f), axis=Embed))
        rec.append((read.rearrange((..., "stream")).array, write.rearrange((..., "stream_out")).array, fn.array, f32(f).array))
    out = tuple(jnp.stack([r[i] for r in rec]) for i in range(4))
    return H, out


t0 = time.time()
run = eqx.filter_jit(lambda tr_, x_: tr_.layers.scan_via(step)(hax.stack(S, [x_] * m), hax.arange(cfg.Layers), key=None))
Hl, (rd, wr, fn, fv) = run(tr, x0)
print(f"forward in {time.time() - t0:.0f}s", flush=True)
L, N = cfg.Layers.size, B * T
rd = np.asarray(rd, np.float64).reshape(2 * L, N, m)           # reader v = 2 l + j: r_c(v)
wr = np.asarray(wr, np.float64).reshape(2 * L, N, m)           # writer u: w_c(u)
fn = np.asarray(fn, np.float64).reshape(2 * L, N)              # |f(u)|
e = np.asarray(f32(x0).array, np.float64).reshape(N, -1)

# Check: the stream sum after the last layer, rebuilt from e and the recorded writes, equals the model's.
fv = np.asarray(fv, np.float32).reshape(2 * L, N, -1)
rebuilt = m * e + np.einsum("un,und->nd", wr.sum(2), fv)
actual = np.asarray(f32(Hl).rearrange(("stream", ..., Embed.name)).array, np.float64).reshape(m, N, -1).sum(0)
rel = np.linalg.norm(rebuilt - actual) / np.linalg.norm(actual)
print(f"stream-sum rebuild relative error {rel:.2e}", flush=True)
assert rel < 2e-2, rel
del fv

# Writers: -1 = embedding (index 0 below), then sublayers 0..2L-1 (index u + 1).
wfull = np.concatenate([np.ones((1, N, m)), wr], 0)            # (2L + 1, N, m)
nfull = np.concatenate([np.linalg.norm(e, axis=1)[None], fn], 0)  # (2L + 1, N)
pi_sub = np.zeros((2 * L, 2 * L + 1, m))   # [v, u + 1, c] mean over tokens of a(u->v, c) / sum
for v in range(2 * L):
    a = rd[v][None] * wfull[: v + 1] * nfull[: v + 1, :, None]   # (v + 1 writers, N, m)
    tot = a.sum((0, 2))
    pi_sub[v, : v + 1] = (a / tot[None, :, None]).mean(1)
pi_ro = np.zeros((L, 2 * L + 1, m))        # readout after layer l sees writers up to sublayer 2 l + 1
for l in range(L):
    k = 2 * l + 3
    a = wfull[:k] * nfull[:k, :, None]
    tot = a.sum((0, 2))
    pi_ro[l, :k] = (a / tot[None, :, None]).mean(1)
np.savez(outp, ckpt=ckpt, tokens=N, pi_sub=pi_sub, pi_ro=pi_ro, read_mean=rd.mean(1), write_mean=wr.mean(1),
         fnorm_mean=fn.mean(1), enorm_mean=nfull[0].mean(), rebuild_rel=rel)
print("saved", outp)
