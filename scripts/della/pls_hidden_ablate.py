"""How much the last layer depends on the part of the state no readout sees, on finished 48-layer HC / mHC checkpoints
(read-only, CPU). Owner popup 2026-10-30 (offline mechanism analysis). Every readout of HC and mHC reads the stream sum,
so the stream differences (each stream minus the stream mean) are invisible to it, and only later sublayers read them.
After layer l0 this replaces every stream by the stream mean: the sum, and so readout l0, is unchanged, and the
differences are deleted. The run then goes on as trained, and the last layer's next-token cross-entropy is compared with
the untouched run:
    dCE(l0) = CE(layer 48 | streams collapsed after layer l0) - CE(layer 48),
with l0 = -1 collapsing the embedding copies (a no-op: they are equal). Also records the share of the state's squared norm
in the differences after each layer, |H - mean|^2 / |H|^2, averaged over tokens.
Text: Paloma c4_en validation tokens [0, B T), causal mask with document segments; CE over every position but the last.
    JAX_PLATFORMS=cpu nice -n 19 $PY pls_hidden_ablate.py CKPT CONFIG.json OUT.json
"""
import dataclasses, json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel

ckpt, cfgp, outp = sys.argv[1:4]
B, T = int(os.environ.get("HA_B", 4)), 1024
wcfg = json.load(open(cfgp))["model"]
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
assert cfg.depth_arch in ("hc", "mhc"), cfg.depth_arch
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
t0 = time.time()
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Axis("vocab", 128256), cfg, key=jrandom.PRNGKey(0)), ckpt, subpath="model")
print(f"loaded {ckpt} ({cfg.depth_arch}, m={cfg.hc_streams}) in {time.time() - t0:.0f}s", flush=True)

import tensorstore as ts
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001
eos[:, 0] = False
Batch, Pos = Axis("batch", B), Axis("position", T)
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(np.cumsum(eos, axis=1), jnp.int32), (Batch, Pos)))
x0 = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
m, Embed = cfg.hc_streams, cfg.Embed
S = Axis("stream", m)
tgt = jnp.asarray(tok[:, 1:])


def run(model_, x_, l0):
    tr = model_.transformer

    def step(layer, H, i, *, key):
        H = layer.hyper_step(H, mask, key=None, pos_ids=None, layer=i).rearrange(H.axes)
        Hf = H.astype(jnp.float32)
        mean = hax.mean(Hf, axis=S)
        d = Hf - mean.broadcast_axis(S)
        share = hax.mean(hax.sum(d * d, axis=(S, Embed)) / (hax.sum(Hf * Hf, axis=(S, Embed)) + 1e-30))
        H = hax.where(i.array == l0, mean.broadcast_axis(S).rearrange(H.axes).astype(H.dtype), H)
        return H, share.array

    H, share = tr.layers.scan_via(step)(hax.stack(S, [x_] * m), hax.arange(cfg.Layers), key=None)
    s = hax.sum(H, axis="stream")
    logits = model_.lm_head(tr.norm(s)).array.astype(jnp.float32)   # (batch, position, vocab)
    lp = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    ce = -jnp.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0].mean()
    return ce, share


f = eqx.filter_jit(run)
t0 = time.time()
base, share = f(model, x0, jnp.int32(-1))
base = float(base)
print(f"base CE {base:.4f} (forward {time.time() - t0:.0f}s)", flush=True)
L0 = list(range(0, cfg.num_layers - 1, 3)) + [cfg.num_layers - 2]
d = {}
for l0 in L0:
    ce, _ = f(model, x0, jnp.int32(l0))
    d[l0] = float(ce) - base
    print(f"collapse after layer {l0 + 1}: dCE {d[l0]:+.4f}", flush=True)
json.dump(dict(ckpt=ckpt, arch=cfg.depth_arch, m=m, detach=cfg.pls_detach_backbone, tokens=B * T, base_ce=base,
               dce={str(k + 1): v for k, v in d.items()}, diff_share=np.asarray(share).tolist()), open(outp, "w"), indent=1)
