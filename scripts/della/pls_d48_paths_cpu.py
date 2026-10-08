"""Diagnostic (2026-10-07): on the same weights, does the per-layer forward (layer_outputs, Stacked.scan_via) give the
baseline forward's (transformer.__call__, Stacked.fold) final hidden state and loss at 48 layers?

T3 (15140369, d48 probes) ended with eval/loss 5.725 through the baseline path and eval/L47/loss 7.570 through the
readout path on the same weights; the CPU tests (L=3, fp32, VANILLA attention) have the two equal. Here: the 48-layer
baseline T2 checkpoint (15140368_4, step 599) loaded into a PerLayer model (pls_weight=0: no new parameters), a few
random sequences, both paths, under fp32 and bf16 compute and under the VANILLA and JAX_FLASH attention backends.

    source scripts/della/pls_env.sh; JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/pls_d48_paths_cpu.py [T]
"""
import os
import sys

if os.environ.get("PATHS_GPU") != "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import jmp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.checkpoint import load_checkpoint  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

CKPT = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-d48-diag600-pdp4/2026.09.13/checkpoints/step-599"
T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
B = int(sys.argv[2]) if len(sys.argv) > 2 else 2
Vocab, Batch, Pos = Axis("vocab", 128256), Axis("batch", B), Axis("position", T)


def config(backend):
    return PerLayerQwen3Config(max_seq_len=T, hidden_dim=512, intermediate_dim=1792, num_layers=48, num_heads=8, num_kv_heads=8, hybrid_norm=True,
                               attn_backend=backend, pls_weight=0.0)


from haliax.partitioning import ResourceAxis  # noqa: E402

tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
_mesh = tc.use_device_mesh()
_mesh.__enter__()
_amap = hax.axis_mapping(tc.parameter_axis_mapping)
_amap.__enter__()
template = eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, config(AttentionBackend.VANILLA), key=jrandom.PRNGKey(0))
model = load_checkpoint(template, CKPT, subpath="model")
print("loaded", CKPT, "devices", jax.devices())
tok = np.random.default_rng(0).integers(0, 50000, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32), (Batch, Pos)), attn_mask=AttentionMask.causal())
ex = hax.shard(ex, tc.compute_axis_mapping)  # the batch split over the data axis, as in training


def paths(m):
    h_fold = m.activations(ex.tokens, ex.attn_mask, key=None)  # transformer.__call__: fold + final norm
    outs = m.layer_outputs(ex.tokens, ex.attn_mask, key=None)  # scan_via, every layer's pre-norm state
    Block = m.transformer.layers.Block
    h_scan = m.transformer.norm(outs[Block.name, -1])
    base = m.compute_next_token_loss(ex, key=None)
    readout = m.readout_losses(ex, (47,))
    # reference: each layer applied on its own (no scan), relative error of scan_via's output after every layer
    x, rel = m.embeddings.embed(ex.tokens), []
    for k in range(Block.size):
        layer = hax.tree_util.tree_map(lambda a, k=k: a[Block.name, k], m.transformer.layers.stacked)
        x = layer(x, mask=ex.attn_mask, key=None, pos_ids=None)
        d = (outs[Block.name, k] - x).array.astype(jnp.float32)
        rel.append(jnp.sqrt(jnp.mean(d ** 2)) / (jnp.sqrt(jnp.mean(x.array.astype(jnp.float32) ** 2)) + 1e-12))
    return h_fold.array.astype(jnp.float32), h_scan.array.astype(jnp.float32), base.array if isinstance(base, hax.NamedArray) else base, readout.array, jnp.stack(rel)


def swap_backend(m, backend):
    """The same weights under another attention backend: every config field on the tree replaced."""
    new = config(backend)
    is_cfg = lambda x: isinstance(x, type(new)) or (dataclasses.is_dataclass(x) and hasattr(x, "attn_backend") and not isinstance(x, jax.Array))
    def rep(x):
        if isinstance(x, type(new)):
            return new
        if dataclasses.is_dataclass(x) and hasattr(x, "attn_backend"):
            return dataclasses.replace(x, attn_backend=backend)
        return x
    return jax.tree_util.tree_map(rep, m, is_leaf=is_cfg)


for backend in (AttentionBackend.VANILLA, AttentionBackend.JAX_FLASH):
    for name, policy in (("fp32", jmp.get_policy("f32")), ("bf16", jmp.get_policy("p=f32,c=bfloat16"))):
        m = swap_backend(model, backend)
        m = policy.cast_to_compute(m)
        try:
            with hax.axis_mapping(tc.compute_axis_mapping):
                hf, hs, base, ro, rel = eqx.filter_jit(paths)(m)
            rel = np.asarray(rel)
            print(f"   relative rms(scan_via out - per-layer loop) by layer: " + " ".join(f"{v:.1e}" for v in rel[[0, 1, 2, 5, 11, 23, 35, 47]]) + f" (layers 1,2,3,6,12,24,36,48; max {rel.max():.1e} at layer {int(rel.argmax()) + 1})")
            print(f"{backend.name:10s} {name}: max|h_fold - h_scan| {float(jnp.max(jnp.abs(hf - hs))):.3e} (|h| ~ {float(jnp.sqrt(jnp.mean(hf ** 2))):.3f}); "
                  f"loss baseline path {float(base):.4f}, readout L47 {float(jnp.mean(ro[0][:, :-1])):.4f}")
        except Exception as e:  # a backend that does not run on CPU
            print(f"{backend.name:10s} {name}: {type(e).__name__}: {str(e)[:200]}")


# 2026-10-08: XLA may treat a 48-trip scan with stacked outputs differently from a 6- or 12-trip one (unrolling
# thresholds). layer_outputs with the scan fully unrolled, against the baseline path, bf16 compute as in training.
def unrolled_paths(m):
    tr, Block = m.transformer, m.transformer.layers.Block
    x = m.embeddings.embed(ex.tokens)
    _, outs = tr.layers.scan_via(lambda layer, c, **kw: (lambda y: (y, y))(layer(c, **kw)), unroll=True)(x, mask=ex.attn_mask, key=None, pos_ids=None)
    h_fold = m.activations(ex.tokens, ex.attn_mask, key=None)
    h_unr = tr.norm(outs[Block.name, -1])
    outs_scan = m.layer_outputs(ex.tokens, ex.attn_mask, key=None)
    d = jnp.stack([jnp.max(jnp.abs((outs_scan[Block.name, k] - outs[Block.name, k]).array.astype(jnp.float32))) for k in range(Block.size)])
    return jnp.max(jnp.abs((h_fold - h_unr).array.astype(jnp.float32))), d


try:
    _m = jmp.get_policy("p=f32,c=bfloat16").cast_to_compute(swap_backend(model, AttentionBackend.JAX_FLASH))
    with hax.axis_mapping(tc.compute_axis_mapping):
        _dfu, _dsu = eqx.filter_jit(unrolled_paths)(_m)
    _dsu = np.asarray(_dsu)
    print(f"unroll=True bf16 JAX_FLASH: max|h_fold - h_unrolled| {float(_dfu):.3e}; max|scan_via out - unrolled out| by layer 1,2,6,12,24,48: "
          + " ".join(f"{_dsu[k]:.1e}" for k in (0, 1, 5, 11, 23, 47)) + f" (max {_dsu.max():.1e} at layer {int(_dsu.argmax()) + 1})")
except Exception as e:
    print(f"unroll check: {type(e).__name__}: {str(e)[:300]}")
