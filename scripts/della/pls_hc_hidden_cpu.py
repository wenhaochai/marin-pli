"""Does a multi-stream design store features where the per-layer readouts cannot see them? (2026-10-14)

Theory under test (docs/della/pls.md, "Hidden-write theory"): separate heads cost the last layer because every layer's
readout (its own RMSNorm + lm_head, full column rank since vocab >> width) sees the whole state the later layers read,
so features meant for later layers perturb that layer's prediction. HC and mHC read out the SUM of their 4 streams; the
zero-sum part of the streams (H_s - mean_s H) is invisible to every readout. HC's writes are 1 + learned offsets
(signed), so HC can write a zero-sum update; mHC's writes are 2 sigmoid(.) > 0 and its mixing is doubly stochastic and
non-negative. Prediction: HC's separate-heads model keeps more of its state in the zero-sum part than HC's probes model,
and mHC keeps little in either.

Measured per layer k on the final checkpoints (step 4958), on real text (paloma c4_en validation):
    hidden share = sum_s ||H_s - mean H||^2 / sum_s ||H_s||^2   (0: all streams equal; the readout sees everything)
and, from the parameters alone, every sublayer's static write vector (HC: 1 + write offset; mHC: 2 sigmoid(write bias)):
its smallest entry and its zero-sum share ||w - mean w|| / ||w||.

    source scripts/della/pls_env.sh; JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/pls_hc_hidden_cpu.py [T] [B]
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import jmp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.checkpoint import load_checkpoint  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1-sep{bb}-a{arch}-vs-d48-g4/2026.09.13/checkpoints/step-4958"
T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
B = int(sys.argv[2]) if len(sys.argv) > 2 else 2
Vocab, Batch, Pos = Axis("vocab", 128256), Axis("batch", B), Axis("position", T)
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"

tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()

import tensorstore as ts  # noqa: E402

flat = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result()
tok = np.asarray(flat, np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001
eos[:, 0] = False
seg = hax.named(jnp.asarray(np.cumsum(eos, axis=1), jnp.int32), (Batch, Pos))
tokens = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))
mask = AttentionMask.causal().with_segment_ids(seg)


def config(arch, probes):
    return PerLayerQwen3Config(max_seq_len=T, hidden_dim=512, intermediate_dim=1792, num_layers=48, num_heads=8, num_kv_heads=8,
                               hybrid_norm=True, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True,
                               pls_detach_backbone=probes, depth_arch=arch, scan_layers=False)


def streams(m):
    """H after every layer, stacked on the layer axis: (layer, stream, batch, position, embed), float32."""
    tr = m.transformer
    L = tr.config.num_layers
    x = m.embeddings.embed(tokens)
    H0 = hax.stack(Axis("stream", tr.config.hc_streams), [x] * tr.config.hc_streams)

    def step(layer, H, i, *, key):
        H = layer.hyper_step(H, mask, key=None, pos_ids=None, layer=i).rearrange(H.axes)
        return H, H

    _, Hs = tr.layers.scan_via(step)(H0, hax.arange(tr.config.Layers), key=None)
    Hf = Hs.astype(jnp.float32)
    mean = hax.mean(Hf, axis="stream")
    hid = hax.sum((Hf - mean) ** 2, axis=("stream", "batch", "position", "embed"))
    tot = hax.sum(Hf ** 2, axis=("stream", "batch", "position", "embed"))
    return (hid / tot).array


def static_writes(m, arch):
    """(sublayer, stream) static write vectors."""
    hyp = m.transformer.layers.stacked.hyper  # tuple of 2 sublayers, each stacked on the layer axis
    out = []
    for j in range(2):
        p = hyp[j]
        if arch == "hc":
            w = 1.0 + np.asarray(p.write.array, np.float32)  # (layer, stream_out)
        elif arch == "hc_pos":   # 2 sigmoid of the write logit (its dynamic term aside), positive like mHC
            w = 2.0 / (1.0 + np.exp(-np.asarray(p.write.array, np.float32)))
        else:
            mm = 4
            bias = np.asarray(p.bias.array, np.float32)  # (layer, mhc): offsets; write constant 0
            w = 2.0 / (1.0 + np.exp(-bias[:, mm:2 * mm]))
        out.append(w)
    return np.stack(out, 1).reshape(-1, out[0].shape[-1])  # (2L, stream) in sublayer order


res = {}
for arch in os.environ.get("HC_ARCHS", "hc,mhc").split(","):   # HC_ARCHS=hc_pos for HC with positive writes
    for probes in (False, True):
        name = f"{arch} {'probes' if probes else 'separate heads'}"
        path = ROOT.format(bb="-bbfrozen" if probes else "", arch=arch)
        template = eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, config(arch, probes), key=jrandom.PRNGKey(0))
        m = load_checkpoint(template, path, subpath="model")
        m = jmp.get_policy("f32").cast_to_compute(m)
        w = static_writes(m, arch)
        zs = np.linalg.norm(w - w.mean(1, keepdims=True), axis=1) / np.linalg.norm(w, axis=1)
        with hax.axis_mapping(tc.compute_axis_mapping):
            hs = np.asarray(eqx.filter_jit(streams)(m))
        res[name] = dict(hidden=hs.tolist(), write_min=w.min(1).tolist(), write_zerosum=zs.tolist())
        sel = [0, 5, 11, 17, 23, 29, 35, 41, 46, 47]
        print(f"{name:24s} hidden share by layer " + " ".join(f"L{k + 1}={hs[k]:.3f}" for k in sel), flush=True)
        print(f"{'':24s} static write: min entry {w.min():+.3f}, sublayers with a negative entry {(w.min(1) < 0).sum()}/{len(w)}, "
              f"zero-sum share median {np.median(zs):.3f} max {zs.max():.3f}", flush=True)
        del m

import json  # noqa: E402

out = os.environ.get("HC_HIDDEN_OUT", "/scratch/gpfs/GROUP/USER/tmp/pls_status/hc_hidden.json")
json.dump(dict(T=T, B=B, source="paloma c4_en validation", runs=res), open(out, "w"))
print("wrote", out)
