"""How much does the last layer need the part of the streams no readout sees? (2026-10-14)

Follow-up of pls_hc_hidden_cpu.py. On the final HC / mHC checkpoints, after layer c every stream is replaced by the
mean of the streams (the zero-sum part H_s - mean H removed). The sum of the streams, which every per-layer readout
reads, is unchanged at layer c, so readout c is unchanged; later layers run on as usual. Reported: the final readout's
(layer 48) next-token CE with the cut minus without, on real text (paloma c4_en validation). A large rise means the
later layers use features stored where the readouts cannot see them.

    source scripts/della/pls_env.sh; JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/pls_hc_hidden_ablate_cpu.py [T] [B]
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import jmp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.checkpoint import load_checkpoint  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1-sep{bb}-a{arch}-vs-d48-g4/2026.09.13/checkpoints/step-4958"
T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
B = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CUTS = (-1, 4, 8, 12, 16, 20, 28, 36, 44)  # 1-based layer after which the zero-sum part is removed; -1 = no cut
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
w = np.ones((B, T), np.float32)
w[:, -1] = 0.0
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.asarray(w), (Batch, Pos)),
               attn_mask=AttentionMask.causal().with_segment_ids(seg))


def config(arch, probes):
    return PerLayerQwen3Config(max_seq_len=T, hidden_dim=512, intermediate_dim=1792, num_layers=48, num_heads=8, num_kv_heads=8,
                               hybrid_norm=True, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True,
                               pls_detach_backbone=probes, depth_arch=arch, scan_layers=False)


def final_ce(m, cut):
    """Layer-48 readout CE with the zero-sum part removed after (0-based) layer cut; cut < 0: no removal."""
    tr = m.transformer
    x = m.embeddings.embed(ex.tokens)
    S = Axis("stream", tr.config.hc_streams)
    H0 = hax.stack(S, [x] * S.size)

    def step(layer, H, i, *, key):
        H = layer.hyper_step(H, ex.attn_mask, key=None, pos_ids=None, layer=i).rearrange(H.axes)
        Hm = hax.mean(H, axis="stream").broadcast_axis(S).rearrange(H.axes)
        H = hax.where(i == cut, Hm, H)
        return H, hax.sum(H, axis="stream")

    _, outs = tr.layers.scan_via(step)(H0, hax.arange(tr.config.Layers), key=None)
    ce = m._readout_ce(outs, tr.config.num_layers - 1, ex)
    return ce.array if isinstance(ce, hax.NamedArray) else ce


for arch in ("hc", "mhc"):
    for probes in (False, True):
        name = f"{arch} {'probes' if probes else 'separate heads'}"
        template = eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, config(arch, probes), key=jrandom.PRNGKey(0))
        m = jmp.get_policy("f32").cast_to_compute(load_checkpoint(template, ROOT.format(bb="-bbfrozen" if probes else "", arch=arch), subpath="model"))
        f = eqx.filter_jit(final_ce)
        with hax.axis_mapping(tc.compute_axis_mapping):
            vals = {c: float(f(m, c - 1 if c > 0 else -1)) for c in CUTS}
        base = vals[-1]
        print(f"{name:24s} L48 CE {base:.4f}; rise after removing the zero-sum part after layer "
              + " ".join(f"{c}:{vals[c] - base:+.3f}" for c in CUTS if c > 0), flush=True)
        del m
