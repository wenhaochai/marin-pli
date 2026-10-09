"""Diagnostic (2026-10-08): the d48 probes checkpoint through the training run's own eval functions.

pls_d48_paths_probe_ckpt.py on 1 H100 (15207626) gives 3.785 on paloma c4_en's first 16384 tokens through both the
baseline path and readout L47, with documents isolated by AttentionMask.with_segment_ids. The training run's evals
logged 4.231 (eval/loss path) and 6.479 (eval/L47) on the full c4_en set. Differences from that diagnostic: the eval
batch arrives as GrugLmExample (GrugLmExample.causal per row, then named_lm_example_from_grug: kv segment ids on
key_position), goes through levanter.eval._default_lm_eval_loss_fn and per_layer_qwen3._readout_eval_loss_fn under
the run's mixed-precision policy, and runs on 4 devices. Here: the same tokens, both masks, both eval functions,
under the device count of the run (EVAL_NDEV host devices on CPU, or the GPUs on GPU).

    EVAL_NDEV=4 JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/pls_d48_eval_path.py [T] [B]
"""
import os
import sys

NDEV = int(os.environ.get("EVAL_NDEV", "1"))
if os.environ.get("PATHS_GPU") != "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={NDEV}"

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
from levanter.data.text.examples import GrugLmExample  # noqa: E402
from levanter.eval import _default_lm_eval_loss_fn  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel, _readout_eval_loss_fn  # noqa: E402

CKPT = os.environ.get("PATHS_CKPT", "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen-d48/2026.09.13/checkpoints/step-4958")
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
B = int(sys.argv[2]) if len(sys.argv) > 2 else 4
EOS = 128001
Vocab, EvalBatch, Pos = Axis("vocab", 128256), Axis("batch", B), Axis("position", T)
cfg = PerLayerQwen3Config(max_seq_len=T, hidden_dim=512, intermediate_dim=1792, num_layers=48, num_heads=8, num_kv_heads=8, hybrid_norm=True,
                          attn_backend=AttentionBackend.JAX_FLASH, pls_weight=1.0, pls_separate_heads=True, pls_detach_backbone=True)
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
mesh = tc.use_device_mesh()
mesh.__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), CKPT, subpath="model")
mp = jmp.get_policy("p=f32,c=bfloat16")
if os.environ.get("PATHS_SHARD"):  # the trainer's parameter layout (FSDP: embed over data), not what load_checkpoint gives
    model = hax.shard(model, tc.parameter_axis_mapping)
_lw = jax.tree_util.tree_leaves(model.transformer.layers.stacked)[0]
print(f"param mapping {dict(tc.parameter_axis_mapping)}; a stacked layer weight {getattr(_lw, 'axes', None)} sharding {getattr(getattr(_lw, 'array', _lw), 'sharding', None)}; "
      f"lm_head sharding {model.get_lm_head().array.sharding}", flush=True)

import tensorstore as ts  # noqa: E402

VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int32).reshape(B, T)
if os.environ.get("EVAL_EOS"):  # document boundaries at different places in every row, so a row given another row's mask shows
    _r = np.random.default_rng(1)
    for _i in range(B):
        tok[_i, _r.choice(np.arange(8, T - 8), size=int(os.environ["EVAL_EOS"]), replace=False)] = EOS
print(f"loaded step {CKPT.rsplit('-', 1)[-1]}; devices {len(jax.devices())} x {jax.devices()[0].platform}; paloma c4_en first {B * T} tokens, "
      f"{int((tok == EOS).sum())} eos, T={T}, B={B}", flush=True)

# 1. the eval loader's batch: GrugLmExample.causal per row (CausalLmDataset), rows stacked by the loader
rows = [GrugLmExample.causal(jnp.asarray(r), eos_id=EOS, block_cross_document_attention=True) for r in tok]
grug = jax.tree.map(lambda *xs: jnp.stack(xs), *rows)
# 2. this session's diagnostic batch: one segment-id array for q and kv
_e = np.roll(tok, 1, axis=1) == EOS
_e[:, 0] = False
seg = hax.named(jnp.asarray(np.cumsum(_e, axis=1), jnp.int32), (EvalBatch, Pos))
mine = LmExample(tokens=hax.named(jnp.asarray(tok), (EvalBatch, Pos)), loss_weight=hax.named(jnp.asarray(np.asarray(rows[0].loss_weight)[None].repeat(B, 0)), (EvalBatch, Pos)),
                 attn_mask=AttentionMask.causal().with_segment_ids(seg))
print("grug segment ids equal mine:", bool(np.array_equal(np.asarray(grug.attn_mask.segment_ids[0]), np.asarray(seg.array))),
      "| loss weights equal:", bool(np.array_equal(np.asarray(grug.loss_weight), np.asarray(mine.loss_weight.array))), flush=True)


def mean_loss(per, w):
    return jnp.sum(per * w) / jnp.sum(w)


def run(batch):
    base, w, _ = _default_lm_eval_loss_fn(model, batch, EvalBatch=EvalBatch, mp=mp)
    ro, w2, _ = _readout_eval_loss_fn(model, batch, readouts=(0, 11, 23, 35, 46, 47), EvalBatch=EvalBatch, mp=mp)
    return mean_loss(base, w), jax.vmap(lambda p: mean_loss(p, w2))(ro)


for name, batch in (("grug eval batch", grug), ("with_segment_ids batch", mine)):
    batch = hax.shard(batch, tc.compute_axis_mapping) if isinstance(batch, LmExample) else jax.device_put(batch, jax.sharding.NamedSharding(tc.device_mesh, jax.sharding.PartitionSpec(tc.compute_axis_mapping["batch"])))
    with hax.axis_mapping(tc.compute_axis_mapping):
        base, ro = hax.named_jit(run, axis_resources=tc.compute_axis_mapping)(batch)
    ro = np.asarray(ro)
    print(f"{name:24s} {len(jax.devices())} dev: eval loss fn {float(base):.4f}; readout eval fn L0 {ro[0]:.4f} L11 {ro[1]:.4f} L23 {ro[2]:.4f} "
          f"L35 {ro[3]:.4f} L46 {ro[4]:.4f} L47 {ro[5]:.4f}", flush=True)


# 3. the training run's own per-layer evaluator: ReadoutTaggedEvaluator's accum_for_batch (all 48 readouts in one jit,
# the run's readout tuple), one batch with one tag. And a jit of the readout loss function alone (no baseline loss).
import types  # noqa: E402
from functools import partial  # noqa: E402

from levanter.eval import _EvalRunningMeans, _per_pos_out_sharding  # noqa: E402

from experiments.references.per_layer_qwen3 import ReadoutTaggedEvaluator  # noqa: E402

R = cfg.num_layers
gb = jax.device_put(grug, jax.sharding.NamedSharding(tc.device_mesh, jax.sharding.PartitionSpec(tc.compute_axis_mapping["batch"])))
ev = object.__new__(ReadoutTaggedEvaluator)
ev.readouts, ev.bytes_per_token, ev.device_mesh, ev.axis_mapping = tuple(range(R)), None, tc.device_mesh, tc.compute_axis_mapping
ev.per_pos_out_sharding = _per_pos_out_sharding(tc.device_mesh, tc.compute_axis_mapping, EvalBatch)
ev.dataset = types.SimpleNamespace(tag_to_index={"paloma/c4_en": 0}, num_tags=1)
ev.loss_fn = partial(_readout_eval_loss_fn, readouts=ev.readouts, EvalBatch=EvalBatch, mp=mp)
try:
    with hax.axis_mapping(tc.compute_axis_mapping):
        st = hax.shard(_EvalRunningMeans.zeros_like(jnp.zeros((R,), jnp.float32), jnp.zeros((R, 1), jnp.float32)))
        st = ev._make_accum_for_batch()(model, st, gb, jnp.ones((B, 1), jnp.int32))
        alone = hax.named_jit(lambda m, b: (lambda o: jax.vmap(lambda p: mean_loss(p, o[1]))(o[0]))(ev.loss_fn(m, b)), axis_resources=tc.compute_axis_mapping)(model, gb)
    tl, alone = np.asarray(st.token_avg_loss.mean), np.asarray(alone)
    print(f"ReadoutTaggedEvaluator accum {len(jax.devices())} dev: L0 {tl[0]:.4f} L11 {tl[11]:.4f} L23 {tl[23]:.4f} L35 {tl[35]:.4f} L46 {tl[46]:.4f} L47 {tl[47]:.4f}", flush=True)
    print(f"readout loss fn alone, R={R} {len(jax.devices())} dev: L0 {alone[0]:.4f} L11 {alone[11]:.4f} L23 {alone[23]:.4f} L35 {alone[35]:.4f} L46 {alone[46]:.4f} L47 {alone[47]:.4f}", flush=True)
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"evaluator check: {type(e).__name__}: {str(e)[:300]}", flush=True)
