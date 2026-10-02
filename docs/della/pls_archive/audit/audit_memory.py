"""Audit E: compile (CPU backend, 4 fake devices, no execution) the value-and-grad of the training loss at the REAL
per-device shapes (global batch 128 = 32 per device, T=4096, vocab 128256, bf16 compute policy, launcher mesh) for the
baseline and pls1/pls0, and compare XLA's memory analysis. CPU buffer assignment differs from GPU, so read the DELTA.
Also compiles the per-layer eval step (forward, all readouts) vs the main eval step.
"""
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import jmp  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.eval import _default_lm_eval_loss_fn  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.models.qwen import Qwen3Config  # noqa: E402
from levanter.trainer import TrainerConfig, WrappedLossFunction  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

import experiments.references.per_layer_qwen3 as pls  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config  # noqa: E402

SIZE = sys.argv[1] if len(sys.argv) > 1 else "300m"
h, inter, layers, heads = {"130m": (512, 1792, 6, 8), "300m": (768, 2688, 12, 12)}[SIZE]
GB, T, V = 128, 4096, 128256
Batch, Pos, Vocab = Axis("batch", GB), Axis("position", T), Axis("vocab", V)
BF16 = jmp.get_policy("p=f32,c=bfloat16")
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
cmap, pmap = tc.compute_axis_mapping, tc.parameter_axis_mapping
kw = dict(max_seq_len=T, hidden_dim=h, intermediate_dim=inter, num_layers=layers, num_heads=heads, num_kv_heads=heads, hybrid_norm=True,
          attn_backend=AttentionBackend.JAX_FLASH)


def gib(x):
    return f"{x / 2**30:6.2f} GiB"


def make_example():
    tokens = hax.zeros((Batch, Pos), dtype=jnp.int32)
    lw = hax.ones((Batch, Pos), dtype=jnp.float32)
    seg = hax.zeros((Batch, Pos), dtype=jnp.int32)
    return LmExample(tokens=tokens, loss_weight=lw, attn_mask=AttentionMask.causal().with_segment_ids(seg))


def analyze(name, cfg):
    t0 = time.time()
    with tc.use_device_mesh(), hax.axis_mapping(cmap):
        model = hax.shard(cfg.build(Vocab, key=jrandom.PRNGKey(0)), pmap)
        ex = hax.shard(make_example(), cmap)
        wl = WrappedLossFunction(lambda m, e, *, key=None: m.compute_next_token_loss(e, key=key, logsumexp_weight=0.0), BF16, cmap)

        def train_grad(m, e, k):
            with hax.axis_mapping(cmap):
                (loss, metrics), g = eqx.filter_value_and_grad(wl, has_aux=True)(m, e, key=k)
            return loss, g

        f = hax.named_jit(train_grad, axis_resources=pmap)
        comp = f.lower(model, ex, jrandom.PRNGKey(1)).compile()
        ma = comp.memory_analysis()
        out = f"{name:10s} train value+grad: temp {gib(ma.temp_size_in_bytes)} args {gib(ma.argument_size_in_bytes)} out {gib(ma.output_size_in_bytes)}"
        if isinstance(cfg, PerLayerQwen3Config):
            ev = hax.named_jit(lambda m, e: pls._readout_eval_loss_fn(m, e, readouts=tuple(range(layers)), EvalBatch=Batch, mp=BF16), axis_resources=cmap)
        else:
            ev = hax.named_jit(lambda m, e: _default_lm_eval_loss_fn(m, e, EvalBatch=Batch, mp=BF16), axis_resources=cmap)
        mae = ev.lower(model, ex).compile().memory_analysis()
        out += f" | eval loss_fn: temp {gib(mae.temp_size_in_bytes)}"
    print(out + f"  ({time.time() - t0:.0f}s)", flush=True)


analyze("baseline", Qwen3Config(**kw))
analyze("pls1", PerLayerQwen3Config(**kw, pls_weight=1.0))
analyze("pls0", PerLayerQwen3Config(**kw, pls_weight=0.0))
