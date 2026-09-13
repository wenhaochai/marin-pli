# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU step-time micro-benchmark for the July d512 grug MoE model.

Times fwd+bwd of ``Transformer.next_token_loss`` for a grid of exact-math kernel choices
(MoE local backend, remat mode) at the per-GPU shape of the 4xH100 run (batch 4 x seq 8192).

    MOE_IMPLS=scatter,sonic REMATS=recompute_all,save_moe ATTN=gpu_fa4_cute SEQ=8192 BATCH=4 \
        python -m scripts.della.bench_d512
"""

import dataclasses
import os
import time

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, set_mesh
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.heuristic import MoeHeuristic
from experiments.grug.moe.model import Transformer

SEQ = int(os.environ.get("SEQ", "8192"))
BATCH = int(os.environ.get("BATCH", "4"))
STEPS = int(os.environ.get("STEPS", "10"))
ATTN = os.environ.get("ATTN", "gpu_fa4_cute")
MOE_IMPLS = os.environ.get("MOE_IMPLS", "scatter,sonic").split(",")
REMATS = os.environ.get("REMATS", "recompute_all,save_moe").split(",")
FLOPS_PER_TOKEN_FWD = 270_336_000  # levanter analytic estimate for July d512 @ seq 8192


def bench(moe_impl: str, remat: str) -> None:
    cfg = dataclasses.replace(
        MoeHeuristic().build_model_config(512, seq_len=SEQ),
        disable_pko=True,
        disable_long_rope=True,
        attention_implementation=None if ATTN == "none" else ATTN,
        moe_implementation=moe_impl,
        remat_mode=remat,
    )
    key = jax.random.key(0)
    mesh = compact_grug_mesh(expert_axis_size=1)
    with set_mesh(mesh):
        params = Transformer.init(cfg, key=key)
        params = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16) if a.dtype == jnp.float32 else a, params)
        tok_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        tokens = jax.device_put(jax.random.randint(jax.random.key(1), (BATCH, SEQ), 0, cfg.vocab_size, jnp.int32), tok_sharding)
        weights = jax.device_put(jnp.ones((BATCH, SEQ), jnp.float32), tok_sharding)

        def loss_fn(p):
            return p.next_token_loss(tokens, weights)

        step = jax.jit(jax.value_and_grad(loss_fn))
        for _ in range(3):
            loss, grads = step(params)
            jax.block_until_ready(grads)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            loss, grads = step(params)
        jax.block_until_ready(grads)
        dt = (time.perf_counter() - t0) / STEPS
    tokens_per_step = BATCH * SEQ
    tps = tokens_per_step / dt
    stats = jax.devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use", 0) / 2**30
    print(
        f"RESULT moe={moe_impl:8s} remat={remat:14s} attn={ATTN} B={BATCH} S={SEQ} "
        f"step={dt*1e3:8.1f} ms  tok/s={tps:10.0f}  model_tflops={3*FLOPS_PER_TOKEN_FWD*tps/1e12:6.1f}  "
        f"peak_mem={peak:5.1f} GiB  loss={float(loss):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    print("device:", jax.devices()[0].device_kind, "| jax", jax.__version__, flush=True)
    for remat in REMATS:
        for impl in MOE_IMPLS:
            try:
                bench(impl.strip(), remat.strip())
            except Exception as e:  # keep the grid going; report the failure inline
                print(f"RESULT moe={impl:8s} remat={remat:14s} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
