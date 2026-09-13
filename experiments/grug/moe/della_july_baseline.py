# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""July Baseline (#6882) on Della 4xH100: the exact moe_may_july_baseline cells, GPU resources.

Only the accelerator plumbing differs from ``moe_may_july_baseline``: a single-process 4xH100
``ResourceConfig`` instead of v5p-8, a selectable GPU attention kernel, and our own W&B project.
Model, optimizer, data mixture, batch, steps, eval and seed are imported unchanged.

    DIM=512 ATTN=gpu_fa4_cute python -m experiments.grug.moe.della_july_baseline --dry_run true
"""

import dataclasses
import os

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import versioned

from experiments.grug.moe.moe_may_july_baseline import _POINTS, _build_step

if os.environ.get("ZEPHYR_SUBPROCESS") == "1":
    # LocalClient runs zephyr workers as threads in one process, so tokenization is GIL-bound
    # (~1.4 cores). Per-shard subprocesses give real multi-core tokenize on a login node.
    import zephyr.execution
    from zephyr.runners import SubprocessRunner

    zephyr.execution._default_stage_runner_factory_for = lambda client: lambda n: SubprocessRunner(num_workers=n)

_NO_WANDB_ARTIFACTS = os.environ.get("WANDB_NO_ARTIFACTS") == "1" or any(
    "della-proxy" in os.environ.get(k, "") for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
)
if _NO_WANDB_ARTIFACTS:
    # Della's proxy allows api.wandb.ai (metrics) but not storage.googleapis.com, so any file or
    # artifact upload (levanter's requirements.txt, save_code) hangs run.finish(). Metrics still stream.
    for _k, _v in {
        "WANDB_IGNORE_GLOBS": "*",
        "WANDB_DISABLE_CODE": "true",
        "WANDB_CONSOLE": "off",
        "WANDB_X_DISABLE_META": "true",
    }.items():
        os.environ.setdefault(_k, _v)

    import logging

    import wandb.sdk.wandb_run

    def _skip_log_artifact(self, artifact_or_path, name=None, type=None, **kwargs):
        logging.getLogger(__name__).info("Skipping wandb artifact upload %s (proxy blocks storage)", name)

    wandb.sdk.wandb_run.Run.log_artifact = _skip_log_artifact

_NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
_ATTN = os.environ.get("ATTN", "gpu_fa4_cute")

if _ATTN == "gpu_fa4_cute":
    # The grug evaluator runs the model on fp32 params (no cast_to_compute), but the FA4 CuTe kernel only
    # accepts bf16/fp16. Run fp32 calls through the kernel in bf16 and cast the output back; bf16 training
    # calls are unchanged.
    import jax.numpy as jnp
    import levanter.grug.attention._fa4_cute as _fa4

    _fa4_impl = _fa4.gpu_fa4_cute_attention

    def _fa4_bf16(q, k, v, mask):
        if q.dtype in (jnp.bfloat16, jnp.float16):
            return _fa4_impl(q, k, v, mask)
        out = _fa4_impl(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16), mask)
        return out.astype(q.dtype)

    _fa4.gpu_fa4_cute_attention = _fa4_bf16
_DIMS = [int(d) for d in os.environ.get("DIM", "512").split(",")]
_TAG = os.environ.get("RUN_TAG", "della4xh100")


def _della_step(hidden_dim: int, batch_size: int, num_steps: int):
    step = _build_step(hidden_dim, batch_size, num_steps)
    cfg = step.config
    model = dataclasses.replace(cfg.model.value, attention_implementation=None if _ATTN == "none" else _ATTN)
    run_id = f"{cfg.run_id}_{_TAG}_{_ATTN}"
    tracker = dataclasses.replace(
        cfg.tracker,
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "marin-della"),
        group="july-baseline-della",
        tags=[*cfg.tracker.tags, _TAG, _ATTN],
    )
    config = dataclasses.replace(
        cfg,
        model=versioned(model),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_gpu("H100", count=_NUM_GPUS)),
        tracker=tracker,
    )
    return dataclasses.replace(step, name=f"grug/{run_id}", config=config)


if __name__ == "__main__":
    steps = [_della_step(d, bs, n) for (d, bs, n) in _POINTS if d in _DIMS]
    executor_main(steps=steps, description=f"July Baseline on Della {_NUM_GPUS}xH100, attn={_ATTN}, dims={_DIMS}")
