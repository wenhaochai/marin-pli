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

_NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
_ATTN = os.environ.get("ATTN", "gpu_fa4_cute")
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
