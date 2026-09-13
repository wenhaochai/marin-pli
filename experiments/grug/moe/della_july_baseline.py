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

# Persist JAX's compilation cache (and, under it, XLA's autotune/kernel sub-caches) on GPFS so a
# resubmitted or resumed run skips the multi-minute compile + autotune. Set before JAX is imported.
_JAX_CACHE_ROOT = os.environ.get("JAX_CACHE_ROOT", "/scratch/gpfs/KARTHIKN/wc9403/.cache/jax")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_JAX_CACHE_ROOT, "compilation"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "all")

# Execution levers chosen by the MFU probe (scripts/della/mfu_probe.sbatch writes this file). Read at start so
# a pending run or resume segment uses the latest choice; explicit REMAT/REPLICATE/LHS env vars still win.
import json

_LEVER_FILE = os.environ.get(
    "LEVER_DEFAULTS_FILE", os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "lever_defaults.json")
)
_PROBED_LEVERS: dict = {}
if os.path.exists(_LEVER_FILE) and not os.environ.get("PROBE_STEPS"):
    with open(_LEVER_FILE) as _f:
        _PROBED_LEVERS = json.load(_f)
_PROBED_FOR_DIM = _PROBED_LEVERS.get(os.environ.get("DIM", "512").split(",")[0], {})
if os.environ.get("LHS", _PROBED_FOR_DIM.get("LHS", "0")) == "1":
    _flag = "--xla_gpu_enable_latency_hiding_scheduler=true"
    if _flag not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {_flag}".strip()

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import versioned

import experiments.grug.moe.launch as _grug_launch
from experiments.grug.moe.moe_may_july_baseline import _POINTS, _build_step

# With a local MARIN_PREFIX the temporary checkpoint base comes back as a file:// URL, which the tensorstore
# writer treats as a relative path: arrays land in <cwd>/file:/... while metadata.json goes to the real
# path, so resume silently finds nothing. Hand the checkpointer a plain local path instead.
_temp_ckpt_base = _grug_launch.temporary_checkpoint_base_path
_grug_launch.temporary_checkpoint_base_path = lambda output_path: _temp_ckpt_base(output_path).removeprefix("file://")

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

# Throughput levers. Each keeps the training math (same model, optimizer, data, batch, steps); only how the
# step is executed changes.
#   REMAT=recompute_all (July default) | save_moe (keep MoE tensors, recompute the rest) | none (no recompute)
#   REPLICATE=1: replicate params across the GPUs instead of FSDP-sharding them over the data axis
#   PROBE_STEPS=N: throughput probe -- N steps, no eval, disposable checkpoints, separate run id
# Per-rung defaults (overridable by env). A resume chain reads this table when each segment starts, so
# update it only between runs. Current values are the July defaults until the MFU probes pick better ones.
_LEVER_DEFAULTS = {
    512: {"REMAT": "recompute_all", "REPLICATE": "0"},
    768: {"REMAT": "recompute_all", "REPLICATE": "0"},
    1024: {"REMAT": "recompute_all", "REPLICATE": "0"},
}
_DIM0 = int(os.environ.get("DIM", "512").split(",")[0])
_DIM_DEFAULTS = {**_LEVER_DEFAULTS.get(_DIM0, {}), **{k: v for k, v in _PROBED_FOR_DIM.items() if k in ("REMAT", "REPLICATE")}}
_REMAT = os.environ.get("REMAT", _DIM_DEFAULTS.get("REMAT", "recompute_all"))
_REPLICATE = os.environ.get("REPLICATE", _DIM_DEFAULTS.get("REPLICATE", "0")) == "1"
_PROBE_STEPS = int(os.environ.get("PROBE_STEPS", "0"))
if _REMAT not in ("recompute_all", "save_moe", "none"):
    raise ValueError(f"REMAT must be recompute_all, save_moe or none, got {_REMAT!r}")

if _REMAT == "none":
    # The grug model wraps every block in eqx.filter_checkpoint. Give that module an equinox view whose
    # filter_checkpoint is the identity, so backward keeps forward activations instead of recomputing them.
    import equinox

    import experiments.grug.moe.model as _grug_model

    class _EqxWithoutRemat:
        def __getattr__(self, name):
            return getattr(equinox, name)

        @staticmethod
        def filter_checkpoint(fun, *args, **kwargs):
            return fun

    _grug_model.eqx = _EqxWithoutRemat()


def _lever_tag() -> str:
    parts = []
    if _REMAT != "recompute_all":
        parts.append(f"remat-{_REMAT}")
    if _REPLICATE:
        parts.append("rep")
    if "latency_hiding_scheduler=true" in os.environ.get("XLA_FLAGS", ""):
        parts.append("lhs")
    return "_".join(parts)


def _della_step(hidden_dim: int, batch_size: int, num_steps: int):
    step = _build_step(hidden_dim, batch_size, num_steps)
    cfg = step.config
    model = dataclasses.replace(cfg.model.value, attention_implementation=None if _ATTN == "none" else _ATTN)
    if _REMAT == "save_moe":
        model = dataclasses.replace(model, remat_mode="save_moe")
    grug_trainer = cfg.grug_trainer.value
    if _REPLICATE:
        grug_trainer = dataclasses.replace(grug_trainer, replica_axis_size=_NUM_GPUS)

    lever = _lever_tag()
    # A real run keeps one id across resume segments whatever levers each segment uses (the math is the same),
    # so its checkpoints are found again. Probes get the lever in their id so variants don't collide.
    run_id = f"{cfg.run_id}_{_TAG}_{_ATTN}"
    group = "july-baseline-della"
    overrides = {}
    if _PROBE_STEPS > 0:
        from datetime import timedelta

        from levanter.checkpoint import CheckpointerConfig

        run_id = f"{run_id}" + (f"_{lever}" if lever else "") + f"_probe{_PROBE_STEPS}"
        group = "mfu-probe"
        probe_root = os.path.join(os.environ["MARIN_PREFIX"], "mfu_probe", run_id)
        overrides = dict(
            steps=versioned(_PROBE_STEPS),
            eval=None,
            checkpointer=CheckpointerConfig(
                base_path=os.path.join(probe_root, "checkpoints"),
                temporary_base_path=os.path.join(probe_root, "checkpoints-temp"),
                append_run_id_to_base_path=False,
                save_interval=timedelta(days=7),
                keep=None,
            ),
        )
    tracker = dataclasses.replace(
        cfg.tracker,
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "marin-della"),
        group=group,
        tags=[*cfg.tracker.tags, _TAG, _ATTN, *([lever] if lever else []), *(["probe"] if _PROBE_STEPS else [])],
    )
    config = dataclasses.replace(
        cfg,
        model=versioned(model),
        grug_trainer=versioned(grug_trainer),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_gpu("H100", count=_NUM_GPUS)),
        tracker=tracker,
        **overrides,
    )
    return dataclasses.replace(step, name=f"grug/{run_id}", config=config)


if __name__ == "__main__":
    steps = [_della_step(d, bs, n) for (d, bs, n) in _POINTS if d in _DIMS]
    executor_main(steps=steps, description=f"July Baseline on Della {_NUM_GPUS}xH100, attn={_ATTN}, dims={_DIMS}")
