# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""July Baseline (#6882) on Della 4xH100: the exact moe_may_july_baseline cells.

Model, optimizer, data mixture, batch, steps, eval and seed come unchanged from ``moe_may_july_baseline``.
This launcher only changes how the run executes on Della: a single-process 4xH100 resource, FA4 attention,
W&B through the compute-node proxy, a working temporary checkpoint path, and throughput levers that keep
the training math (REMAT, REPLICATE, latency-hiding scheduler). PROBE_STEPS turns a run into a short
throughput probe; scripts/della/mfu_probe.sbatch writes the best levers per rung to logs/lever_defaults.json,
which real runs read at start.

    DIM=512 python -m experiments.grug.moe.della_july_baseline --dry_run true
"""

import dataclasses
import json
import os

_DIM = int(os.environ.get("DIM", "512"))
_ATTN = os.environ.get("ATTN", "gpu_fa4_cute")
_NUM_GPUS = 4
_PROBE_STEPS = int(os.environ.get("PROBE_STEPS", "0"))

# Levers picked by the MFU probe for this rung; probes set their own through env instead.
_LEVER_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "lever_defaults.json")
_probed = {}
if not _PROBE_STEPS and os.path.exists(_LEVER_FILE):
    with open(_LEVER_FILE) as f:
        _probed = json.load(f).get(str(_DIM), {})
_REMAT = os.environ.get("REMAT", _probed.get("REMAT", "recompute_all"))
_REPLICATE = os.environ.get("REPLICATE", _probed.get("REPLICATE", "0")) == "1"
if _probed.get("LHS") == "1":
    # Must be in the environment before JAX creates its GPU client.
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_enable_latency_hiding_scheduler=true".strip()

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
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

if "della-proxy" in os.environ.get("https_proxy", ""):
    # Della's compute-node proxy passes api.wandb.ai (metrics) but not storage.googleapis.com, so any file or
    # artifact upload (levanter logs requirements.txt as an artifact) hangs run.finish(). Metrics still stream.
    os.environ.update(WANDB_IGNORE_GLOBS="*", WANDB_DISABLE_CODE="true", WANDB_CONSOLE="off", WANDB_X_DISABLE_META="true")
    import wandb.sdk.wandb_run

    wandb.sdk.wandb_run.Run.log_artifact = lambda self, *args, **kwargs: None

if _ATTN == "gpu_fa4_cute":
    # The grug evaluator runs the model on fp32 params (no cast_to_compute), but the FA4 CuTe kernel only
    # accepts bf16/fp16. Run fp32 calls through the kernel in bf16 and cast the output back; bf16 training
    # calls are unchanged.
    import jax.numpy as jnp
    import levanter.grug.attention._fa4_cute as _fa4

    _fa4_impl = _fa4.gpu_fa4_cute_attention

    def _fa4_bf16(q, k, v, mask):
        if q.dtype == jnp.float32:
            return _fa4_impl(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16), mask).astype(q.dtype)
        return _fa4_impl(q, k, v, mask)

    _fa4.gpu_fa4_cute_attention = _fa4_bf16

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


def _della_step(hidden_dim: int, batch_size: int, num_steps: int):
    step = _build_step(hidden_dim, batch_size, num_steps)
    cfg = step.config
    model = dataclasses.replace(cfg.model.value, attention_implementation=None if _ATTN == "none" else _ATTN)
    if _REMAT == "save_moe":
        model = dataclasses.replace(model, remat_mode="save_moe")
    grug_trainer = cfg.grug_trainer.value
    if _REPLICATE:
        grug_trainer = dataclasses.replace(grug_trainer, replica_axis_size=_NUM_GPUS)

    lever = "_".join(
        [f"remat-{_REMAT}"] * (_REMAT != "recompute_all")
        + ["rep"] * _REPLICATE
        + ["lhs"] * ("latency_hiding_scheduler=true" in os.environ.get("XLA_FLAGS", ""))
    )
    # A real run keeps one id across resume segments whatever levers a segment uses (the math is the same),
    # so its checkpoints are found again. Probes carry the lever in their id so variants don't collide.
    run_id = f"{cfg.run_id}_della4xh100_{_ATTN}"
    group = "july-baseline-della"
    overrides = {}
    if _PROBE_STEPS:
        from datetime import timedelta

        from levanter.checkpoint import CheckpointerConfig

        run_id = "_".join(filter(None, [run_id, lever, f"probe{_PROBE_STEPS}"]))
        group = "mfu-probe"
        probe_root = os.path.join(os.environ["MARIN_PREFIX"], "mfu_probe", run_id)
        overrides = dict(
            steps=versioned(_PROBE_STEPS),
            # One eval pass at the end, at the real eval batch size, to check that eval fits in memory.
            eval=versioned(dataclasses.replace(cfg.eval.value, steps_per_eval=10**9, max_eval_batches=1)),
            checkpointer=CheckpointerConfig(
                base_path=os.path.join(probe_root, "checkpoints"),
                temporary_base_path=os.path.join(probe_root, "checkpoints-temp"),
                append_run_id_to_base_path=False,
                save_interval=timedelta(days=7),
            ),
        )
    tracker = dataclasses.replace(
        cfg.tracker,
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "marin-della"),
        group=group,
        tags=[*cfg.tracker.tags, "della4xh100", _ATTN, *filter(None, [lever]), *(["probe"] if _PROBE_STEPS else [])],
    )
    config = dataclasses.replace(
        cfg,
        model=versioned(model),
        grug_trainer=versioned(grug_trainer),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_gpu("H100", count=_NUM_GPUS)),
        tracker=tracker,
        # A 5-step trace well after compile (logs/<run_id>/profiler) shows where step time goes that the
        # matmul-only MFU does not count (128k-vocab softmax, expert dispatch, optimizer, collectives).
        profiler=ProfilerConfig(enabled=not _PROBE_STEPS, start_step=500, num_steps=5),
        **overrides,
    )
    return dataclasses.replace(step, name=f"grug/{run_id}", config=config)


if __name__ == "__main__":
    ((_, batch_size, num_steps),) = [p for p in _POINTS if p[0] == _DIM]
    executor_main(steps=[_della_step(_DIM, batch_size, num_steps)], description=f"July Baseline d{_DIM} on Della 4xH100")
