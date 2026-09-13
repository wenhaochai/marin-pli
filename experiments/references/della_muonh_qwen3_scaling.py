# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kaiyue Wen's ``muonh_qwen3_scaling`` speedrun baselines on Della 4xH100.

Reproduces the released Qwen3 (LLaMA-geometry-matched) MuonH runs at 130m, 300m, 520m and 1_2b (W&B
``marin-community/marin`` ``qwen3_<size>_muonh_4096_*``, 2025-10-12, 64 TPU devices). Shapes, batch, steps and
optimizer values are transcribed from those runs' logged configs. Current main matches their training math except
for two defaults set explicitly here: the Newton-Schulz coefficients (main switched to per-iteration quintic
coefficients in 2026-01) and the data order (the originals used a full linear permutation with data seed 42; main
defaults to a Feistel block shuffle). Paloma is tokenized with the training tokenizer, as in the originals.
SMOKE_STEPS turns a run into a short smoke test with its own run id and output, evaluating one batch at the end.

    SIZE=130m python -m experiments.references.della_muonh_qwen3_scaling        # DRY_RUN=1 prints the plan
"""

import dataclasses
import os
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.attention import AttentionBackend
from levanter.main import train_lm
from levanter.models.qwen import Qwen3Config
from levanter.optim.muonh import MuonHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.execution.remote import remote
from marin.experiment.data import mixture
from marin.training import training as _training
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, resolve_training_env, run_levanter_train_lm

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.prebuilt_caches import fineweb_edu_10B_dataset
from experiments.marin_tokenizer import marin_tokenizer

# With a local MARIN_PREFIX the temporary checkpoint base comes back as a file:// URL, which the tensorstore
# writer treats as a relative path (arrays land in <cwd>/file:/...), so resume finds only metadata.json.
_temp_ckpt_base = _training.temporary_checkpoint_base_path
_training.temporary_checkpoint_base_path = lambda output_path: _temp_ckpt_base(output_path).removeprefix("file://")

if "della-proxy" in os.environ.get("https_proxy", ""):
    # Della's compute-node proxy passes api.wandb.ai but not storage.googleapis.com: file/artifact uploads
    # (levanter logs requirements.txt as an artifact) hang run.finish(). Metrics still stream.
    os.environ.update(WANDB_IGNORE_GLOBS="*", WANDB_DISABLE_CODE="true", WANDB_CONSOLE="off", WANDB_X_DISABLE_META="true")
    import wandb.sdk.wandb_run

    wandb.sdk.wandb_run.Run.log_artifact = lambda self, *args, **kwargs: None

VERSION = "2026.09.13"
NUM_GPUS = 4
SEQ_LEN = 4096
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "0"))
# Transcribed from the original runs' W&B configs; ref_c4_en_bpb is their final eval/paloma/c4_en/bpb.
SIZES = {
    "130m": dict(hidden=512, inter=1792, layers=6, heads=8, kv=8, batch=128, steps=4959, lr=0.02, adam_lr=0.008, eps=1e-20, momentum=0.95, schedule="linear", decay=0.8, warmup=0, max_grad_norm=1.0, ref_c4_en_bpb=1.16354),
    "300m": dict(hidden=768, inter=2688, layers=12, heads=12, kv=12, batch=128, steps=11444, lr=0.01, adam_lr=0.002, eps=1e-15, momentum=0.98, schedule="cosine", decay=None, warmup=1000, max_grad_norm=1.0, ref_c4_en_bpb=1.05602),
    "520m": dict(hidden=1024, inter=3584, layers=24, heads=16, kv=8, batch=256, steps=9918, lr=0.01, adam_lr=0.002, eps=1e-15, momentum=0.98, schedule="cosine", decay=None, warmup=1000, max_grad_norm=1.0, ref_c4_en_bpb=0.98824),
    "1_2b": dict(hidden=2048, inter=7168, layers=16, heads=16, kv=8, batch=256, steps=22888, lr=0.01, adam_lr=0.0015, eps=1e-15, momentum=0.98, schedule="cosine", decay=None, warmup=1000, max_grad_norm=2.0, ref_c4_en_bpb=0.92731),
}


def _run_size(config: TrainLmOnPodConfig) -> None:
    env = resolve_training_env(config.env_vars, config.resources)
    remote(run_levanter_train_lm, resources=config.resources, env_vars=env)(dataclasses.replace(config, env_vars=env))


def muonh_qwen3_run(size: str) -> ArtifactStep[LevanterCheckpoint]:
    s = SIZES[size]
    run_id = f"muonh-qwen3-{size}-della4xh100" + (f"-smoke{SMOKE_STEPS}" if SMOKE_STEPS else "")
    train = {fineweb_edu_10B_dataset(): 1.0}
    validation = list(paloma_datasets(tokenizer=marin_tokenizer).values())
    model = Qwen3Config(
        max_seq_len=SEQ_LEN,
        hidden_dim=s["hidden"],
        intermediate_dim=s["inter"],
        num_layers=s["layers"],
        num_heads=s["heads"],
        num_kv_heads=s["kv"],
        hybrid_norm=True,
        # No Transformer Engine on Della, so the GPU default (NVTE) would fall back to unfused O(S^2) attention.
        attn_backend=AttentionBackend.JAX_FLASH,
    )
    optimizer = MuonHConfig(
        learning_rate=s["lr"],
        adam_lr=s["adam_lr"],
        beta1=0.9,
        beta2=0.98,
        epsilon=s["eps"],
        momentum=s["momentum"],
        nesterov=True,
        backend_steps=5,
        muon_epsilon=1e-5,
        max_grad_norm=s["max_grad_norm"],
        weight_decay=0.1,
        lr_schedule=s["schedule"],
        decay=s["decay"],
        warmup=s["warmup"],
        min_lr_ratio=0.0,
        # The originals ran the fixed (3.4445, -4.7750, 2.0315) iteration; main defaults to quintic coefficients.
        coefficient_type="simple",
    )

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        data = dataclasses.replace(mixture(ctx, train, validation=validation, shuffle=True), permutation_type="linear")
        inner = train_lm.TrainLmConfig(
            data=data,
            trainer=TrainerConfig(
                tracker=WandbConfig(
                    entity=os.environ.get("WANDB_ENTITY"),
                    project=os.environ.get("WANDB_PROJECT", "marin-della"),
                    group="muonh-qwen3-smoke" if SMOKE_STEPS else "muonh-qwen3-della",
                    tags=["speedrun", "muonh", "qwen3", size, "della4xh100", "jax_flash"],
                ),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=s["batch"],
                per_device_parallelism=-1,
                num_train_steps=SMOKE_STEPS or s["steps"],
                steps_per_eval=SMOKE_STEPS or 1000,
                max_eval_batches=1 if SMOKE_STEPS else None,
                checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10), keep=[dict(every=10000)]),
                mesh=MeshConfig(
                    axes={"data": -1, "replica": 1, "model": 1},
                    compute_mapping={
                        "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                        "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    },
                ),
                seed=0,
                allow_nondivisible_batch_size=True,
            ),
            train_seq_len=SEQ_LEN,
            model=model,
            optimizer=optimizer,
            z_loss_weight=0.0,
            data_seed=42,
            # No HF export: compute nodes cannot reach the Hub, and loading the tokenizer-only marin-tokenizer repo through
            # AutoTokenizer needs a config.json lookup that fails offline. The levanter checkpoint is the artifact.
            hf_save_steps=None,
        )
        return TrainLmOnPodConfig(
            train_config=inner,
            resources=ResourceConfig.with_gpu("H100", count=NUM_GPUS),
            output_path=ctx.output_path,
            env_vars={"RUN_ID": run_id},
        )

    return ArtifactStep(
        name=f"speedrun/{run_id}",
        version=VERSION,
        artifact_type=LevanterCheckpoint,
        run=remote(_run_size, resources=ResourceConfig.with_cpu()),
        build_config=build_config,
        deps=(*train, *validation),
    )


if __name__ == "__main__":
    step = muonh_qwen3_run(os.environ["SIZE"])
    if os.environ.get("DRY_RUN") == "1":
        spec = lower(step)
        print("output:", spec.override_output_path or spec.name)
        print("deps:", len(spec.deps))
    else:
        run(step)
