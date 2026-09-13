# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi scaling-suite rungs on Della 4xH100.

Reproduces the released compute-optimal Delphi runs at 3e18, 9e18 and 2e19 FLOPs (W&B project
``marin-community/marin``, ``isoflop-*-adamh_scaling_v6``). Model shape, batch, step count, optimizer
values, z-loss and checkpoint policy are transcribed from those runs' logged configs rather than
re-derived, so later heuristic drift cannot leak in. The originals trained on 4 TPU v4 chips; this
launcher keeps the same per-device batch on 4 H100s and only changes the accelerator plumbing.

    BUDGET=3e18 python -m experiments.references.della_delphi_ladder        # DRY_RUN=1 prints the plan
"""

import dataclasses
import os

import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.attention import AttentionBackend
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.execution.remote import remote
from marin.experiment.data import mixture
from marin.training import training as _training
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, resolve_training_env, run_levanter_train_lm

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.llama import llama3_tokenizer
from experiments.references.completed_adamh import SEQ_LEN, completed_adamh_heuristic

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
# Transcribed from the original runs' W&B configs (isoflop-<budget>-d<hidden>-L<layers>-B<batch>-adamh_scaling_v6).
RUNGS = {
    "3e18": dict(hidden=1024, batch=8, steps=37335, tokens=1_223_393_280, lr=0.00275999052746201, adam_lr=0.00033154735825338737, eps=3.6604122149949323e-08, ref_macro_loss=3.58973),
    "9e18": dict(hidden=1152, batch=16, steps=44317, tokens=2_904_358_912, lr=0.003011461785505323, adam_lr=0.000304311605183897, eps=3.988017477238883e-08, ref_macro_loss=3.38401),
    "2e19": dict(hidden=1408, batch=16, steps=55125, tokens=3_612_672_000, lr=0.002820610414485248, adam_lr=0.0002728525996258053, eps=4.4478227499549274e-08, ref_macro_loss=3.27437),
}
# Same Nemotron mixture as the originals (weights logged in their W&B configs).
NEMOTRON_WEIGHTS = {"hq_actual": 0.91351, "hq_synth": 2.72, "medium_high": 0.82471, "medium": 3.38, "medium_low": 1.54, "low_actual": 0.70123, "low_synth": 0.62771}


def _train_data():
    nem = nemotron_datasets(tokenizer=llama3_tokenizer)
    train = {nem[split]: w for split, w in NEMOTRON_WEIGHTS.items()}
    train[starcoder_dataset(tokenizer=llama3_tokenizer)] = 0.25
    train[proofpile_dataset(tokenizer=llama3_tokenizer)] = 0.055
    validation = [*paloma_datasets().values(), *uncheatable_datasets().values()]
    return train, validation


def _run_rung(config: TrainLmOnPodConfig) -> None:
    env = resolve_training_env(config.env_vars, config.resources)
    remote(run_levanter_train_lm, resources=config.resources, env_vars=env)(dataclasses.replace(config, env_vars=env))


def delphi_rung(budget: str) -> ArtifactStep[LevanterCheckpoint]:
    r = RUNGS[budget]
    run_id = f"delphi-{budget}-della4xh100"
    train, validation = _train_data()
    model = completed_adamh_heuristic._build_model_config(r["hidden"], seq_len=SEQ_LEN)
    # No Transformer Engine on Della, so the GPU default (NVTE) would fall back to unfused O(S^2) attention.
    model = dataclasses.replace(model, attn_backend=AttentionBackend.JAX_FLASH)
    optimizer = dataclasses.replace(
        completed_adamh_heuristic.build_optimizer_config(r["batch"], r["tokens"]),
        learning_rate=r["lr"],
        adam_lr=r["adam_lr"],
        epsilon=r["eps"],
        max_grad_norm=0.1,
        weight_decay=0.1,
    )

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        inner = train_lm.TrainLmConfig(
            data=mixture(ctx, train, validation=validation),
            trainer=TrainerConfig(
                tracker=WandbConfig(
                    entity=os.environ.get("WANDB_ENTITY"),
                    project=os.environ.get("WANDB_PROJECT", "marin-della"),
                    group="delphi-della",
                    tags=["delphi", "completed-adamh", f"FLOPs={budget}", "della4xh100", "jax_flash"],
                ),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=r["batch"],
                per_device_parallelism=-1,
                num_train_steps=r["steps"],
                steps_per_eval=1000,
                checkpointer=CheckpointerConfig(save_interval=__import__("datetime").timedelta(minutes=10), keep=[dict(every=10000)]),
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
            z_loss_weight=1e-7,
            # The periodic export resolves Qwen/Qwen3-0.6B from the Hub, which compute nodes cannot reach;
            # keep only the end-of-run export.
            hf_save_steps=10**9,
        )
        return TrainLmOnPodConfig(
            train_config=inner,
            resources=ResourceConfig.with_gpu("H100", count=NUM_GPUS),
            output_path=ctx.output_path,
            env_vars={"RUN_ID": run_id},
        )

    return ArtifactStep(
        name=f"delphi/{run_id}",
        version=VERSION,
        artifact_type=LevanterCheckpoint,
        run=remote(_run_rung, resources=ResourceConfig.with_cpu()),
        build_config=build_config,
        deps=(*train, *validation),
    )


if __name__ == "__main__":
    step = delphi_rung(os.environ["BUDGET"])
    if os.environ.get("DRY_RUN") == "1":
        spec = lower(step)
        print("output:", spec.override_output_path or spec.name)
        print("deps:", len(spec.deps))
    else:
        run(step)
