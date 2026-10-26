#!/bin/bash
#SBATCH --job-name=muonh-qwen3
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# One muonh_qwen3_scaling size (SIZE=130m|300m|520m|1_2b) on 4xH100: single JAX process, Fray LocalClient.
# Data must already be under MARIN_PREFIX: the fineweb-edu-10B prebuilt cache and Paloma tokenized with the
# marin tokenizer (both fetched/built on a login node).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh   # TMPDIR under scratch, per task, removed at exit (never the shared /tmp)

module load proxy/default  # wandb only
source scripts/della/wandb_guard.sh   # offline W&B when the proxy cannot reach api.wandb.ai (2026-10-16)
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export SIZE=${SIZE:-130m}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9}  # never inherited from the submitting shell (a stray .50 there halved the GPUs, 2026-10-04); XLA_PYTHON_CLIENT_ALLOCATOR passes through
# XLA_CMD_BUFFER_WHILE=1 (speed, 2026-10-24; default 1 since 2026-10-26 after the H100 A/B w203/w204, 0 = the old launch):
# XLA also captures while/conditional loops in CUDA graphs. The JAX_FLASH inner causal loops have data-dependent trip
# counts; without this each iteration copies its predicate to the host and relaunches its kernels (~55 us per iteration in
# the charwin logs: 20K-34K iterations per step at flash block 128). Same kernels in the same order, so the same math. The
# NCCL watchdog flag is repeated here because marin's training env adds it only when XLA_FLAGS is unset
# (os.environ.setdefault), so the job keeps it.
if [ "${XLA_CMD_BUFFER_WHILE:-1}" = 1 ]; then
  case " ${XLA_FLAGS:-} " in *xla_gpu_nccl_termination_timeout_seconds*) ;; *) XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_nccl_termination_timeout_seconds=600" ;; esac
  export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL"
  echo "XLA_CMD_BUFFER_WHILE=1: XLA_FLAGS=$XLA_FLAGS"
fi
# LEVANTER_CKPT_DEADLINE (speed, 2026-10-25; levanter commit from the pls session): one temporary checkpoint 150 s before
# this job's end, SLURM_JOB_END_TIME (Slurm's own, or the host's end for a wq guest), so a resumable run that the end stops
# redoes almost no work on resume. No end time: no effect.
if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
  export LEVANTER_CKPT_DEADLINE=$((SLURM_JOB_END_TIME - 150))
  echo "LEVANTER_CKPT_DEADLINE=$LEVANTER_CKPT_DEADLINE ($(date -d @$LEVANTER_CKPT_DEADLINE +%F' '%T))"
fi
# LEVANTER_CE_BTILED_ONEHOT=1 (speed, 2026-10-25; the pls session's CE commit, bitwise equal to the scatter on H100 in its
# deterministic A/B): the cross-entropy label term as a select instead of a scatter, so XLA fuses the dlogits pass. The
# same additions on the same values: every run's numbers are unchanged.
export LEVANTER_CE_BTILED_ONEHOT=${LEVANTER_CE_BTILED_ONEHOT:-1}
# LEVANTER_CE_GRAD_IN_FWD=1 (speed, 2026-10-26; the pls session's CE commit): the fused cross-entropy computes its
# gradients in the forward from the same logits GEMM per block (3 logits GEMMs instead of 4). Not bitwise: H100 det round
# w245 at 300m 128K gives per-step loss within 1.9e-6 of the base over 58 steps, far inside the run-to-run spread of
# GEMM autotuning (pls w205 vs w229: 0.076). -4.7% step time on top of A1 under FA4 (w227).
export LEVANTER_CE_GRAD_IN_FWD=${LEVANTER_CE_GRAD_IN_FWD:-0}   # OFF again (2026-10-26 02:20): its fallback crashes under shard_map when a shard does not take the b-tiled path, which every vocabulary below 65536 does (pls bug report); back on after the fix

nvidia-smi -L
.venv/bin/python -c 'import jax; print("jax", jax.__version__, "devices", jax.devices())'

.venv/bin/python -m experiments.references.della_muonh_qwen3_scaling
