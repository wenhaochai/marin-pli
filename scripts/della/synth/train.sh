#!/bin/bash
# One training run of the synth-pretrain worktree (della_muonh_qwen3_scaling.py, this branch's code): the knobs come as
# VAR=value from the plan line (SIZE, SYNTH_MIX, SYNTH_LABEL, TOTAL_STEPS, SMOKE_STEPS, RUN_TAG). Same environment as
# scripts/della/muonh_qwen3_h100x4.sbatch, with this worktree's sources on PYTHONPATH (the shared project/marin/.venv
# otherwise imports project/marin's lib) and the speed settings of speed_env.sh.
set -euo pipefail
WT=/scratch/gpfs/KARTHIKN/wc9403/project/marin-synth
PY=/scratch/gpfs/KARTHIKN/wc9403/project/marin/.venv/bin/python
cd $WT
source scripts/della/job_tmpdir.sh
module load proxy/default  # wandb only
source scripts/della/wandb_guard.sh
_pp=$WT; for d in "$WT"/lib/*/src; do _pp=$_pp:$d; done; export PYTHONPATH=$_pp
export MARIN_PREFIX=/scratch/gpfs/KARTHIKN/wc9403/marin_store_big
export HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export SIZE=${SIZE:-130m}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9}
source scripts/della/speed_env.sh
echo "code $(git -C $WT log --oneline -1 | cut -c1-80)"
nvidia-smi -L
$PY -c 'import jax; print("jax", jax.__version__, "devices", jax.devices())'
exec $PY -m experiments.references.della_muonh_qwen3_scaling
