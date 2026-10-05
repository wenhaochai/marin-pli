#!/bin/bash
#SBATCH --job-name=ov-allocdiag
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:15:00
#SBATCH --output=logs/slurm/%x_%j.out
# Which XLA allocator survives the 1.2B 8-GPU OV memory pattern? Persistent state is ~27 GiB per device and each train
# step allocates one 25-30 GiB temp buffer; under the default BFC allocator a long-lived small block lands next to it
# and the next step finds no contiguous block (ovss died at step ~130, ov at ~1,290; 2026-10-01). Runs the 1.2B ovss
# config (the faster reproducer) for SMOKE_STEPS steps under each allocator config in turn, with MEMSTATS every 30 s.
# A DIAG line per config: rc, OOM count, last progress, last MEMSTATS.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
module load proxy/default  # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della SIZE=1_2b VARIANT=${VARIANT:-ovss} NUM_GPUS=8 PDP=16 MEMSTATS_EVERY=30
export SMOKE_STEPS=${SMOKE_STEPS:-300}
nvidia-smi -L
CONFIGS=${CONFIGS:-"bfc90 async90np async75 vmm90"}
for c in $CONFIGS; do
  case $c in
    bfc90)   envs="XLA_PYTHON_CLIENT_MEM_FRACTION=0.9" ;;
    async90np) envs="XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_PYTHON_CLIENT_PREALLOCATE=false" ;;
    async75) envs="XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.75" ;;
    vmm90)   envs="XLA_PYTHON_CLIENT_ALLOCATOR=vmm XLA_PYTHON_CLIENT_MEM_FRACTION=0.9" ;;
    *) echo "unknown config $c"; continue ;;
  esac
  export RUN_TAG=-j${SLURM_JOB_ID}$c
  run_id=$(DRY_RUN=1 .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p')
  [ -n "$run_id" ] || { echo "DIAG $c: the dry run gave no run id"; continue; }
  log=logs/slurm/ov-allocdiag_${SLURM_JOB_ID}_$c.log
  echo "DIAG $c START $(date +%H:%M:%S) $envs"
  env $envs timeout --signal=TERM ${RUN_TIMEOUT:-32m} .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
  rc=$?
  oom=$(grep -a -c -E "RESOURCE_EXHAUSTED|ran out of memory" $log)
  prog=$(tac $log | grep -am1 -oE "train [0-9.k]+it/[0-9.k]+it rate:[0-9.]+s/it")
  mem=$(tac $log | grep -am1 "^MEMSTATS" | cut -c1-200)
  minfree=$(grep -a "^MEMSTATS" $log | grep -oE "largest_free_block_bytes=[0-9.]+" | cut -d= -f2 | sort -g | head -1)
  echo "DIAG $c rc=$rc oom_lines=$oom progress='$prog' min_largest_free=${minfree:-NA}GiB last='$mem'"
  case "$run_id" in *-j${SLURM_JOB_ID}*) rm -rf "$MARIN_PREFIX/speedrun/$run_id" ;; *) echo "kept speedrun/'$run_id': not a run of this job" ;; esac   # never the whole speedrun/ (empty run_id)
done
echo "DIAG DONE $(date +%H:%M:%S)"
