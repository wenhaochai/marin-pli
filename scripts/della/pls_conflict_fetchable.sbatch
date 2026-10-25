#!/bin/bash
#SBATCH --job-name=pls-conflict
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:10:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (the checkpoint set comes from the job name)
# Gradient conflict between the intermediate layer losses and the last-layer loss on finished 48-layer separate-heads
# checkpoints (tmp/pls_conflict/conflict.py, read-only, no training): one checkpoint per GPU, 4 side by side.
# pls-conflict-a: Pre-LN (readouts see all, 1/2, 1/4) and Pre-LN seed 1; pls-conflict-b: HC 48 / 24 / 12 layers, mHC.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
module load proxy/default 2>/dev/null || true
if [ -n "${FETCHED_JOB_ID:-}" ]; then NAME=${FETCHED_JOB_NAME:-pls-conflict-a}; else NAME=$SLURM_JOB_NAME; fi
P=130m-della4xh100-pls1-sep
case $NAME in
  pls-conflict-a) R=($P-apreln-vs-d48-g4 $P-apreln-vis0.5-vs-d48 $P-apreln-vis0.25-vs-d48 $P-apreln-vs-d48-s1);;
  pls-conflict-b) R=($P-ahc-vs-d48-g4 $P-ahc-vs-d24 $P-ahc-vs-d12 $P-amhc-vs-d48-g4);;
  *) echo "unknown set '$NAME'"; exit 2;;
esac
O=/scratch/gpfs/GROUP/USER/tmp/pls_conflict/gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g timeout 62m "$PY" /scratch/gpfs/GROUP/USER/tmp/pls_conflict/conflict.py ${R[$g]} $O/${R[$g]}.json --b 8 --t 1024 > $O/${R[$g]}.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=$?; done
echo "done rc=$rc"; exit 0
