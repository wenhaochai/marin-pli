#!/bin/bash
#SBATCH --job-name=speedall-par1
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=96G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10
# Up to four one-GPU A/B arms side by side, one per GPU (scripts/della/speed_all/ab_arms1.sbatch each, with its own
# job id suffix g<i> and so its own TMPDIR), then each arm's train-loss pair against the first arm. The controller asked
# for this on 2026-10-26: one-GPU jobs queued one at a time left the other three GPUs of a 4-GPU block idle.
# Usage: ab_par1.sbatch <config> <arm0> [arm1 arm2 arm3]
set -uo pipefail
CFG=${1:?usage: ab_par1.sbatch <config> <arm0> [arm1 arm2 arm3]}
shift
ARMS=("$@")
[ ${#ARMS[@]} -ge 1 ] && [ ${#ARMS[@]} -le 4 ] || { echo "1 to 4 arms"; exit 2; }
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/async
PY=$MAIN/.venv/bin/python
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
LOGDIR=$MAIN/logs/slurm
cd "$WT"
source scripts/della/wandb_guard.sh
nvidia-smi -L
echo "PAR START $(date +%H:%M:%S) $(git -C $WT log --oneline -1)"
pids=()
for i in "${!ARMS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i FETCHED_JOB_ID=${JID}g$i bash scripts/della/speed_all/ab_arms1.sbatch "$CFG" "${ARMS[$i]}" \
    > $LOGDIR/speedall-par1_${JID}_g$i.out 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
declare -A RID
for i in "${!ARMS[@]}"; do
  grep -E "^SUMMARY|^ARM .* dry run|SMOKE DONE" $LOGDIR/speedall-par1_${JID}_g$i.out
  RID[$i]=$(sed -n 's/^SUMMARY \([^ ]*\) rc=0 .*/\1/p' $LOGDIR/speedall-par1_${JID}_g$i.out | head -1)
done
for i in "${!ARMS[@]}"; do
  [ "$i" -gt 0 ] && [ -n "${RID[0]}" ] && [ -n "${RID[$i]}" ] || continue
  $PY - "${RID[0]}" "${RID[$i]}" <<'PY'
import sys
import numpy as np
import wandb
a, b = sys.argv[1:3]
try:
    api = wandb.Api(timeout=60)
    h = {r: {x["_step"]: x["train/loss"] for x in api.run(f"marin-della/{r}").scan_history(keys=["_step", "train/loss"])} for r in (a, b)}
    steps = sorted(set(h[a]) & set(h[b]))
    d = np.array([h[b][t] - h[a][t] for t in steps])
    print(f"PAIR {b} - {a}: steps={len(steps)} max_abs_loss_diff={np.max(np.abs(d)):.3e} mean_diff={d.mean():+.5f} "
          f"last10_mean_diff={d[-10:].mean():+.5f}", flush=True)
except Exception as e:
    print(f"PAIR {a} {b}: wandb_error={str(e)[:160]}", flush=True)
PY
done
echo "PAR DONE $(date +%H:%M:%S) fail=$fail"
exit $fail
