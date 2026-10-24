#!/bin/bash
#SBATCH --job-name=cbw-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=00:30:00
#SBATCH --exclude=della-k14g3,della-k17g3
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out

# H100 smoke + timing of XLA_CMD_BUFFER_WHILE (speed-command-buffer-while): the same SMOKE_STEPS run of 300m v64k at the
# character window SEQ_LEN=4224 (flash block 128, the charwin-v64k config), first with the flag off, then on, through
# this branch's scripts/della/muonh_qwen3_h100x4.sbatch (training code: the main checkout, unchanged). Per arm: exit
# code, s/it from the tqdm lines after compile, the smoke's final loss from W&B when reachable, and the XLA_FLAGS line.
# Pass: both arms exit 0, the cbw arm prints its XLA_FLAGS line, losses agree within the base run-to-run band.
# Run ids -j<job>-base / -j<job>-cbw; their checkpoints are deleted at the end (cbw_cleanup.sh).
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
module load proxy/default   # W&B summary below (the launcher loads it for training)
unset XLA_FLAGS SIZE VOCAB_K SEQ_LEN DATA_EPOCHS SEED BATCH RUN_TAG VARIANT PASS_DATA PDP PRECISION TOTAL_STEPS INIT_FROM XLA_CMD_BUFFER_WHILE   # no knob from the submitting shell
AB=/scratch/gpfs/GROUP/USER/tmp/speed_wt/command-buffer-while/scripts/della
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
CONF="SIZE=300m VOCAB_K=64000 SEQ_LEN=4224"
export SMOKE_STEPS=150
fail=0; ids=()
for arm in base cbw; do
  flag=0; [ $arm = cbw ] && flag=1
  env $CONF RUN_TAG=-j$JID-$arm DRY_RUN=1 .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling > $TMPDIR/dry.$arm 2>/dev/null
  run_id=$(sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' $TMPDIR/dry.$arm | head -1)
  [ -n "$run_id" ] || { echo "ARM $arm: dry run gave no run id"; fail=1; continue; }
  ids+=("$run_id")
  log=logs/slurm/cbw-smoke_${JID}_$arm.log
  echo "ARM $arm $run_id START $(date +%H:%M:%S)"
  timeout --signal=TERM 13m scripts/della/pgroup.sh env $CONF RUN_TAG=-j$JID-$arm XLA_CMD_BUFFER_WHILE=$flag bash $AB/muonh_qwen3_h100x4.sbatch > $log 2>&1
  rc=$?
  err=$(grep -a -m1 -E "RESOURCE_EXHAUSTED|out of memory|Traceback|CUDA_ERROR|INTERNAL:" $log | cut -c1-200)
  { [ $rc -ne 0 ] || [ -n "$err" ]; } && fail=1
  echo "ARM $arm rc=$rc rate=[$(bash $AB/ab/cbw_rate.sh $log 20)] flags=[$(grep -a -m1 '^XLA_CMD_BUFFER_WHILE=1' $log)] err='$err'"
  .venv/bin/python - "$run_id" <<'PY' || true
import sys
import numpy as np
import wandb
try:
    r = wandb.Api(timeout=60).run(f"marin-della/{sys.argv[1]}")
    h = [x for x in r.scan_history(keys=["_step", "train/loss", "throughput/duration"])]
    d = [x["throughput/duration"] for x in h if x["_step"] >= 20]
    print(f"WANDB {sys.argv[1]} step_s_mean={np.mean(d):.4f}(n={len(d)}) loss@10={[x['train/loss'] for x in h if x['_step'] == 10]} "
          f"loss@last={h[-1]['train/loss'] if h else None} (step {h[-1]['_step'] if h else None})", flush=True)
except Exception as e:
    print(f"WANDB {sys.argv[1]} unavailable: {str(e)[:160]}", flush=True)
PY
done
bash $AB/ab/cbw_cleanup.sh "${ids[@]}" || true
echo "CBW SMOKE DONE $(date +%H:%M:%S) fail=$fail"
exit $fail
