#!/bin/bash
#SBATCH --job-name=ov-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:01:00
#SBATCH --output=logs/slurm/%x_%j.out

# OV (over-encoding + over-decoding) H100 smoke: for each SIZE in SIZES (default 130m 300m) and VARIANT in ov ovss, SMOKE_STEPS
# steps (default 60, so every ovss stage runs) at the real batch plus a one-batch eval. A SUMMARY line per run gives the
# exit code, the peak GPU memory (nvidia-smi, sampled every 2 s), the mean step time, the ntp/od losses and the eval loss.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
module load proxy/default  # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export SMOKE_STEPS=${SMOKE_STEPS:-60} RUN_TAG=-j${SLURM_JOB_ID}
nvidia-smi -L
for SIZE in ${SIZES:-130m 300m}; do
  for VARIANT in ov ovss; do
    export SIZE VARIANT
    run_id=$(DRY_RUN=1 .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p')
    log=logs/slurm/ov-smoke_${SLURM_JOB_ID}_${run_id}.log
    echo "SMOKE $run_id START $(date +%H:%M:%S)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > $log.mem &
    mpid=$!
    timeout --signal=TERM 20m .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
    rc=$?
    kill $mpid
    peak=$(sort -n $log.mem | tail -1)
    err=$(grep -m1 -E "RESOURCE_EXHAUSTED|out of memory|Traceback" $log | cut -c1-160)
    .venv/bin/python - "$run_id" "$rc" "$err" "$peak" <<'PY'
import sys
import numpy as np
import wandb
run_id, rc, err, peak = sys.argv[1:5]
try:
    run = wandb.Api(timeout=60).run(f"marin-della/{run_id}")
    s = run.summary
    d = [r["throughput/duration"] for r in run.scan_history(keys=["_step", "throughput/duration"]) if r["_step"] >= 3]
    print(f"SUMMARY {run_id} rc={rc} peak_mem_MiB={peak} step_ms={np.mean(d) * 1e3:.0f}(n={len(d)}) ntp={s.get('train/ntp_loss')} od={s.get('train/od_loss')} "
          f"loss={s.get('train/loss')} eval_loss={s.get('eval/loss')} eval_macro={s.get('eval/paloma/macro_loss')} err='{err}'", flush=True)
except Exception as e:
    print(f"SUMMARY {run_id} rc={rc} peak_mem_MiB={peak} wandb_error={str(e)[:160]} err='{err}'", flush=True)
PY
    rm -rf "$MARIN_PREFIX/speedrun/${run_id}"
  done
done
echo "SMOKE DONE $(date +%H:%M:%S)"
