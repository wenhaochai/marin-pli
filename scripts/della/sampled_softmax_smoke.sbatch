#!/bin/bash
#SBATCH --job-name=ss-smoke
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

# Sampled-softmax H100 check: scripts/della/sampled_softmax_gpu_check.py on one GPU (kernel parity and timing at the real
# device shape), then SIZE (default 130m) smokes of VARIANT=ss and VARIANT=baseline for SMOKE_STEPS steps (default 60, so
# every SS_SCHEDULE stage runs), each ending in a one-batch full-softmax eval. A SUMMARY line per run gives the exit code,
# the mean step time per candidate count, the ss metrics and the eval loss; each smoke output is deleted afterwards.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin

module load proxy/default  # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export SMOKE_STEPS=${SMOKE_STEPS:-60} SIZE=${SIZE:-130m} RUN_TAG=-j${SLURM_JOB_ID}
nvidia-smi -L

echo "GPU CHECK START $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM 15m .venv/bin/python scripts/della/sampled_softmax_gpu_check.py 2>&1 | grep -v -E "^(W|I)[0-9]{4} "
echo "GPU CHECK rc=${PIPESTATUS[0]} $(date +%H:%M:%S)"

for VARIANT in ss baseline; do
  export VARIANT
  run_id=$(DRY_RUN=1 .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p')
  log=logs/slurm/ss-smoke_${SLURM_JOB_ID}_${run_id}.log
  echo "SMOKE $run_id START $(date +%H:%M:%S)"
  timeout --signal=TERM 20m .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
  rc=$?
  err=$(grep -m1 -E "RESOURCE_EXHAUSTED|out of memory|Traceback" $log | cut -c1-160)
  .venv/bin/python - "$run_id" "$rc" "$err" <<'EOF'
import sys
import numpy as np
import wandb

run_id, rc, err = sys.argv[1:4]
try:
    run = wandb.Api(timeout=60).run(f"marin-della/{run_id}")
    s = run.summary
    rows = list(run.scan_history(keys=["_step", "throughput/duration", "train/loss"]))
    ss = {r["_step"]: r for r in run.scan_history(keys=["_step", "train/ss/candidates", "train/ss/present_max", "train/ss/overflow_frac"])}
    by_p = {}
    for r in rows:
        if r["_step"] >= 3:  # the first steps carry compilation
            by_p.setdefault(int(ss[r["_step"]]["train/ss/candidates"]) if r["_step"] in ss else 0, []).append(r["throughput/duration"])
    times = " ".join(f"P={p or 'full'}:{np.mean(v) * 1e3:.0f}ms(n={len(v)})" for p, v in sorted(by_p.items()))
    extra = ""
    if ss:
        extra = f" present_max={max(r['train/ss/present_max'] for r in ss.values()):.0f} overflow_max={max(r['train/ss/overflow_frac'] for r in ss.values()):.3f}"
        extra += " stages=" + ",".join(f"{st}:{int(r['train/ss/candidates'])}" for st, r in sorted(ss.items()) if st == 0 or int(r["train/ss/candidates"]) != int(ss.get(st - 1, r)["train/ss/candidates"]))
    print(f"SUMMARY {run_id} rc={rc} steps={len(rows)} step_time[{times}] train_loss_last={s.get('train/loss')} eval_loss={s.get('eval/loss')} eval_macro={s.get('eval/paloma/macro_loss')}{extra} err='{err}'", flush=True)
except Exception as e:
    print(f"SUMMARY {run_id} rc={rc} wandb_error={str(e)[:160]} err='{err}'", flush=True)
EOF
  rm -rf "$MARIN_PREFIX/speedrun/${run_id}"
done
echo "SMOKE DONE $(date +%H:%M:%S)"
