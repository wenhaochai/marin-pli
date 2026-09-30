#!/bin/bash
#SBATCH --job-name=pls-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# H100 smoke of VARIANT=pls: for each SIZE:PLS_W in CONFIGS, SMOKE_STEPS steps at the real batch, one eval batch (main +
# per-layer readouts) and a final checkpoint. A SUMMARY line per config gives the exit code, the first OOM/traceback line,
# W&B MFU and tokens/s, and eval/loss next to eval/L{k}/loss (the last layer's must equal eval/loss). Outputs are deleted.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh

module load proxy/default  # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export SMOKE_STEPS=${SMOKE_STEPS:-40}
export VARIANT=pls
nvidia-smi -L

for cfg in ${CONFIGS:-130m:1 300m:1}; do
  export SIZE=${cfg%%:*} PLS_W=${cfg##*:}
  log=logs/slurm/pls-smoke_${SLURM_JOB_ID}_${SIZE}_pls${PLS_W}.log
  echo "SMOKE $cfg START $(date +%H:%M:%S)"
  timeout --signal=TERM 14m $PY -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
  rc=$?
  err=$(grep -m1 -E "RESOURCE_EXHAUSTED|out of memory|Traceback" $log | cut -c1-160)
  $PY - "$SIZE" "$PLS_W" "$rc" "$err" <<'PYEOF'
import os, sys, wandb
size, w, rc, err = sys.argv[1:5]
run_id = f"muonh-qwen3-{size}-della4xh100-pls{w}-smoke{os.environ['SMOKE_STEPS']}"
try:
    s = wandb.Api(timeout=60).run(f"marin-della/{run_id}").summary
    layers = sorted(int(k.split("/")[1][1:]) for k in s.keys() if k.startswith("eval/L") and k.endswith("/loss") and k.count("/") == 2)
    ev = " ".join(f"L{k}={s.get(f'eval/L{k}/loss'):.4f}" for k in layers)
    tr = " ".join(f"L{k}={s.get(f'train/pls/L{k}'):.4f}" for k in layers if s.get(f"train/pls/L{k}") is not None)
    print(f"SUMMARY size={size} pls{w} rc={rc} mfu={s.get('throughput/mfu')} tok_s={s.get('throughput/tokens_per_second')} "
          f"train/loss={s.get('train/loss')} eval/loss={s.get('eval/loss')} | eval {ev} | train {tr} | err='{err}'", flush=True)
except Exception as e:
    print(f"SUMMARY size={size} pls{w} rc={rc} err='{err}' wandb_error={e!r}", flush=True)
PYEOF
  rm -rf "$MARIN_PREFIX/speedrun/muonh-qwen3-${SIZE}-della4xh100-pls${PLS_W}-smoke${SMOKE_STEPS}"
done
echo "SMOKE DONE $(date +%H:%M:%S)"
