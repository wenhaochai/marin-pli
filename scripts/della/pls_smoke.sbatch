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
# per-layer readouts) and a final checkpoint. Run ids carry -j<jobid> (RUN_TAG), so a smoke never resumes an earlier one,
# and are read from the launcher's own dry run rather than rebuilt here. A SUMMARY line per config gives the exit code,
# the first OOM/traceback line, W&B MFU and tokens/s, and eval/loss next to eval/L{k}/loss (the last layer's must equal
# eval/loss). The smoke's output and temporary checkpoints are deleted afterwards (paths containing that exact run id).
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
unset TOTAL_STEPS INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG SEED DRY_RUN

module load proxy/default  # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export SMOKE_STEPS=${SMOKE_STEPS:-40}
[ "$SMOKE_STEPS" -gt 0 ] 2>/dev/null || { echo "SMOKE_STEPS must be a positive integer, got '$SMOKE_STEPS'"; exit 1; }
export VARIANT=pls RUN_TAG=-j${SLURM_JOB_ID}
nvidia-smi -L

for cfg in ${CONFIGS:-130m:1 300m:1}; do
  export SIZE=${cfg%%:*} PLS_W=${cfg##*:}
  out=$(DRY_RUN=1 $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's/^output: //p')
  run_id=$(basename "$(dirname "$out")")
  case "$run_id" in
    *-j${SLURM_JOB_ID}-smoke${SMOKE_STEPS}) ;;
    *) echo "SUMMARY cfg=$cfg bad run id '$run_id' from dry run output '$out'"; continue ;;
  esac
  log=logs/slurm/pls-smoke_${SLURM_JOB_ID}_${SIZE}_pls${PLS_W}.log
  echo "SMOKE $cfg $run_id START $(date +%H:%M:%S)"
  timeout --signal=TERM 14m $PY -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
  rc=$?
  err=$(grep -m1 -E "RESOURCE_EXHAUSTED|out of memory|Traceback" $log | cut -c1-160)
  $PY - "$run_id" "$cfg" "$rc" "$err" <<'PYEOF'
import sys, wandb
run_id, cfg, rc, err = sys.argv[1:5]
try:
    s = wandb.Api(timeout=60).run(f"marin-della/{run_id}").summary
    layers = sorted(int(k.split("/")[1][1:]) for k in s.keys() if k.startswith("eval/L") and k.endswith("/loss") and k.count("/") == 2)
    ev = " ".join(f"L{k}={s.get(f'eval/L{k}/loss'):.4f}" for k in layers)
    tr = " ".join(f"L{k}={s.get(f'train/pls/L{k}'):.4f}" for k in layers if s.get(f"train/pls/L{k}") is not None)
    print(f"SUMMARY {cfg} {run_id} rc={rc} mfu={s.get('throughput/mfu')} tok_s={s.get('throughput/tokens_per_second')} "
          f"train/loss={s.get('train/loss')} eval/loss={s.get('eval/loss')} | eval {ev} | train {tr} | err='{err}'", flush=True)
except Exception as e:
    print(f"SUMMARY {cfg} {run_id} rc={rc} err='{err}' wandb_error={e!r}", flush=True)
PYEOF
  rm -rf "$MARIN_PREFIX/speedrun/$run_id"
  find "$MARIN_PREFIX/tmp/checkpoints-temp" -maxdepth 4 -type d -name "*${run_id}*" -prune -exec rm -rf {} + 2>/dev/null
done
echo "SMOKE DONE $(date +%H:%M:%S)"
