#!/bin/bash
#SBATCH --job-name=muonh-qwen3-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:01:00
#SBATCH --output=logs/slurm/%x_%j.out

# H100 smoke test of the muonh_qwen3_scaling sizes back to back (SIZES env selects a subset): SMOKE_STEPS steps
# at the real batch, one eval batch and a final checkpoint each. A SUMMARY line per size gives the exit code, the
# first OOM/traceback line and the W&B MFU and tokens/s; each smoke output is deleted afterwards.
set -uo pipefail
cd /scratch/gpfs/KARTHIKN/wc9403/project/marin

module load proxy/default  # wandb only
export MARIN_PREFIX=/scratch/gpfs/KARTHIKN/wc9403/marin_store_big
export HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export SMOKE_STEPS=${SMOKE_STEPS:-40}
export VARIANT=${VARIANT:-baseline}
export PRECISION=${PRECISION:-bf16}
SUFFIX=$([ "$VARIANT" = fbt ] && echo "-fbt2")$([ "$PRECISION" = fp8 ] && echo "-fp8")$([ "${TIE:-0}" = 1 ] && echo "-tied")
[ "$VARIANT" = fbt ] && SUFFIX="$SUFFIX$([ "${FBT_NOISE:-0.02}" = 0 ] && echo "-nonoise")$([ "${FBT_RESIDUAL:-0}" = 1 ] && echo "-res")$([ "${FBT_LAYERWISE:-0}" = 1 ] && echo "-lw")$([ "${FBT_INPUT:-1}" = 0 ] && echo "-noin")$([ "${FBT_INPUT_LAYERS:-1}" -gt 1 ] && echo "-ml${FBT_INPUT_LAYERS}")$([ "${FBT_ALPHA_MULT:-1}" != 1 ] && echo "-am${FBT_ALPHA_MULT}")"
case "$VARIANT" in  # must mirror variant_tags in della_muonh_qwen3_scaling.py (weights in shortest form: 0.1, not 0.10)
  twin)   SUFFIX="$SUFFIX-twin${TWIN_W:-0.1}$([ "${TWIN_OFF:-2}" != 2 ] && echo "o${TWIN_OFF}")" ;;
  sr)     SUFFIX="$SUFFIX-sr${SR_GAMMA:-0.9}w${SR_W:-0.1}" ;;
  twinsr) SUFFIX="$SUFFIX-twin${TWIN_W:-0.1}$([ "${TWIN_OFF:-2}" != 2 ] && echo "o${TWIN_OFF}")-sr${SR_GAMMA:-0.9}w${SR_W:-0.1}" ;;
  pi)     SUFFIX="$SUFFIX-pik${PI_K:-4}w${PI_W:-0.1}$([ "${PI_TAU:-0.1}" != 0.1 ] && echo "t${PI_TAU}")$([ "${PI_NEG:-512}" != 512 ] && echo "n${PI_NEG}")" ;;
  eos)    SUFFIX="$SUFFIX-eosw${EOS_W:-0.1}" ;;
esac
case "$VARIANT" in twin|sr|twinsr|pi|eos) [ "${FREE_HEADS:-1}" = 1 ] && SUFFIX="$SUFFIX-fh" ;;
esac
[ -n "${INIT_FROM:-}" ] && SUFFIX="$SUFFIX${CPT_TAG:--cpt}"
nvidia-smi -L

for SIZE in ${SIZES:-130m 300m 520m 1_2b}; do
  export SIZE
  log=logs/slurm/muonh-qwen3-smoke_${SLURM_JOB_ID}_${SIZE}${SUFFIX}.log
  echo "SMOKE $SIZE START $(date +%H:%M:%S)"
  timeout --signal=TERM 14m .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
  rc=$?
  err=$(grep -m1 -E "RESOURCE_EXHAUSTED|out of memory|Traceback" $log | cut -c1-160)
  .venv/bin/python - "$SIZE" "$rc" "$err" "$SUFFIX" <<'EOF'
import os, sys, wandb
size, rc, err, suffix = sys.argv[1:5]
run_id = f"muonh-qwen3-{size}-della4xh100{suffix}-smoke{os.environ['SMOKE_STEPS']}"
try:
    s = wandb.Api(timeout=60).run(f"marin-della/{run_id}").summary
    print(f"SUMMARY size={size}{suffix} rc={rc} mfu={s.get('throughput/mfu')} tok_s={s.get('throughput/tokens_per_second')} err='{err}'", flush=True)
except Exception as e:
    print(f"SUMMARY size={size}{suffix} rc={rc} wandb_error={str(e)[:100]} err='{err}'", flush=True)
EOF
  rm -rf "$MARIN_PREFIX/speedrun/muonh-qwen3-${SIZE}-della4xh100${SUFFIX}-smoke${SMOKE_STEPS}"
done
echo "SMOKE DONE $(date +%H:%M:%S)"
