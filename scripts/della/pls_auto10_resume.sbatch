#!/bin/bash
#SBATCH --job-name=auto10-resume
#SBATCH --account=pli
#SBATCH --partition=pli
#SBATCH --qos=pli-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (the run's knobs come from the job name, nothing from the submit shell)

# Finishes one Figure 15 run that its pair job stopped (pls_auto10_pair.sbatch: the other half ended first and this one
# had PAIR_GRACE_MIN more minutes alone). Same knobs, so the launcher builds the same run id and resumes from its last
# temporary checkpoint on 4 GPUs. Its smoke copy passed in the pair job. Before submitting, sync the stopped part's
# offline W&B run (if it ran offline), so this part continues that run. --job-name=auto10-resume-<kd|ad|mix|mlpo|conf|first|os|randl>.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls-auto10
source scripts/della/pls_env.sh
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
LAUNCH=experiments.references.della_muonh_qwen3_scaling
unset SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN
unset SIZE LAYERS ARCH PDP PLS_W PLS_DETACH PLS_SEP PLS_DETACH_BB PLS_LOCAL PLS_SHAPE PLS_PCGRAD PLS_OFF PLS_ON PLS_PROBE PLS_MON PLS_EVAL SEED RUN_TAG LR_MUL WARMUP HEADS_FROM PARAM_MAP PLS_SCAN_FIX PLS_OPT_SPLIT HEADS_WARMUP PLS_MIX PLS_KD PLS_CONF PLS_ADAPT PLS_MLPONLY PLS_FIRST PLS_RANDL PLS_SEP_UPTO PROBE_FROM LENS_FROM
module load proxy/default 2>/dev/null || true  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della VARIANT=pls XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PACKED_LEDGER_DIR=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/packed
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
if [ -n "${FETCHED_JOB_ID:-}" ]; then
  NAME=$(sacct -X -n -j "$FETCHED_JOB_ID" -o JobName%80 | head -1 | tr -d ' '); lim=$(sacct -X -n -j "$FETCHED_JOB_ID" -o Timelimit | head -1 | tr -d ' ')
else
  NAME=$SLURM_JOB_NAME; lim=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
fi
SEP="SIZE=130m PLS_W=1 PLS_SEP=1"
case $NAME in
  auto10-resume-kd) S="$SEP PLS_KD=1";;          auto10-resume-ad) S="$SEP PLS_ADAPT=1024";;
  auto10-resume-mix) S="$SEP PLS_MIX=0.2";;      auto10-resume-mlpo) S="$SEP PLS_MLPONLY=1";;
  auto10-resume-conf) S="$SEP PLS_CONF=1";;      auto10-resume-first) S="$SEP PLS_FIRST=256";;
  auto10-resume-os) S="$SEP PLS_OPT_SPLIT=1";;   auto10-resume-randl) S="$SEP PLS_RANDL=1";;
  auto10-resume-upto3) S="$SEP PLS_SEP_UPTO=3";; auto10-resume-upto2) S="$SEP PLS_SEP_UPTO=2";;
  *) echo "unknown resume job name '$NAME'"; exit 2;;
esac
if [ "${WANDB_MODE:-}" != offline ] && ! curl -s -o /dev/null --max-time 20 https://api.wandb.ai/healthz; then
  export WANDB_MODE=offline; echo "W&B not reachable through the proxy: running offline (sync later)"
  echo "$(date +%F_%T) job $JID ($PWD) offline" >> /scratch/gpfs/GROUP/USER/project/marin-pls/logs/wandb_offline.txt
fi
JOB_MIN=$(awk -F'[-:]' '{ if (NF == 4) print $1 * 1440 + $2 * 60 + $3; else if (NF == 3) print $1 * 60 + $2; else print $1 }' <<< "$lim")
[[ $JOB_MIN =~ ^[0-9]+$ ]] && [ "$JOB_MIN" -gt 30 ] || { echo "cannot read the job's time limit from '$lim'"; exit 2; }
rid=$(basename "$(dirname "$(env $S DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu "$PY" -m $LAUNCH 2>/dev/null | sed -n 's/^output: //p')")")
[[ $rid == muonh-qwen3-130m-* ]] || { echo "bad run id '$rid'"; exit 2; }
[ -d "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$rid" ] || { echo "no temporary checkpoint of $rid: nothing to resume"; exit 3; }
mkdir -p "$PACKED_LEDGER_DIR"; ledger=$PACKED_LEDGER_DIR/$SLURM_JOB_ID.tsv
[[ -s $ledger ]] || printf 'task\trun\tkind\tgpus\tstart\tend\texit\n' > "$ledger"
trap 'pls_tmp_cleanup' EXIT
nvidia-smi -L || true
LIMIT=$(( JOB_MIN * 60 - SECONDS - 300 ))s
echo "resume $rid [$S] limit $LIMIT $(date +%H:%M:%S)"
r=0; env $S bash $PT train "${rid#muonh-qwen3-}" main 4 "$LIMIT" -- "$PY" -m $LAUNCH > "logs/slurm/${NAME}_${JID}.log" 2>&1 || r=$?
echo "resume $rid exited rc=$r $(date +%H:%M:%S)"
exit $r
