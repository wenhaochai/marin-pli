#!/bin/bash
#SBATCH --job-name=pls-d48-tempsize
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:45:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Temp-buffer sizes (2026-10-10). Under PARAM_MAP=vocab at 48 layers, hc's train step needs 32.91 GiB at microbatch 4
# (OOM at step 257 in the 0.95 pool, 15266153) and mhc's 110.30 GiB at microbatch 2 (OOM at step 1, 15266155). Each
# design and microbatch below runs a 10-step smoke with separate heads (the larger of the pair) in the real 0.95 pool;
# XLA dumps jit__train_step's buffer assignment at compile time, before any OOM, so every line reports the temp
# buffer whether or not the step fits (the largest over the train-step variants). Passing 10 steps is not safety: the
# pool needs 2 x temp + state. Step time = seconds between the last two progress lines over the steps between them
# (both compiles fall before them). Two 4-GPU slots run their lists in order, at most 13 min per entry.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
[[ ${PLS_JOB_TMPDIR:-} == "/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID" ]] || { echo "no per-job TMPDIR"; exit 2; }
module load proxy/default 2>/dev/null || true
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della WANDB_MODE=offline
unset TOTAL_STEPS CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG SEED DRY_RUN ARCH PLS_LOCAL PLS_SHAPE PLS_PCGRAD PLS_OFF PLS_ON PLS_PROBE HEADS_FROM LR_MUL WARMUP INIT_FROM PARAM_MAP_PDP PLS_EVAL
export VARIANT=pls SIZE=130m LAYERS=48 PDP=4 PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=0 PARAM_MAP=vocab PARAM_MAP_MEM_FRACTION=0.95 SMOKE_STEPS=10 PLS_EVAL=0
nvidia-smi -L
slot0=(hc:4 hc:2 mhc:4 mhc:1 attnres_block:4 attnres_block:2)
slot1=(attnres:4 attnres:2 moda:4 moda:2 lns:4)
run_slot() {   # $1 = GPU list, rest = arch:microbatch entries
  local g=$1; shift
  for e in "$@"; do
    local a=${e%%:*} m=${e##*:} tag=-j${SLURM_JOB_ID}t${e//[:_]/}
    local log=logs/slurm/pls-d48-tempsize_${SLURM_JOB_ID}_${a}_${m}.log dump=$PLS_JOB_TMPDIR/xla_${a}_${m}
    local out id rc=0
    out=$(ARCH=$a RUN_TAG=$tag PARAM_MAP_PDP=$m DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's/^output: //p' || true)
    id=$(basename "$(dirname "$out")")
    [[ $id == *${tag}*-smoke10 ]] || { echo "$e: bad run id '$id'"; continue; }
    mkdir -p "$dump"
    XLA_FLAGS="--xla_dump_to=$dump --xla_dump_hlo_module_re=.*train_step.*" ARCH=$a RUN_TAG=$tag PARAM_MAP_PDP=$m \
      CUDA_VISIBLE_DEVICES=$g timeout --signal=TERM --kill-after=30s 13m $PY -m experiments.references.della_muonh_qwen3_scaling > "$log" 2>&1 || rc=$?
    local b temp="no dump"
    b=$(ls "$dump"/*train_step*buffer-assignment.txt 2>/dev/null | tr '\n' ' ' || true)
    if [ -n "$b" ]; then
      # shellcheck disable=SC2086
      temp=$(grep -hE 'preallocated-temp' $b | grep -oE 'size [0-9]+' | awk '$2>m{m=$2} END {if (m) printf "%.2f GiB over %d dumps", m/2^30, '"$(echo $b | wc -w)"'; else print "dump without preallocated-temp"}' || true)
    fi
    local sec
    sec=$(tr '\r' '\n' < "$log" | grep -E 'Progress on:train [0-9.]+it' | sed -E 's/^I[0-9]{8} ([0-9:]{8}) .*train ([0-9]+)(\.[0-9]+)?it.*/\1 \2/' | awk '!s[$2]++' | tail -2 | awk -F'[: ]' '{t[NR]=$1*3600+$2*60+$3; n[NR]=$4} END {if (NR==2 && n[2]>n[1]) printf "%.1f s/step (steps %d-%d)", (t[2]-t[1])/(n[2]-n[1]), n[1], n[2]; else print "none"}' || true)
    echo "$e rc=$rc microbatch=$(grep -m1 -oE 'per-device microbatch [0-9-]+' "$log" | awk '{print $3}') temp=$temp oom=$(grep -m1 -oE 'RESOURCE_EXHAUSTED: Out of memory while trying to allocate [0-9.]+GiB' "$log" | grep -oE '[0-9.]+GiB' || echo none) step=$sec"
    rm -rf -- "$dump"
    [[ $id == *${tag}*-smoke10 ]] && rm -rf -- "$MARIN_PREFIX/speedrun/$id" "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$id"
  done
}
run_slot 0,1,2,3 "${slot0[@]}" &   # each result line goes to the .out as soon as its entry ends
p0=$!
run_slot 4,5,6,7 "${slot1[@]}" &
p1=$!
wait $p0 || true; wait $p1 || true
