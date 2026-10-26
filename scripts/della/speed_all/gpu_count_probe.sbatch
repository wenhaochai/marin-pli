#!/bin/bash
#SBATCH --job-name=gpucount-probe
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# Which GPU count per run is most GPU-hour efficient for 130m and 300m (owner via marin.cpp, 2026-10-25; 4 vs 8 is measured:
# 4 wins). Branch speed-all code with REMAT=flash and CUDA-graph while loops. For SIZE in 130m 300m and G in 1 2 4: a
# SMOKE_STEPS=50 run on the first G of its GPUs with NUM_GPUS=G and 32 sequences per device per microbatch (PDP=32, the 4-GPU split),
# so a smaller G accumulates over more microbatches at the same per-device memory. Runs in turn, never side by side.
# Per run a PROBE line: median step time over steps >= 5 and tokens per second per GPU (128 x 4096 tokens per step).
set -uo pipefail
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/all
PY=$MAIN/.venv/bin/python
cd "$WT"
source scripts/della/job_tmpdir.sh
module load proxy/default  # wandb only
source scripts/della/wandb_guard.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH=$WT/lib/levanter/src:$WT/lib/haliax/src:$WT
unset VARIANT DATA_EPOCHS PASS_DATA TOTAL_STEPS INIT_FROM SEED BATCH VOCAB_K SEQ_LEN TIE PRECISION ATTN_BACKEND
export REMAT=flash XLA_FLAGS="--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL"
export SMOKE_STEPS=50 PDP=32
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # a host hands a guest its own subset: take G of those, never 0..G-1
[ ${#GPUS[@]} -ge 4 ] || { echo "need 4 GPUs, got ${CUDA_VISIBLE_DEVICES:-}"; exit 2; }
nvidia-smi -L
fail=0
for SIZE in 130m 300m; do
  for G in 4 2 1; do
    export SIZE NUM_GPUS=$G CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]:0:$G}") RUN_TAG=-gpuprobe-j$JID
    run_id=$(DRY_RUN=1 $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' | head -1)
    case "$run_id" in *-gpuprobe-j$JID*) ;; *) echo "PROBE $SIZE G=$G: dry run gave '$run_id'"; fail=1; continue ;; esac
    log=$MAIN/logs/slurm/gpucount-probe_${JID}_${run_id}.log
    echo "RUN $run_id START $(date +%H:%M:%S)"
    timeout --signal=TERM 20m $PY -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
    rc=$?; [ $rc -ne 0 ] && fail=1
    $PY - "$run_id" "$rc" "$G" <<'PY'
import sys
import numpy as np
import wandb
run_id, rc, g = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    d = [r["throughput/duration"] for r in wandb.Api(timeout=60).run(f"marin-della/{run_id}").scan_history(keys=["_step", "throughput/duration"]) if r["_step"] >= 5]
    m = float(np.median(d))
    print(f"PROBE {run_id} rc={rc} gpus={g} median_step_s={m:.3f} n={len(d)} tok_per_s_per_gpu={128 * 4096 / m / g:.0f}", flush=True)
except Exception as e:
    print(f"PROBE {run_id} rc={rc} gpus={g} wandb_error={str(e)[:160]}", flush=True)
PY
    case "$run_id" in *-gpuprobe-j$JID*) rm -rf "$MARIN_PREFIX/speedrun/$run_id" ;; esac
  done
done
echo "PROBE DONE $(date +%H:%M:%S) fail=$fail"
exit $fail
