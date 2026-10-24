#!/bin/bash
#SBATCH --job-name=fa4-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out

# FA4 attention smoke + timing on 4xH100 (branch speed-fa4-segmented-attention-backend; this worktree's code for every
# arm). 300m v64k, seed 0, SMOKE_STEPS (default 60) steps at the real batch plus a one-batch eval, for each
# SEQ_LEN in SEQS (default 4096 4224) and ATTN_BACKEND in ARMS (default jax_flash fa4). Per run a SUMMARY line: exit code,
# peak nvidia-smi memory, mean W&B throughput/duration over steps >= 3, final train/eval loss. Per SEQ_LEN a PAIR line:
# the per-step train/loss difference fa4 - jax_flash (same seed, data, schedule: should sit at bf16 noise, ~1e-3).
set -uo pipefail
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/fa4-segmented-attention-backend
PY=/scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python
cd "$WT"
source scripts/della/job_tmpdir.sh
module load proxy/default  # wandb only
source scripts/della/wandb_guard.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9}
export PYTHONPATH=$WT/lib/levanter/src:$WT/lib/haliax/src:$WT
unset VARIANT DATA_EPOCHS PASS_DATA TOTAL_STEPS INIT_FROM SEED BATCH PDP NUM_GPUS TIE PRECISION
export SIZE=300m VOCAB_K=64000 SMOKE_STEPS=${SMOKE_STEPS:-60} RUN_TAG=-abfa4smoke-j${SLURM_JOB_ID}
LOGDIR=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm
nvidia-smi -L
$PY -c 'import levanter, haliax, jax; print("levanter", levanter.__file__, "haliax", haliax.__file__, "jax", jax.__version__, jax.devices())'
fail=0
for SEQ in ${SEQS:-4096 4224}; do
  ids=()
  for ARM in ${ARMS:-jax_flash fa4}; do
    export SEQ_LEN=$SEQ ATTN_BACKEND=$ARM
    run_id=$(DRY_RUN=1 $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' | head -1)
    case "$run_id" in *-j${SLURM_JOB_ID}*) ;; *) echo "SMOKE $SEQ $ARM: dry run gave '$run_id'"; fail=1; continue ;; esac
    ids+=("$run_id")
    log=$LOGDIR/fa4-smoke_${SLURM_JOB_ID}_${run_id}.log
    echo "SMOKE $run_id START $(date +%H:%M:%S)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > $log.mem &
    mpid=$!
    timeout --signal=TERM ${RUN_TIMEOUT:-6m} $PY -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
    rc=$?
    kill $mpid
    peak=$(sort -n $log.mem | tail -1)
    err=$(grep -a -m1 -E "RESOURCE_EXHAUSTED|out of memory|Traceback|NotImplementedError|CUDA_ERROR" $log | cut -c1-200)
    { [ "$rc" -ne 0 ] || [ -n "$err" ]; } && fail=1
    rate=$(tr '\r' '\n' < $log | grep -a "Progress on:train" | tail -1 | sed -n 's/.*rate:\([^ ]*\).*/\1/p')
    echo "SMOKE $run_id END rc=$rc tqdm_rate=$rate $(date +%H:%M:%S)"
    $PY - "$run_id" "$rc" "$err" "$peak" <<'PY'
import sys
import numpy as np
import wandb
run_id, rc, err, peak = sys.argv[1:5]
try:
    run = wandb.Api(timeout=60).run(f"marin-della/{run_id}")
    s = run.summary
    d = [r["throughput/duration"] for r in run.scan_history(keys=["_step", "throughput/duration"]) if r["_step"] >= 3]
    print(f"SUMMARY {run_id} rc={rc} peak_mem_MiB={peak} step_ms={np.mean(d) * 1e3:.0f}(n={len(d)}) median_step_ms={np.median(d) * 1e3:.0f} "
          f"loss={s.get('train/loss')} eval_loss={s.get('eval/loss')} err='{err}'", flush=True)
except Exception as e:
    print(f"SUMMARY {run_id} rc={rc} peak_mem_MiB={peak} wandb_error={str(e)[:160]} err='{err}'", flush=True)
PY
    case "$run_id" in *-abfa4smoke-j${SLURM_JOB_ID}*) rm -rf "$MARIN_PREFIX/speedrun/$run_id" ;; esac
  done
  if [ ${#ids[@]} -eq 2 ]; then
    $PY - "${ids[0]}" "${ids[1]}" <<'PY'
import sys
import numpy as np
import wandb
a, b = sys.argv[1:3]
try:
    api = wandb.Api(timeout=60)
    h = {r: {x["_step"]: x["train/loss"] for x in api.run(f"marin-della/{r}").scan_history(keys=["_step", "train/loss"])} for r in (a, b)}
    steps = sorted(set(h[a]) & set(h[b]))
    d = np.array([h[b][t] - h[a][t] for t in steps])
    print(f"PAIR {b} - {a}: steps={len(steps)} max_abs_loss_diff={np.max(np.abs(d)):.5f} mean_diff={d.mean():+.5f} "
          f"last10_mean_diff={d[-10:].mean():+.5f} first5={np.round(d[:5], 5).tolist()}", flush=True)
except Exception as e:
    print(f"PAIR {a} {b}: wandb_error={str(e)[:160]}", flush=True)
PY
  fi
done
echo "SMOKE DONE $(date +%H:%M:%S) fail=$fail"
exit $fail
