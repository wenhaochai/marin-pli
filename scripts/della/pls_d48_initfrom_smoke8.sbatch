#!/bin/bash
#SBATCH --job-name=pls-smoke-d48-initfrom8
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Diagnostic (2026-10-08). Standalone, the d48 probes checkpoint step-4958 gives readout L47 = main eval (4.20 on 8 GPUs,
# 4.25 on 4; 15235521), but the run that wrote it logged eval/L47 c4_en 6.48 (main 4.231) and train/pls/L47 6.25.
# Here the real launcher and trainer start from that checkpoint (INIT_FROM) for SMOKE_STEPS steps at warmup learning
# rates, then run the trainer's own eval (main + per-layer, one batch per set). If eval/L47 differs from eval/loss here,
# the fault is in the trainer path; if they agree, it is in the state the original run carried.
# 8-GPU copy (2026-10-09): on 4 GPUs (15252596) the trainer's eval gave c4_en main 4.2529 and L47 4.238, close; the
# original run (eval/L47 c4_en 6.479 against main 4.231) trained and evaluated on 8 GPUs. After the trainer smoke, the
# standalone script on 8 GPUs with the trainer's parameter sharding (PATHS_SHARD=1).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
module load proxy/default 2>/dev/null || true  # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
unset TOTAL_STEPS CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG SEED DRY_RUN ARCH PLS_LOCAL PLS_SHAPE PLS_PCGRAD PLS_OFF PLS_ON PLS_PROBE HEADS_FROM LR_MUL WARMUP
export VARIANT=pls SIZE=130m LAYERS=48 PDP=4 PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 SMOKE_STEPS=${SMOKE_STEPS:-10} RUN_TAG=-j${SLURM_JOB_ID}
export INIT_FROM=$MARIN_PREFIX/speedrun/muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen-d48/2026.09.13/checkpoints/step-4958
nvidia-smi -L
out=$(DRY_RUN=1 $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's/^output: //p' || true)
run_id=$(basename "$(dirname "$out")")
[[ $run_id == *-j${SLURM_JOB_ID}*-smoke${SMOKE_STEPS} ]] || { echo "bad run id '$run_id'"; exit 2; }
echo "run $run_id"
rc=0; timeout --signal=TERM --kill-after=60s 30m $PY -m experiments.references.della_muonh_qwen3_scaling > logs/slurm/pls-smoke-d48-initfrom8_${SLURM_JOB_ID}.log 2>&1 || rc=$?
echo "exit $rc"
grep -E "First train step|Loaded|initializ|INIT|from checkpoint" logs/slurm/pls-smoke-d48-initfrom8_${SLURM_JOB_ID}.log | grep -v "Unknown leaf" | head -8 | cut -c1-250
tr '\r' '\n' < logs/slurm/pls-smoke-d48-initfrom8_${SLURM_JOB_ID}.log | grep -E "postfix:loss" | tail -2 | cut -c1-250
# the trainer's eval results, from its metrics file
python3 - "$MARIN_PREFIX/speedrun/$run_id" <<'EOF'
import json, sys, glob
for p in glob.glob(sys.argv[1] + "/**/eval_metrics.jsonl", recursive=True):
    for l in open(p):
        d = json.loads(l)
        print("step", d.get("step"), {k: round(v, 4) for k, v in d.items() if isinstance(v, float) and ("c4_en" in k or k in ("eval/loss",)) and k.endswith("loss") and ("/L" not in k or any(f"/L{n}/" in k for n in (0, 23, 46, 47)))})
EOF
grep -E "train/pls/L47|pls/L47" logs/slurm/pls-smoke-d48-initfrom8_${SLURM_JOB_ID}.log | tail -2 | cut -c1-250 || true
# the smoke's own output only (its exact tagged run id)
[[ $run_id == *-j${SLURM_JOB_ID}*-smoke${SMOKE_STEPS} ]] && rm -rf -- "$MARIN_PREFIX/speedrun/$run_id"
echo "== standalone, 8 GPUs, trainer sharding, T=4096, B=32"
PATHS_GPU=1 PATHS_SHARD=1 timeout 15m $PY scripts/della/pls_d48_eval_path.py 4096 32 2>&1 | grep -E "param mapping|loaded|grug eval|with_seg|Readout|readout loss" | cut -c1-330 || true
exit $rc
