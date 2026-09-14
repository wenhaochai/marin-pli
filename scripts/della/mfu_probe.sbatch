#!/bin/bash
#SBATCH --job-name=mfu-probe
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# MFU probes for the July rungs on 4xH100, run back to back in one <=2h job (one queue slot).
# Each case is a short run of one exact-math execution lever, ending in one eval pass to check eval memory; a SUMMARY line per case gives
# the W&B p50 MFU and tokens/s. CASES env (space-separated indices) selects a subset.
set -uo pipefail
cd /scratch/gpfs/KARTHIKN/wc9403/project/marin-july

# index: DIM PROBE_STEPS REMAT REPLICATE LHS CASE_TIMEOUT_MIN
ALL_CASES=(
  "512 300 recompute_all 0 0 16"
  "512 300 none 0 0 16"
  "512 300 recompute_all 1 0 16"
  "512 300 none 1 0 16"
  "512 300 none 1 1 16"
  "512 300 save_moe 1 0 16"
  "768 150 recompute_all 0 0 18"
  "768 150 none 1 0 18"
  "1024 60 recompute_all 0 0 20"
  "1024 60 save_moe 0 0 20"
  "1024 60 none 0 0 20"
  "1024 60 recompute_moe 0 0 25"
  "1024 60 chunk_moe 0 0 25"
)
SELECT=${CASES:-$(seq -s ' ' 0 $((${#ALL_CASES[@]} - 1)))}

module load proxy/default  # wandb only
export HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export ATTN=gpu_fa4_cute
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.9}
BASE_XLA_FLAGS="${XLA_FLAGS:-}"
nvidia-smi -L

for i in $SELECT; do
  read -r DIM PROBE_STEPS REMAT REPLICATE LHS TMIN <<< "${ALL_CASES[$i]}"
  export DIM PROBE_STEPS REMAT REPLICATE
  export XLA_FLAGS="$BASE_XLA_FLAGS"
  [ "$LHS" = "1" ] && export XLA_FLAGS="$BASE_XLA_FLAGS --xla_gpu_enable_latency_hiding_scheduler=true"
  if [ "$DIM" = "512" ]; then export MARIN_PREFIX=/scratch/gpfs/KARTHIKN/wc9403/marin_store
  else export MARIN_PREFIX=/scratch/gpfs/KARTHIKN/wc9403/marin_store_big; fi
  echo "CASE $i START $(date +%H:%M:%S): DIM=$DIM STEPS=$PROBE_STEPS REMAT=$REMAT REPLICATE=$REPLICATE LHS=$LHS"
  timeout --signal=TERM $((TMIN * 60)) .venv/bin/python -m experiments.grug.moe.della_july_baseline > logs/slurm/mfu-probe_${SLURM_JOB_ID}_case$i.log 2>&1
  rc=$?
  err=$(grep -m1 -E "RESOURCE_EXHAUSTED|ran out of memory|TypeError|Traceback" logs/slurm/mfu-probe_${SLURM_JOB_ID}_case$i.log | cut -c1-160)
  .venv/bin/python - "$i" "$rc" "$err" <<'EOF'
import os, sys, wandb
i, rc, err = sys.argv[1], sys.argv[2], sys.argv[3]
import experiments.grug.moe.della_july_baseline as L
from experiments.grug.moe.moe_may_july_baseline import _POINTS
d = int(os.environ["DIM"]); bs, n = next((b, s) for (dd, b, s) in _POINTS if dd == d)
run_id = L._della_step(d, bs, n).config.run_id
try:
    r = wandb.Api(timeout=60).run(f"marin-della/{run_id}")
    s = r.summary
    label = f"{d}-e{os.environ['EXPERTS']}" if os.environ.get("EXPERTS", "256") != "256" else d
    print(f"SUMMARY case={i} rc={rc} dim={label} remat={os.environ['REMAT']} rep={os.environ['REPLICATE']} "
          f"xla='{os.environ.get('XLA_FLAGS', '').strip()}' steps={r.lastHistoryStep} "
          f"p50_mfu={s.get('throughput/p50_mfu')} tok_s={s.get('throughput/tokens_per_second')} err='{err}'", flush=True)
except Exception as e:
    print(f"SUMMARY case={i} rc={rc} dim={d} experts={os.environ.get('EXPERTS', '256')} wandb_error={str(e)[:120]} err='{err}'", flush=True)
EOF
  rm -rf "$MARIN_PREFIX/mfu_probe"
done
echo "PROBES DONE $(date +%H:%M:%S)"

# Pick the fastest successful case per rung and publish it for the real runs (launcher reads this at start).
grep "^SUMMARY" logs/slurm/mfu-probe_${SLURM_JOB_ID}.out > logs/slurm/mfu-probe_${SLURM_JOB_ID}_summary.txt || true
python3 - logs/slurm/mfu-probe_${SLURM_JOB_ID}_summary.txt logs/lever_defaults.json <<'EOF'
import json, re, sys
src, dst = sys.argv[1], sys.argv[2]
best = {}
for line in open(src):
    kv = dict(re.findall(r"(\w+)=('[^']*'|\S+)", line))
    if kv.get("rc") != "0" or kv.get("p50_mfu") in (None, "None"):
        continue
    dim, mfu = kv["dim"], float(kv["p50_mfu"])
    lhs = "1" if "latency_hiding_scheduler=true" in kv.get("xla", "") else "0"
    if dim not in best or mfu > best[dim]["p50_mfu"]:
        best[dim] = {"REMAT": kv["remat"], "REPLICATE": kv["rep"], "LHS": lhs, "p50_mfu": mfu, "tok_s": float(kv["tok_s"])}
try:
    merged = json.load(open(dst))
except FileNotFoundError:
    merged = {}
merged.update(best)
json.dump(merged, open(dst, "w"), indent=1)
print("LEVER DEFAULTS WRITTEN:", json.dumps(merged))
EOF
