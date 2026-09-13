#!/bin/bash
# Two-phase data prep on a Della login node: networked raw downloads, then offline tokenize.
set -uo pipefail
cd /scratch/gpfs/KARTHIKN/wc9403/project/marin-july
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/KARTHIKN/wc9403/marin_store} HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface DIM=${DIM:-512}
echo "PHASE raw $(date)"
nice -n 19 .venv/bin/python -m experiments.grug.moe.della_july_baseline --run_only '["raw/"]' --max_concurrent 2 || { echo "PHASE raw FAILED $(date)"; exit 1; }
echo "PHASE tokenize $(date)"
ZEPHYR_SUBPROCESS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nice -n 19 .venv/bin/python -m experiments.grug.moe.della_july_baseline --run_only '["tokenized/"]' --max_concurrent 3 || { echo "PHASE tokenize FAILED $(date)"; exit 1; }
echo "PREP DONE $(date)"
