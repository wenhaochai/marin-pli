#!/bin/bash
# The 512-token truncation (owner 2026-10-31, "补一下512"): build, validate and convert (convert_vk_vis.sh), then its exact
# bpb tables (bpb_exact_tables_extra.py key "512"); on della-vis1:
#     ssh della-vis1 'cd .../marin && setsid nohup bash scripts/della/v512_vis.sh > logs/slurm/v512-vis.out 2>&1 < /dev/null &'
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
bash scripts/della/convert_vk_vis.sh 512
echo "== tables $(date +%T)"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 19 .venv/bin/python /scratch/gpfs/GROUP/USER/tmp_plots/vocab/bpb_exact_tables_extra.py
echo "all done $(date +%T)"
