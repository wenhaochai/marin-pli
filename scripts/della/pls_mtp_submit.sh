#!/bin/bash
# Multi-token prediction and every layer (owner, 2026-10-13: DeepSeek-V3 form). One 8-GPU pair job, 130m (6 layers):
#   a  NTP: probes (PLS_SEP=1 PLS_DETACH_BB=1, own optimizer groups PLS_OPT_SPLIT=1) for x_{t+1} and future probes for x_{t+2}
#   b  the same plus one DeepSeek-V3 MTP module on the last layer, weight 0.3 (PLS_MTP_W=0.3)
# Every probe is detached, so a's backbone trains as the baseline and b's as the baseline plus MTP (pls_mtp_cpu_test.py).
# Same seed and data order. Time: the 130m baseline takes ~1:15 and each readout ~16 min at 130m (probes run 14888104:
# 2:35 with 5 extra readouts); here 12 readouts beyond the baseline's head -> ~4.5 h, plus the smoke copies.
# Runs from the pls-mtp worktree (its pair script cds there). Usage: scripts/della/pls_mtp_submit.sh [--dry]
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls-mtp
DRY=${1:-}
base="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PLS_OPT_SPLIT=1 PLS_FUT=1"
cmd=(env PAIR_A="$base PLS_MTP_W=0" PAIR_B="$base PLS_MTP_W=0.3" PAIR_JOB_MIN=390 sbatch --parsable --job-name=pls-mtp --time=06:30:00
     --mem-per-gpu=32G scripts/della/pls_h100x8_pair.sbatch)
if [ "$DRY" = --dry ]; then echo "${cmd[*]}"; else jid=$("${cmd[@]}"); echo "pls-mtp: $jid"; fi
