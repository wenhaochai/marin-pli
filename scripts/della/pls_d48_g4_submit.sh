#!/bin/bash
# The 48-layer runs on 4 GPUs each (2026-10-08). Every 48-layer run so far used 8 GPUs, and on 8 GPUs its per-layer
# readouts are wrong in the train step and in eval (train/pls/L47 6.25, eval/L47 7.18) while its backbone is healthy
# (main eval c4_en 4.231; on 1 GPU readout L47 = 4.253). Every 6- and 12-layer run used 4 GPUs and is right.
# Each design's separate-heads and probes runs share one 8-GPU node (pls_h100x8_pair.sbatch, GPUs 0-3 and 4-7),
# as a chain of 24 h jobs (afterany; each resumes from the last checkpoint). RUN_TAG=-g4 keeps the ids apart from
# the 8-GPU runs. Usage: scripts/della/pls_d48_g4_submit.sh [--dry]
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
DRY=${1:-}
# design  PDP  host GiB per GPU  chain length (4 GPUs: about 2x the 8-GPU wall; 8-GPU walls 14 h, attnres 18 h, moda 24 h)
designs=(
  "sandwich 4 24 2"
  "preln 4 24 2"
  "lns 4 24 2"
  "deepnorm 4 24 2"
  "keel 4 24 2"
  "hc 4 24 2"
  "mhc 4 24 2"
  "attnres_block 2 64 2"
  "attnres 2 64 3"
  "moda 2 64 3"
)
for d in "${designs[@]}"; do
  read -r name pdp mem n <<< "$d"
  arch=$name; [ "$name" = sandwich ] && arch=   # the baseline design is the launcher's default (ARCH empty)
  base="SIZE=130m LAYERS=48 PDP=$pdp PLS_W=1 PLS_SEP=1 RUN_TAG=-g4"
  [ -n "$arch" ] && base="$base ARCH=$arch"
  a="$base PLS_DETACH_BB=0"; b="$base PLS_DETACH_BB=1"
  dep=
  for k in $(seq 1 "$n"); do
    # PAIR_A / PAIR_B reach the job through the environment (as for 15168919); the pair script clears every other knob
    cmd=(env PAIR_A="$a" PAIR_B="$b" PAIR_JOB_MIN=1440 sbatch --parsable --job-name="pls-d48g4-$name" --time=24:00:00
         --mem-per-gpu="${mem}G" ${dep:+--dependency=afterany:$dep} scripts/della/pls_h100x8_pair.sbatch)
    if [ "$DRY" = --dry ]; then echo "${cmd[*]}"; dep=DRY$k; else dep=$("${cmd[@]}"); echo "$name step $k: $dep"; fi
  done
done
