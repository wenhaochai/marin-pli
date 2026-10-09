#!/bin/bash
# The 48-layer runs on 4 GPUs each (2026-10-08). Every 48-layer run so far used 8 GPUs, and on 8 GPUs its per-layer
# readouts are wrong in the train step and in eval (train/pls/L47 6.25, eval/L47 7.18) while its backbone is healthy
# (main eval c4_en 4.231; on 1 GPU readout L47 = 4.253). Every 6- and 12-layer run used 4 GPUs and is right.
# Each design's separate-heads and probes runs share one 8-GPU node (pls_h100x8_pair.sbatch, GPUs 0-3 and 4-7),
# as a chain of 24 h jobs (afterany; each resumes from the last checkpoint). RUN_TAG=-g4 keeps the ids apart from
# the 8-GPU runs. Usage: scripts/della/pls_d48_g4_submit.sh [--dry]
# 2026-10-09: the cause is the FSDP parameter layout, not the GPU count. Under embed-over-data sharding a GPU program
# with the 48-layer readouts gives values that depend on what else it holds and change between runs (15255032,
# 15264999: L47 4.2378 / 4.2737 against main 4.2529; the 8-GPU trainer 7.29 against 5.45, 15257881). With the layers
# whole on every GPU (PARAM_MAP=vocab: only vocab-sized tensors sharded) every program agrees (15264999). Every run
# here uses PARAM_MAP=vocab (tag -vs), and each smoke copy must show last readout = main eval within 0.005 for every
# design (pls_h100x8_pair.sbatch; a failure stops the rest of that design's chain via logs/pair_failed/).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
DRY=${1:-}
# design  PDP  host GiB per GPU (two processes per node; the 8-GPU d48 runs peaked at 75-80 GB RSS each)  chain length (4 GPUs: about 2x the 8-GPU wall; 8-GPU walls 14 h, attnres 18 h, moda 24 h)
designs=(
  "sandwich 4 32 2"
  "preln 4 32 2"
  "lns 4 32 2"
  "deepnorm 4 32 2"
  "keel 4 32 2"
  "hc 4 32 2"
  "mhc 4 32 2"
  "attnres_block 2 64 2"
  "attnres 2 64 3"
  "moda 2 64 3"
)
for d in "${designs[@]}"; do
  read -r name pdp mem n <<< "$d"
  arch=$name; [ "$name" = sandwich ] && arch=   # the baseline design is the launcher's default (ARCH empty)
  base="SIZE=130m LAYERS=48 PDP=$pdp PLS_W=1 PLS_SEP=1 PARAM_MAP=vocab RUN_TAG=-g4"
  [ -n "$arch" ] && base="$base ARCH=$arch"
  a="$base PLS_DETACH_BB=0"; b="$base PLS_DETACH_BB=1"
  dep=
  for k in $(seq 1 "$n"); do
    # PAIR_A / PAIR_B reach the job through the environment (as for 15168919); the pair script clears every other knob
    cmd=(env PAIR_A="$a" PAIR_B="$b" PAIR_JOB_MIN=1440 PAIR_SMOKE_MIN=45 PAIR_FIRST_STEP_MIN=40 sbatch --parsable --job-name="pls-d48g4-$name" --time=24:00:00
         --mem-per-gpu="${mem}G" ${dep:+--dependency=afterany:$dep} scripts/della/pls_h100x8_pair.sbatch)
    if [ "$DRY" = --dry ]; then echo "${cmd[*]}"; dep=DRY$k; else dep=$("${cmd[@]}"); echo "$name step $k: $dep"; fi
  done
done
