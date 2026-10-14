#!/bin/bash
# AttnRes (Full) at 48 layers, a second init seed (SEED=1; data order unchanged, data_seed 42), 2026-10-13. With seed 0
# the ordinary (probes) run's last readout jumped from 3.41 to 8.32 at step 2106: the head's depth softmax collapsed
# (grad/norm/transformer.final_query 0.12 -> 3.8e-07) and the run took ~1000 steps to come back, so its final 3.045 and
# the design's missing last-layer cost (-0.02) may come from that one event. This pair measures the cost again with
# another init: same knobs as pls_d48_g4_submit.sh for attnres plus SEED=1 (run ids ...-vs-d48-s1-g4), chain of 3 x 24 h
# like the seed-0 attnres chain. Usage: scripts/della/pls_d48_g4_attnres_seed1.sh [--dry]
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
DRY=${1:-}
base="SIZE=130m LAYERS=48 PDP=2 PLS_W=1 PLS_SEP=1 PARAM_MAP=vocab RUN_TAG=-g4 ARCH=attnres SEED=1"
a="$base PLS_DETACH_BB=0"; b="$base PLS_DETACH_BB=1"
dep=
for k in 1 2 3; do
  cmd=(env PAIR_A="$a" PAIR_B="$b" PAIR_JOB_MIN=1440 PAIR_SMOKE_MIN=45 PAIR_FIRST_STEP_MIN=40 sbatch --parsable --job-name="pls-d48g4-attnres-s1" --time=24:00:00
       --mem-per-gpu=64G ${dep:+--dependency=afterany:$dep} scripts/della/pls_h100x8_pair.sbatch)
  if [ "$DRY" = --dry ]; then echo "${cmd[*]}"; dep=DRY$k; else dep=$("${cmd[@]}"); echo "attnres seed 1 step $k: $dep"; fi
done
