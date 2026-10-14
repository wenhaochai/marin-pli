#!/bin/bash
# HC with positive writes at 48 layers (ARCH=hc_pos), 2026-10-14: the causal test of the hidden-write account of why HC
# pays about half the separate-heads cost at the last layer (docs/della/pls.md). Everything as the hc pair
# (pls_d48_g4_resubmit_hc_mhc.sh: microbatch 1, 48G host memory per GPU, two 24 h chain jobs; hc needed 24 h + 12.6 h)
# except that every write is 2 sigmoid(.) > 0, so no write can be zero-sum (invisible to the readouts, which read the sum
# of the streams). Prediction: the layer-48 train gap (separate heads minus probes) rises from hc's 0.048 to about mHC's
# 0.10. Runs from the pls-hcpos worktree (its pls_h100x8_pair.sbatch cds there). Usage: ... [--dry]
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls-hcpos
DRY=${1:-}
if squeue -h -u "$USER" -o '%j' | grep -qx 'pls-d48g4-hc_pos'; then echo "an hc_pos job is already queued; not submitting" >&2; exit 1; fi
base="SIZE=130m LAYERS=48 PDP=1 PLS_W=1 PLS_SEP=1 PARAM_MAP=vocab RUN_TAG=-g4 ARCH=hc_pos"
a="$base PLS_DETACH_BB=0"; b="$base PLS_DETACH_BB=1"
dep=
for k in 1 2; do
  cmd=(env PAIR_A="$a" PAIR_B="$b" PAIR_JOB_MIN=1440 PAIR_SMOKE_MIN=45 PAIR_FIRST_STEP_MIN=40 sbatch --parsable --job-name="pls-d48g4-hc_pos" --time=24:00:00
       --mem-per-gpu=48G ${dep:+--dependency=afterany:$dep} scripts/della/pls_h100x8_pair.sbatch)
  if [ "$DRY" = --dry ]; then echo "${cmd[*]}"; dep=DRY$k; else dep=$("${cmd[@]}"); echo "hc_pos step $k: $dep"; fi
done
