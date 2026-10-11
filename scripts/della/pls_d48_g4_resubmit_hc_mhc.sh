#!/bin/bash
# Resubmit hc and mhc of the 48-layer set (pls_d48_g4_submit.sh) at microbatch 1, 2026-10-10. hc died of OOM at step 257
# at microbatch 4 (15266153), and microbatch 2 needs 110 GiB for both (15322826); microbatch 1 needs 17.69 GiB (mhc,
# 25.3 s/step, about 35 h per run: two 24 h jobs). Same run ids as before (PDP is not in the id), so hc resumes from its
# last checkpoint. Host memory 48G per GPU (32G at microbatch 4; attnres_block at microbatch 1 holds 188 GB RSS per process,
# not measured for hc and mhc). Removes the two designs' fail markers first; run only after checking that no hc or mhc job is queued.
# Usage: scripts/della/pls_d48_g4_resubmit_hc_mhc.sh [--dry]
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
DRY=${1:-}
queued=$(squeue -h -u "$USER" -o '%j')
if grep -qxE 'pls-d48g4-(hc|mhc)' <<< "$queued"; then echo "an hc or mhc job is still queued; not submitting" >&2; exit 1; fi
for name in hc mhc; do
  base="SIZE=130m LAYERS=48 PDP=1 PLS_W=1 PLS_SEP=1 PARAM_MAP=vocab RUN_TAG=-g4 ARCH=$name"
  a="$base PLS_DETACH_BB=0"; b="$base PLS_DETACH_BB=1"
  marker=logs/pair_failed/muonh-qwen3-130m-della4xh100-pls1-sep-a$name-vs-d48-g4.fail
  dep=
  if [ "$DRY" = --dry ]; then echo "rm -f $marker"; else rm -f "$marker"; fi
  for k in 1 2; do
    cmd=(env PAIR_A="$a" PAIR_B="$b" PAIR_JOB_MIN=1440 PAIR_SMOKE_MIN=45 PAIR_FIRST_STEP_MIN=40 sbatch --parsable --job-name="pls-d48g4-$name" --time=24:00:00
         --mem-per-gpu=48G ${dep:+--dependency=afterany:$dep} scripts/della/pls_h100x8_pair.sbatch)
    if [ "$DRY" = --dry ]; then echo "${cmd[*]}"; dep=DRY$k; else dep=$("${cmd[@]}"); echo "$name step $k: $dep"; fi
  done
done
