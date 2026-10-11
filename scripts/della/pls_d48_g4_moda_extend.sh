#!/bin/bash
# moda after the layout fix (604cdc18c7, 2026-10-10): one forward + backward 2.9x faster, so a run takes ~48 h at
# microbatch 1 (expected ~35 s/step, not measured on H100). The queued chain 15266163 -> 15266164 (submitted with
# pls_d48_g4_submit.sh, PDP=2, which the launcher turns into microbatch 1) reads the new code at start; this adds a
# 4th 24 h job after 15266164 with the same environment, then removes moda's fail marker so the chain runs.
# Usage: scripts/della/pls_d48_g4_moda_extend.sh [--dry]
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
DRY=${1:-}
queued=$(squeue -h -u "$USER" -o '%i %j %T')
grep -qx '15266164 pls-d48g4-moda PENDING' <<< "$queued" || { echo "15266164 (moda chain step 3) is not queued; not extending" >&2; exit 1; }
grep -qx '15266163 pls-d48g4-moda PENDING' <<< "$queued" || { echo "15266163 (moda chain step 2) is not queued; not extending" >&2; exit 1; }
base="SIZE=130m LAYERS=48 PDP=2 PLS_W=1 PLS_SEP=1 PARAM_MAP=vocab RUN_TAG=-g4 ARCH=moda"
cmd=(env PAIR_A="$base PLS_DETACH_BB=0" PAIR_B="$base PLS_DETACH_BB=1" PAIR_JOB_MIN=1440 PAIR_SMOKE_MIN=45 PAIR_FIRST_STEP_MIN=40 sbatch --parsable
     --job-name=pls-d48g4-moda --time=24:00:00 --mem-per-gpu=64G --dependency=afterany:15266164 scripts/della/pls_h100x8_pair.sbatch)
marker=logs/pair_failed/muonh-qwen3-130m-della4xh100-pls1-sep-amoda-vs-d48-g4.fail
if [ "$DRY" = --dry ]; then echo "${cmd[*]}"; echo "rm -f $marker"; exit 0; fi
id=$("${cmd[@]}")
echo "moda step 4: $id"
rm -f "$marker"
