#!/bin/bash
# run_checked.sh RUN SCRIPT [VAR=value ...]: train only if the scaling script's dry run, under exactly these VAR=value,
# gives the run id muonh-qwen3-RUN (a knob leaked from the submitting shell would change it); then run SCRIPT with them.
set -uo pipefail
run=$1 script=$2; shift 2
cd /scratch/gpfs/GROUP/USER/project/marin
got=$(env "$@" DRY_RUN=1 .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' | head -1)
if [ "$got" != "muonh-qwen3-$run" ]; then echo "RUN ID MISMATCH: plan says muonh-qwen3-$run, dry run gives '$got' [$*]"; exit 3; fi
echo "run id checked: $got"
exec env "$@" bash "$script"
