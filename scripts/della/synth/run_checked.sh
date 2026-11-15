#!/bin/bash
# run_checked.sh RUN SCRIPT [VAR=value ...]: train only if the scaling script's dry run, under exactly these VAR=value,
# gives the run id muonh-qwen3-RUN (a knob leaked from the submitting shell would change it); then run SCRIPT with them.
set -uo pipefail
WT=/scratch/gpfs/KARTHIKN/wc9403/project/marin-synth; _pp=$WT; for d in "$WT"/lib/*/src; do _pp=$_pp:$d; done
run=$1 script=$2; shift 2
cd /scratch/gpfs/KARTHIKN/wc9403/project/marin-synth
got=$(env "$@" DRY_RUN=1 PYTHONPATH="$_pp" /scratch/gpfs/KARTHIKN/wc9403/project/marin/.venv/bin/python -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' | head -1)
if [ "$got" != "muonh-qwen3-$run" ]; then echo "RUN ID MISMATCH: plan says muonh-qwen3-$run, dry run gives '$got' [$*]"; exit 3; fi
echo "run id checked: $got"
exec env "$@" bash "$script"
