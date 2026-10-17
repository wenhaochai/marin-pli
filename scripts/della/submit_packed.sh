#!/bin/bash
# Submit a packed job (packed_job.sbatch) that a host job can fetch (owner via the controller, 2026-10-16).
#   scripts/della/submit_packed.sh NAME PLAN TIME [GPUS] [host]     # prints the job id
# Writes logs/plans/NAME.sbatch, a self-contained batch script: the Slurm resources, the #FETCHABLE line, PLAN and
# PACKED_NAME set inside, then packed_job.sbatch. Nothing comes from the submitting shell, so tools/fetch_pending.sh can
# run it in a clean environment inside another job's GPUs (it checks --comment=fetchable and the #FETCHABLE line).
# TIME is the whole job's limit (about 1.25x the expected wall time); the plan's own task limits must fit inside it.
# GPUS: 8 (default, -c 96) or 4 (-c 48); a fetch takes only a job with exactly the host's GPU count.
# host: a long job (TIME 24:00:00) that no host can fetch; after its plan it runs tools/fetch_pending.sh, which runs
# pending fetchable jobs in the time left (CLAUDE.md, 2026-10-15). No #FETCHABLE line and no fetchable comment.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
[ $# -ge 3 ] || { sed -n 2,3p "$0"; exit 2; }
name=$1; plan=$(realpath "$2"); time=$3; gpus=${4:-8}; mode=${5:-fetchable}
[[ $mode == fetchable || $mode == host ]] || { echo "mode must be fetchable or host"; exit 2; }
[[ $name =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad job name '$name'"; exit 2; }
[ -f "$plan" ] || { echo "no plan $plan"; exit 2; }
case $gpus in 8) cpus=96 ;; 4) cpus=48 ;; *) echo "GPUS must be 8 or 4"; exit 2 ;; esac
[[ $time =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] || { echo "TIME as [D-]HH:MM:SS"; exit 2; }
.venv/bin/python scripts/della/check_plans.py "$plan" >/dev/null || { echo "check_plans.py rejects $plan"; exit 2; }
out=logs/plans/$name.sbatch
cat > "$out.tmp" <<EOT
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:$gpus
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem-per-gpu=24G
#SBATCH --time=$time
#SBATCH --exclude=della-k14g3,della-k17g3
$( [ $mode = fetchable ] && echo '#SBATCH --comment=fetchable' )
#SBATCH --output=logs/slurm/%x_%j.out
$( [ $mode = fetchable ] && echo '#FETCHABLE self-contained: written by scripts/della/submit_packed.sh; needs nothing from the submitting shell' )
export PLAN=$plan PACKED_NAME=$name
$( [ $mode = fetchable ] && echo 'exec bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/packed_job.sbatch' || printf '%s\n' 'bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/packed_job.sbatch; rc=$?' 'bash /scratch/gpfs/GROUP/USER/tools/fetch_pending.sh   # the time left runs pending fetchable jobs' 'exit $rc' )
EOT
mv "$out.tmp" "$out"
sbatch --parsable "$out"
