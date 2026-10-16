#!/bin/bash
#SBATCH --job-name=fetchtest-host
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:01:00
#SBATCH --output=logs/slurm/%x_%j.out
# Emulates tools/fetch_pending.sh running a fetchable script (same clean environment, FETCHED_JOB_ID, SLURM_SUBMIT_DIR,
# its own StdOut) on a generated submit_packed.sh script with a smoke-only plan. Checks: logs, TMPDIR and ledger lines.
cd /scratch/gpfs/GROUP/USER/project/marin
fid=9$SLURM_JOB_ID
keep=(); while IFS= read -r -d '' kv; do
  [[ ${kv%%=*} =~ ^(HOME|USER|LOGNAME|SHELL|PATH|LANG|LC_[A-Z]+|TERM|HOSTNAME|SLURM_[A-Z0-9_]+|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL|ROCR_VISIBLE_DEVICES|MODULEPATH|MODULESHOME|MODULEPATH_ROOT|LMOD[A-Z_]*|__LMOD[A-Z_]*|BASH_FUNC_[a-z_]+%%|_ModuleTable[0-9_]*_|LOADEDMODULES|_LMFILES_)$ ]] && keep+=("$kv")
done < <(env -0)
env -i "${keep[@]}" SLURM_SUBMIT_DIR="$PWD" FETCHED_JOB_ID="$fid" timeout 3300s bash logs/plans/fetchtest-cvd.sbatch >> logs/slurm/fetchtest-cvd_$fid.out 2>&1
echo "fetched rc=$? fid=$fid"
ls -d /scratch/gpfs/GROUP/USER/tmp/job/$fid 2>/dev/null && echo "TMPDIR LEFT" || echo "TMPDIR removed"
ls logs/slurm/*_$fid* ; cat logs/packed/$SLURM_JOB_ID.tsv
