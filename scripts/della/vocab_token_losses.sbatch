#!/bin/bash
#SBATCH --job-name=vo-toklosses
#SBATCH --account=pli
#SBATCH --partition=pli
#SBATCH --qos=pli-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:01:00
#SBATCH --output=logs/slurm/%x_%j.out

# Per-position Paloma losses (experiments.references.vocab_token_losses) for each checkpoint in SPECS, one after
# another on one GPU. SPECS: ';'-separated entries "<run id> <step> [VAR=value ...]"; the checkpoint is
# speedrun/<run id>/*/checkpoints/step-<step> and the output logs/vocab_token_losses/<run id>.npz.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh   # TMPDIR under scratch, per task, removed at exit (never the shared /tmp)
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9}
unset VOCAB_K VARIANT CV_SIDE CV_K CV_OUT_REDUCE SEQ_LEN SIZE   # each SPECS entry sets its own (--export=ALL)
fail=0
IFS=';' read -ra ENTRIES <<< "${SPECS:?set SPECS}"
for e in "${ENTRIES[@]}"; do
  read -r run step extra <<< "$e"
  ckpt=$(ls -d $MARIN_PREFIX/speedrun/$run/*/checkpoints/step-$step 2>/dev/null | head -1)
  [ -n "$ckpt" ] || { echo "NO CHECKPOINT $run step-$step"; fail=1; continue; }
  echo "RUN $run $ckpt [$extra] $(date +%H:%M:%S)"
  env $extra CKPT=$ckpt OUT=logs/vocab_token_losses/$run.npz .venv/bin/python -m experiments.references.vocab_token_losses || fail=1
done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit $fail
