# Sourced by trec-runs*.sbatch: the environment of experiments/references/trec_score.py and score_runs, which scores
# each "run step subpath" entry (RUN mode: model, batch, steps and tokenizer from the run's cached W&B config).
# Owner popups 2026-10-27: the other MuonH checkpoints in groups that change one setting (depth, size, learning rate, EMA,
# training length, batch, tokenizer, seed), 1 in 8 training batches; the runs that read their data more than once
# (520m, 1.2b, 130m -tpp200) go to their own figure.
WT=/scratch/gpfs/GROUP/USER/project/marin-trec
cd $WT
source scripts/della/job_tmpdir.sh
_pp=$WT; for d in "$WT"/lib/*/src; do _pp=$_pp:$d; done; export PYTHONPATH=$_pp   # this worktree's sources, not project/marin's
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9}
unset VOCAB_K VARIANT SEQ_LEN DATA_EPOCHS PASS_DATA MODEL TREC_BATCH
S=$MARIN_PREFIX/speedrun
nvidia-smi -L
fail=0

# score_runs STEPS OUT_ROOT: reads "run step subpath" lines on stdin; the output dir is OUT_ROOT/<run>, with -ema for the
# EMA weights. Extra trec_score settings (SAMPLE, BLOCK) come from the caller's environment.
score_runs() {
  local steps=$1 root=$2 run step sub c out
  while read -r run step sub; do
    [ -n "$run" ] || continue
    c=$(ls -d $S/$run/*/checkpoints/step-$step 2>/dev/null | head -1); [ -n "$c" ] || { echo "NO CHECKPOINT $run"; fail=1; continue; }
    out=$root/$run; [ "$sub" = model ] || out=$out-ema
    echo "TREC $run $sub $(date +%H:%M:%S)"
    CKPT=$c RUN=$run SUBPATH=$sub STEPS=$steps OUT=$out /scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python -m experiments.references.trec_score < /dev/null || fail=1
  done
}
