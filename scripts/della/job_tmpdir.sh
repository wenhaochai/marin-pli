# Sourced by every job script (rule 2026-10-08: compute-node /tmp is small and shared; a full /tmp once failed every
# compile of a scoring gate). Gives this script its own TMPDIR under /scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID
# and deletes it when the script exits. A packed job runs several tasks at once in one Slurm job, so each sourcing script
# gets its own subdirectory (a shared one would vanish when the first task ends); packed_job.sbatch removes the job
# directory at the end, and a task script also removes it when it is the last one out. Compiler caches (Triton, CUDA)
# go there too; apptainer gets the same directory. Outside Slurm (a vis node) it changes nothing.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  JOBTMP=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID
  mkdir -p "$JOBTMP"
  TMPDIR=$(mktemp -d -p "$JOBTMP" "${JOB_TMP_TAG:-task}.XXXXXX")
  export JOBTMP TMPDIR TMP=$TMPDIR TEMP=$TMPDIR
  export TRITON_CACHE_DIR=$TMPDIR/triton CUDA_CACHE_PATH=$TMPDIR/nv
  export APPTAINERENV_TMPDIR=$TMPDIR APPTAINER_BINDPATH=${APPTAINER_BINDPATH:+$APPTAINER_BINDPATH,}$TMPDIR
  # EXIT runs on a normal end and, through the TERM/INT trap, on a stop by timeout, pgroup or Slurm (exit 143 as before)
  case "$TMPDIR" in "$JOBTMP"/*) ;; *) echo "job_tmpdir: mktemp gave '$TMPDIR'" >&2; exit 2 ;; esac   # never rm -rf a bad path
  trap 'rm -rf "$TMPDIR"; rmdir "$JOBTMP" 2>/dev/null || true' EXIT   # || true: safe under set -e
  trap 'exit 143' TERM INT
fi
