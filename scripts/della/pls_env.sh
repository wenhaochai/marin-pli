# Source me from the pls worktree. Python is the worktree's own copy of the shared venv (project/marin/.venv, copied
# 2026-09-30), so a `uv sync` in the other checkout cannot change jax/CUDA under a running resume chain. Its editable
# installs still point at project/marin, so this worktree's sources go first on PYTHONPATH: levanter/haliax/marin/... and
# experiments are imported from here.
WT=/scratch/gpfs/GROUP/USER/project/marin-pls
PY=$WT/.venv/bin/python
_pp=$WT
for d in "$WT"/lib/*/src; do _pp=$_pp:$d; done
export PYTHONPATH=$_pp${PYTHONPATH:+:$PYTHONPATH}
unset _pp d

# Inside a Slurm job: a per-job temp dir on scratch instead of the node's /tmp, which is small and shared by every job on
# the node (rule of 2026-10-08). Python's tempfile, XLA, ptxas and the compilers read TMPDIR. Removed when the job's
# shell exits; an EXIT trap the job script set before sourcing this file still runs first. A script that sets its own
# EXIT trap after sourcing calls pls_tmp_cleanup in it.
pls_tmp_cleanup() {  # only this job's own dir; a no-op outside a job. Never fails: under set -e a failing cleanup in the
  # EXIT trap would turn a good job's exit code into 1 (audit 2026-10-11)
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "${PLS_JOB_TMPDIR:-}" = "/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID" ]; then
    rm -rf -- "$PLS_JOB_TMPDIR" || true
  fi
  return 0
}
# A PLS_JOB_TMPDIR inherited from the submit shell (another job's dir) is replaced, not used and later deleted (audit
# 2026-10-11); a second source inside the same job sees its own dir and does nothing.
if [ -n "${SLURM_JOB_ID:-}" ] && [ "${PLS_JOB_TMPDIR:-}" != "/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID" ]; then
  export PLS_JOB_TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID
  mkdir -p "$PLS_JOB_TMPDIR"
  export TMPDIR=$PLS_JOB_TMPDIR TMP=$PLS_JOB_TMPDIR TEMP=$PLS_JOB_TMPDIR APPTAINERENV_TMPDIR=$PLS_JOB_TMPDIR
  _pls_prev_exit_trap() { eval "set -- $(trap -p EXIT)"; printf '%s' "${3:-}"; }
  _pls_prev=$(_pls_prev_exit_trap)
  trap "${_pls_prev:+$_pls_prev; }pls_tmp_cleanup" EXIT
  unset _pls_prev
  unset -f _pls_prev_exit_trap
fi
