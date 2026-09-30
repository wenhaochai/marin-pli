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
