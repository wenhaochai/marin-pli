# Source me from the pls worktree. The shared venv (project/marin/.venv) installs every lib editable from
# project/marin; put this worktree's sources first so its levanter/haliax/marin/... and experiments are the ones imported.
WT=/scratch/gpfs/GROUP/USER/project/marin-pls
PY=/scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python
_pp=$WT
for d in "$WT"/lib/*/src; do _pp=$_pp:$d; done
export PYTHONPATH=$_pp${PYTHONPATH:+:$PYTHONPATH}
unset _pp d
