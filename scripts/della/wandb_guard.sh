# Sourced by job scripts after `module load proxy/default` (2026-10-16: from about 22:00 new W&B runs on compute nodes
# could not initialize through the proxy, "Network error (ConnectionError)" then a 90 s init timeout, and Levanter hung,
# holding its GPUs until the task limit). Probes api.wandb.ai through the proxy for up to 20 s; when that fails the run
# logs offline (WANDB_MODE=offline) and is listed in logs/wandb_offline.tsv for a later `wandb sync` from a vis node.
if ! curl -sS -m 20 -o /dev/null -w '%{http_code}' https://api.wandb.ai/healthz 2>/dev/null | grep -q '^200$'; then
  export WANDB_MODE=offline
  echo "wandb_guard: api.wandb.ai not reachable through the proxy; WANDB_MODE=offline" >&2
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "${FETCHED_JOB_ID:-${SLURM_JOB_ID:-nojob}}" "${PAIR_LOG_TAG:--}" "$(hostname)" "${SIZE:-}" \
    >> /scratch/gpfs/GROUP/USER/project/marin/logs/wandb_offline.tsv || true
fi
