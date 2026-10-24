#!/bin/bash
# cbw_cleanup.sh RUN_ID...: delete the checkpoints of A/B or smoke runs of the XLA_CMD_BUFFER_WHILE test, nothing else.
# Each RUN_ID must be a full muonh-qwen3-* id with an A/B tag (-abbase / -abcbw / -j<digits>-base / -j<digits>-cbw);
# anything else is refused, so an empty or partial id can never reach speedrun/ as a whole (rm -rf rule, 2026-10).
# Removes <MARIN_PREFIX>/speedrun/<id>/*/checkpoints/step-* (final and step checkpoints) and the temporary (resume)
# checkpoints under <MARIN_PREFIX>/tmp/checkpoints-temp/<...>/speedrun/<id>. Keeps eval_metrics.jsonl and executor files.
set -u
P=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
rc=0
for id in "$@"; do
  case "$id" in
    muonh-qwen3-*-abbase|muonh-qwen3-*-abcbw|muonh-qwen3-*-j[0-9]*-base*|muonh-qwen3-*-j[0-9]*-cbw*) ;;
    *) echo "cbw_cleanup: refusing '$id' (not an A/B run id)"; rc=1; continue ;;
  esac
  for d in "$P/speedrun/$id"/*/checkpoints/step-* "$P/tmp/checkpoints-temp$P/speedrun/$id"; do
    [ -e "$d" ] || continue
    case "$d" in "$P/speedrun/$id/"*|"$P/tmp/checkpoints-temp$P/speedrun/$id") rm -rf -- "$d" && echo "cbw_cleanup: removed $d" ;; esac
  done
done
exit $rc
