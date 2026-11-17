#!/bin/bash
# First task of a plan whose runs read caches that may still be building (controller 2026-11-17: a ready 4-GPU wq job can
# start in any gap of a host). Exit 0 if every cache named in NEED (comma-separated names under MARIN_PREFIX/synth) is
# complete: stats.json (written after the cache is closed) and a finished ledger. Else exit 98, which puts a --partial wq
# job back in the queue (not before 30 min; the run is not counted). Plan line, mode smoke so that a 98 ends the job:
#   smoke check - other 4 5m scripts/della/synth/need_caches.sh NEED=name1,name2
set -u
P=${MARIN_PREFIX:-/scratch/gpfs/KARTHIKN/wc9403/marin_store_big}
[ -n "${NEED:-}" ] || { echo "set NEED to cache names"; exit 2; }
for n in ${NEED//,/ }; do
  if [ -f "$P/synth/$n/stats.json" ] && grep -q '"is_finished": true' "$P/synth/$n/train/shard_ledger.json" 2>/dev/null; then
    echo "cache $n complete"
  else
    echo "cache $n not complete: exit 98, back in the wq queue"
    exit 98
  fi
done
