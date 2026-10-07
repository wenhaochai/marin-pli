#!/bin/bash
# Run a command in its own process group and pass TERM/INT to the whole group (then KILL after 30 s), so that
# packed_task.sh's timeout also stops the python a bash job script started. Exit code: the command's.
setsid "$@" < /dev/null &
pid=$!
stop() { kill -TERM -- -$pid 2>/dev/null; sleep 30; kill -KILL -- -$pid 2>/dev/null; }
trap stop TERM INT
wait $pid; rc=$?
while kill -0 $pid 2>/dev/null; do wait $pid; rc=$?; done
exit $rc
