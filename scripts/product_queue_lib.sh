#!/bin/bash
# Shared by the queues launched from 2026-09-24 on. Two rules the older queues broke:
# a run is never resumed or retried under the same run_id (a resumed run is not bitwise
# equal to an uninterrupted deterministic one), and a failure is written to
# demo/RUNS.jsonl before the queue moves on. A relaunch takes a new id, e.g. _r2.
# Detach the calling queue with: nohup setsid bash <queue> >> <log> 2>&1 < /dev/null &
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/fastdata/lavaulta/tmp
# Queues run from a worktree pinned at a commit (scripts/product_runner_worktree.sh), so
# code edited in the main tree during a multi-day queue never reaches its runs. The
# venv's editable install points at the main tree's src/, hence PYTHONPATH.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}/src"

# The GPU is shared with RACLETTE and ollama: wait for 18.5 GiB free six minutes running.
wait_gpu() {
  local free_minutes=0
  while [ "${free_minutes}" -lt 6 ]; do
    if [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 18500 ]; then
      free_minutes=$((free_minutes + 1))
    else
      free_minutes=0
    fi
    sleep 60
  done
}

# run_once <run_id> <bench args...>: one attempt, output in demo/runs/<run_id>.log.
run_once() {
  local run_id=$1
  shift
  if [ -f "demo/nablafx_bench/${run_id}.json" ]; then
    echo "=== ${run_id}: done, skipped"
    return 0
  fi
  wait_gpu
  echo "=== ${run_id}: start $(date '+%a %d %H:%M')"
  .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${run_id}" "$@" \
    > "demo/runs/${run_id}.log" 2>&1
  local code=$?
  echo "=== ${run_id}: end $(date '+%a %d %H:%M') (exit ${code})"
  if [ ! -f "demo/nablafx_bench/${run_id}.json" ]; then
    printf '{"run_id": "%s", "status": "failed", "exit_code": %d, "timestamp": "%s", "args": "%s", "log": "demo/runs/%s.log"}\n' \
      "${run_id}" "${code}" "$(date -Iseconds)" "$*" "${run_id}" >> demo/RUNS.jsonl
    echo "=== ${run_id}: FAILED, recorded in demo/RUNS.jsonl"
    return 1
  fi
}
