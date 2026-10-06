#!/bin/bash
# KIPR known-answer bench (demo/kipr_known_answer/predictions.md, pre-registered
# 2026-10-06): six short trainings, symmetric and asymmetric targets for seeds 42, 43
# and 44, run from the runner given as $1. About 3.8 GiB of GPU each, hence the low
# free-memory wait. A failure is written to demo/RUNS.jsonl and stops the queue.
RUNNER=${1:?usage: product_kipr_queue.sh <runner path>}
source "${RUNNER}/scripts/product_queue_lib.sh"
GPU_FREE_MIB=${GPU_FREE_MIB:-6000}
for seed in 42 43 44; do
  for config in sym asym; do
    run="kipr_${config}_s${seed}"
    if [ -f "demo/runs/${run}/record.json" ]; then
      echo "=== ${run}: done, skipped"
      continue
    fi
    wait_gpu
    echo "=== ${run}: start $(date '+%a %d %H:%M')"
    .venv/bin/python3 demo/kipr_known_answer/known_answer.py train "${config}" "${seed}" \
      > "demo/runs/${run}.log" 2>&1
    code=$?
    echo "=== ${run}: end $(date '+%a %d %H:%M') (exit ${code})"
    if [ ! -f "demo/runs/${run}/record.json" ]; then
      printf '{"run_id": "%s", "status": "failed", "exit_code": %d, "timestamp": "%s", "log": "demo/runs/%s.log"}\n' \
        "${run}" "${code}" "$(date -Iseconds)" "${run}" >> demo/RUNS.jsonl
      echo "=== ${run}: FAILED, queue stopped"
      exit 1
    fi
  done
done
echo "=== KIPR queue finished $(date '+%a %d %H:%M')"
