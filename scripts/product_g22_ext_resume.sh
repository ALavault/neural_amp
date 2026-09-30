#!/bin/bash
# Continuation of product_g22_ext_queue.sh, whose shell was killed on 2026-09-30 at
# 17:00:55 by a stray `pkill -f queue.sh` (a literature-search agent launched by this
# session for another task).
# Its run g2_poles_additive_s744 (PID given as argument) kept running, orphaned. This waits
# for it, records its failure as run_once would have if it wrote no JSON, then runs the
# two remaining runs of the pre-registered extension under their never-started ids, with
# the same flags and runner. The gates are not replayed: they passed the same day, same
# code (561eef4) and driver.
ORPHAN_PID="$1"
source /fastdata/lavaulta/neural_amp_runner_g22r2/scripts/product_queue_lib.sh
while kill -0 "${ORPHAN_PID}" 2>/dev/null; do sleep 120; done
echo "=== g2_poles_additive_s744: orphan ended $(date '+%a %d %H:%M')"
if [ ! -f demo/nablafx_bench/g2_poles_additive_s744.json ]; then
  printf '{"run_id": "g2_poles_additive_s744", "status": "failed", "exit_code": null, "timestamp": "%s", "note": "orphaned after its queue shell was killed; exit code unknown", "log": "demo/runs/g2_poles_additive_s744.log"}\n' \
    "$(date -Iseconds)" >> demo/RUNS.jsonl
  echo "=== g2_poles_additive_s744: FAILED, recorded in demo/RUNS.jsonl; extension stopped"
  exit 1
fi
COMMON="--model ssm-wavenet --device marshall --discretization zoh --honor-optim"
COMMON="${COMMON} --deterministic --seed 42 --l1-weight 0.5 --mrstft-weight 0.5"
A="--cond film --cond-head joint --cond-hidden 32"
D="--cond poles --cond-head additive --cond-hidden 7 7 6 6"
run_once g2_film_joint_s369 ${COMMON} --split-seed 369 ${A} \
  || { echo "=== G2.2 extension stopped: A at 369 failed"; exit 1; }
run_once g2_poles_additive_s369 ${COMMON} --split-seed 369 ${D} \
  || { echo "=== G2.2 extension stopped: D at 369 failed"; exit 1; }
echo "=== G2.2 extension finished $(date '+%a %d %H:%M')"
