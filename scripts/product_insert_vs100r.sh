#!/bin/bash
# Put the first VS100R model between two G2.2 runs (user decision, 2026-09-27): when D at
# split 736 ends, pause the G2.2 queue shell (SIGSTOP, never a run), run VS100R alone on
# the GPU from its pinned runner, then resume the queue (SIGCONT, also on any exit).
# Does nothing if the queue is gone or its next run has already started.
# Usage: product_insert_vs100r.sh <G2.2 queue PID>
G22_PID="$1"
LOG=demo/runs/chain_after_pilotC.log
cd "$(dirname "$0")/.."
until grep -q "=== g2_poles_additive_s736: end" "${LOG}"; do sleep 20; done
if ! ps -o args= -p "${G22_PID}" | grep -q "product_g22_queue.sh"; then
  echo "=== insert: G2.2 queue ${G22_PID} gone, nothing done $(date '+%a %d %H:%M')"; exit 0
fi
if grep -q "=== g2_film_joint_s688: start" "${LOG}"; then
  echo "=== insert: A at 688 already started, nothing done $(date '+%a %d %H:%M')"; exit 0
fi
trap 'kill -CONT "${G22_PID}" 2>/dev/null; echo "=== insert: G2.2 queue resumed $(date "+%a %d %H:%M")"' EXIT
kill -STOP "${G22_PID}"
echo "=== insert: G2.2 queue paused $(date '+%a %d %H:%M')"
bash /fastdata/lavaulta/neural_amp_runner_vs/scripts/product_vs100r_queue.sh
echo "=== insert: VS100R queue exit $? $(date '+%a %d %H:%M')"
