#!/bin/bash
# Run a queue between two G2.2 runs: once the line "=== <run>: end" is in the G2.2 log,
# pause the G2.2 queue shell (SIGSTOP, never a run), run the given queue alone on the
# GPU, then resume G2.2 (SIGCONT, also on any exit). Does nothing if the G2.2 queue is
# gone or its next run already started. Generalises product_insert_vs100r.sh.
# Usage: product_insert_after.sh <G2.2 queue PID> <run that must end> <next run> <queue>
G22_PID="$1"; AFTER="$2"; NEXT="$3"; QUEUE="$4"
LOG=demo/runs/chain_after_pilotC.log
cd "$(dirname "$0")/.."
until grep -q "=== ${AFTER}: end" "${LOG}"; do sleep 20; done
if ! ps -o args= -p "${G22_PID}" | grep -q "product_g22_queue.sh"; then
  echo "=== insert: G2.2 queue ${G22_PID} gone, nothing done $(date '+%a %d %H:%M')"; exit 0
fi
if grep -q "=== ${NEXT}: start" "${LOG}"; then
  echo "=== insert: ${NEXT} already started, nothing done $(date '+%a %d %H:%M')"; exit 0
fi
trap 'kill -CONT "${G22_PID}" 2>/dev/null; echo "=== insert: G2.2 queue resumed $(date "+%a %d %H:%M")"' EXIT
kill -STOP "${G22_PID}"
echo "=== insert: G2.2 queue paused $(date '+%a %d %H:%M')"
bash "${QUEUE}"
code=$?
echo "=== insert: ${QUEUE} exit ${code} $(date '+%a %d %H:%M')"
