#!/bin/bash
# Replaces product_chain_after_pilotC.sh from G2.2 on, to put the first VS100R model
# before the rest (user decision, 2026-09-26). Waits for the running G2.2 queue (its PID
# as argument), then VS100R from this runner, then the remaining queues from the G2
# runner, whose pinned code they were registered with. Each queue is independent.
G22_PID="$1"
G2_RUNNER=/fastdata/lavaulta/neural_amp_runner_g2
cd "$(dirname "$0")/.."
while kill -0 "${G22_PID}" 2>/dev/null; do sleep 300; done
echo "=== chain: G2.2 queue gone $(date '+%a %d %H:%M')"
echo "=== chain: vs100r start $(date '+%a %d %H:%M')"
bash scripts/product_vs100r_queue.sh
echo "=== chain: vs100r end $(date '+%a %d %H:%M') (exit $?)"
for queue in paired_ext ffuzz_fix foh_training; do
  echo "=== chain: ${queue} start $(date '+%a %d %H:%M')"
  bash "${G2_RUNNER}/scripts/product_${queue}_queue.sh"
  echo "=== chain: ${queue} end $(date '+%a %d %H:%M') (exit $?)"
done
echo "=== chain finished $(date '+%a %d %H:%M')"
