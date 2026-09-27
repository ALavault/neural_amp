#!/bin/bash
# Replaces product_chain_after_g22.sh from the paired extension on, to put the G2.2
# relaunch before the rest (user decision, 2026-09-28). Waits for the running paired
# extension queue (its PID as argument), then D at 688 _r2 from this runner, then the
# remaining queues from the G2 runner, whose pinned code they were registered with.
PAIRED_PID="$1"
G2_RUNNER=/fastdata/lavaulta/neural_amp_runner_g2
cd "$(dirname "$0")/.."
while kill -0 "${PAIRED_PID}" 2>/dev/null; do sleep 300; done
echo "=== chain: paired extension queue gone $(date '+%a %d %H:%M')"
echo "=== chain: g22_r2 start $(date '+%a %d %H:%M')"
bash scripts/product_g22_r2_queue.sh
echo "=== chain: g22_r2 end $(date '+%a %d %H:%M') (exit $?)"
for queue in ffuzz_fix foh_training; do
  echo "=== chain: ${queue} start $(date '+%a %d %H:%M')"
  bash "${G2_RUNNER}/scripts/product_${queue}_queue.sh"
  echo "=== chain: ${queue} end $(date '+%a %d %H:%M') (exit $?)"
done
echo "=== chain finished $(date '+%a %d %H:%M')"
