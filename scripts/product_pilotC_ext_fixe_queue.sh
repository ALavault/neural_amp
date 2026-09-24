#!/bin/bash
# Pilot C extension, phase 3 (diagnosis/butterfly/pilot_C_extension.md): the eight
# "fixe" arms, only once the early/late classification is committed.
source "$(dirname "$0")/product_queue_lib.sh"
if ! git cat-file -e HEAD:diagnosis/butterfly/pilot_C_extension_groups.json 2>/dev/null; then
  echo "=== classification not committed at HEAD: phase 2 refused"
  exit 1
fi
COMMON="--model ssm-wavenet --discretization zoh --honor-optim --deterministic"
for seed in 50 51 52 53 54 55 56 57; do
  run_once "pilotC_fixe_seed${seed}" ${COMMON} --seed "${seed}" \
    --lr-halvings 203 265 422 486 565 608 661 722 --no-early-stopping --max-steps 5950
done
echo "=== pilot C extension, fixe arms finished $(date '+%a %d %H:%M')"
