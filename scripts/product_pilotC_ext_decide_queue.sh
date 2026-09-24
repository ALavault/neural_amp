#!/bin/bash
# Pilot C extension, phase 1 (diagnosis/butterfly/pilot_C_extension.md): the eight
# "decide" arms of seeds 50-57, flags identical to scripts/product_pilotC_queue.sh.
# Then the S4-TF-L-16 seed-42 redo (diagnosis/seeds/s4_nested.md), which leaves time to
# commit the early/late classification before phase 2 takes the GPU.
source "$(dirname "$0")/product_queue_lib.sh"
COMMON="--model ssm-wavenet --discretization zoh --honor-optim --deterministic"
for seed in 50 51 52 53 54 55 56 57; do
  run_once "pilotC_decide_seed${seed}" ${COMMON} --seed "${seed}"
done
echo "=== pilot C extension, decide arms finished $(date '+%a %d %H:%M')"
# Flags of s4tfl16_seed42_repeat.json: published training, no guard, no exemption,
# not deterministic.
run_once s4tfl16_seed42_redo --model s4-tf-l-16 --seed 42 --no-polarity-guard
echo "=== queue finished $(date '+%a %d %H:%M')"
