#!/bin/bash
# G2.2 (demo/param/predictions_poles.md, verdict G2.1): arms A (FiLM joint) and D (poles
# additive) at splits 736 and 688, flags of G2.1, interleaved by split so that a stopped
# queue still leaves complete splits. A failed run stops G2.2.
source "$(dirname "$0")/product_queue_lib.sh"
COMMON="--model ssm-wavenet --device marshall --discretization zoh --honor-optim"
COMMON="${COMMON} --deterministic --seed 42 --l1-weight 0.5 --mrstft-weight 0.5"
A="--cond film --cond-head joint --cond-hidden 32"
D="--cond poles --cond-head additive --cond-hidden 7 7 6 6"
for split in 736 688; do
  run_once "g2_film_joint_s${split}" ${COMMON} --split-seed "${split}" ${A} \
    || { echo "=== G2.2 stopped: A at ${split} failed"; exit 1; }
  run_once "g2_poles_additive_s${split}" ${COMMON} --split-seed "${split}" ${D} \
    || { echo "=== G2.2 stopped: D at ${split} failed"; exit 1; }
done
echo "=== G2.2 finished $(date '+%a %d %H:%M')"
