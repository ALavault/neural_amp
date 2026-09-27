#!/bin/bash
# G2.2, arm D at split 688 relaunched once after its NaN (predictions_poles.md, "Écart de
# G2.2"): flags of product_g22_queue.sh, code with the underflow fix (2d49ac7).
source "$(dirname "$0")/product_queue_lib.sh"
COMMON="--model ssm-wavenet --device marshall --discretization zoh --honor-optim"
COMMON="${COMMON} --deterministic --seed 42 --l1-weight 0.5 --mrstft-weight 0.5"
D="--cond poles --cond-head additive --cond-hidden 7 7 6 6"
run_once "g2_poles_additive_s688_r2" ${COMMON} --split-seed 688 ${D} \
  || { echo "=== G2.2 r2 failed"; exit 1; }
echo "=== G2.2 r2 finished $(date '+%a %d %H:%M')"
