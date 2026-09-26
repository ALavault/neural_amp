#!/bin/bash
# First exploratory VS100R model (demo/capture/valvestate/QC.md): parametric SSM-WaveNet,
# FiLM with a joint head (arm A of G2), OD2 channel, flags of G2 on the new device.
source "$(dirname "$0")/product_queue_lib.sh"
COMMON="--model ssm-wavenet --device vs100r --discretization zoh --honor-optim"
COMMON="${COMMON} --deterministic --seed 42 --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5"
A="--cond film --cond-head joint --cond-hidden 32"
run_once "vs100r_film_joint_s925" ${COMMON} ${A} \
  || { echo "=== VS100R stopped: run failed"; exit 1; }
echo "=== VS100R finished $(date '+%a %d %H:%M')"
