#!/bin/bash
# FIX_ffuzz.md: the two F-Fuzz runs again, targets advanced by 4 samples. Flags of
# param_*_ffuzz_r2 (scripts/product_param_queue_r2.sh) plus --target-advance 4.
source "$(dirname "$0")/product_queue_lib.sh"
COMMON="--seed 42 --split-seed 925 --deterministic --l1-weight 0.5 --mrstft-weight 0.5"
COMMON="${COMMON} --device ffuzz --target-advance 4"
run_once param_ssm_wavenet_ffuzz_adv4 ${COMMON} --model ssm-wavenet --discretization zoh --honor-optim
run_once param_s4_ffuzz_adv4 ${COMMON} --model s4-param
echo "=== F-Fuzz fix finished $(date '+%a %d %H:%M')"
