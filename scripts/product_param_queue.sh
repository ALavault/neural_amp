#!/bin/bash
# Four parametric runs, one at a time, as pre-registered in demo/param/predictions.md.
cd "$(dirname "$0")/.."
export TMPDIR=/fastdata/lavaulta/tmp
LOG=demo/runs/param_queue.log
COMMON="--seed 42 --split-seed 925 --deterministic --l1-weight 0.5 --mrstft-weight 0.5"
SSM="--discretization zoh --honor-optim"
run() {
  local id=$1; shift
  echo "=== $id : debut $(date '+%a %H:%M')" >> "$LOG"
  uv run python scripts/product_nablafx_bench.py --run-id "$id" $COMMON "$@" \
    > "demo/runs/$id.log" 2>&1
  local code=$?
  echo "=== $id : fin $(date '+%a %H:%M') (code $code)" >> "$LOG"
}
run param_ssm_wavenet_marshall --model ssm-wavenet --device marshall $SSM
run param_s4_marshall --model s4-param --device marshall
run param_ssm_wavenet_ffuzz --model ssm-wavenet --device ffuzz $SSM
run param_s4_ffuzz --model s4-param --device ffuzz
echo "=== file parametrique terminee $(date '+%a %H:%M')" >> "$LOG"
