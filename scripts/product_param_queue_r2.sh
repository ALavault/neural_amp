#!/bin/bash
# F-Fuzz relaunch with DRY-with-markers (demo/param/DEVIATION.md); same settings.
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
  echo "=== $id : fin $(date '+%a %H:%M') (code $?)" >> "$LOG"
}
run param_ssm_wavenet_ffuzz_r2 --model ssm-wavenet --device ffuzz $SSM
run param_s4_ffuzz_r2 --model s4-param --device ffuzz
echo "=== relance F-Fuzz terminee $(date '+%a %H:%M')" >> "$LOG"
