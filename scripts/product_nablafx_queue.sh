#!/bin/bash
# ToneTwist Big Muff benchmark in the nablafx framework, one run at a time
# (each run peaks at 16-17 GiB on the 24 GiB GPU). A run whose result file
# exists is skipped, so the queue can be restarted after a kill.
set -u
export TMPDIR=/fastdata/lavaulta/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /fastdata/lavaulta/neural_amp

run() {
  local run_id=$1
  shift
  if [ -f "demo/nablafx_bench/${run_id}.json" ]; then
    echo "=== ${run_id}: done, skipped"
    return
  fi
  echo "=== ${run_id}: start $(date '+%a %H:%M')"
  uv run python scripts/product_nablafx_bench.py --run-id "${run_id}" --resume "$@" 2>&1 \
    | grep -vE "warn\(msg\)|UserWarning|warnings.warn|is differentiable|^\s*$"
  echo "=== ${run_id}: end $(date '+%a %H:%M') (exit ${PIPESTATUS[0]})"
}

run s4tfl16_seed42 --model s4-tf-l-16 --seed 42
run ssmwavenet_seed42 --model ssm-wavenet --seed 42
run ssmwavenet_seed43 --model ssm-wavenet --seed 43
run ssmwavenet_seed44 --model ssm-wavenet --seed 44
run s4l16_seed42 --model s4-l-16 --seed 42
run s4tfl16_seed43 --model s4-tf-l-16 --seed 43
run s4tfl16_seed44 --model s4-tf-l-16 --seed 44
echo "=== queue finished $(date '+%a %H:%M')"
