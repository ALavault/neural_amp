#!/bin/bash
# Pilot C (diagnosis/butterfly/pilot_C_common_schedule.md), one run at a time.
# Eight seeds, two arms, deterministic mode. "decide" is the NablAFx protocol; "fixe"
# replaces its validation-driven decisions by a schedule fixed in advance, whose epochs
# are the rank-by-rank medians of the seven existing SSM-WaveNet runs. The arms alternate
# by seed, so an interrupted queue still leaves a balanced comparison.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /fastdata/lavaulta/neural_amp

run() {
  local run_id=$1
  shift
  for attempt in 1 2 3; do
    if [ -f "demo/nablafx_bench/${run_id}.json" ]; then
      echo "=== ${run_id}: done, skipped"
      return
    fi
    local free_minutes=0
    while [ "${free_minutes}" -lt 6 ]; do
      if [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 18500 ]; then
        free_minutes=$((free_minutes + 1))
      else
        free_minutes=0
      fi
      sleep 60
    done
    echo "=== ${run_id}: start $(date '+%a %H:%M') (attempt ${attempt})"
    .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${run_id}" --resume "$@" 2>&1 \
      | grep -vE "warn\(msg\)|UserWarning|warnings.warn|is differentiable|^\s*$"
    echo "=== ${run_id}: end $(date '+%a %H:%M') (exit ${PIPESTATUS[0]})"
  done
  [ -f "demo/nablafx_bench/${run_id}.json" ] || { echo "=== ${run_id}: failed 3 times, queue stopped"; exit 1; }
}

COMMON="--model ssm-wavenet --discretization zoh --honor-optim --deterministic"
for seed in 42 43 44 45 46 47 48 49; do
  run "pilotC_decide_seed${seed}" ${COMMON} --seed "${seed}"
  run "pilotC_fixe_seed${seed}" ${COMMON} --seed "${seed}" \
      --lr-halvings 203 265 422 486 565 608 661 722 --no-early-stopping --max-steps 5950
done
echo "=== pilot C finished $(date '+%a %H:%M')"
