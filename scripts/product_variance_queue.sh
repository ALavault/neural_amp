#!/bin/bash
# Pilots E and F (diagnosis/seeds/pilot_E_split.md, pilot_F_threshold.md), one run at a
# time on the shared GPU. Both ask where the run-to-run variance comes from, by changing
# exactly one thing against the six reference runs of the bench.
# E: six seeds, 42 to 47, with the data chain made common (--split-seed 42).
# F: three seeds by two runs, with the plateau threshold raised above the measured noise.
set -u
export TMPDIR=/fastdata/lavaulta/tmp
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
    uv run python scripts/product_nablafx_bench.py --run-id "${run_id}" --resume "$@" 2>&1 \
      | grep -vE "warn\(msg\)|UserWarning|warnings.warn|is differentiable|^\s*$"
    echo "=== ${run_id}: end $(date '+%a %H:%M') (exit ${PIPESTATUS[0]})"
  done
  [ -f "demo/nablafx_bench/${run_id}.json" ] || { echo "=== ${run_id}: failed 3 times, queue stopped"; exit 1; }
}

# Pilot E: the data chain is common, only the initialisation follows --seed.
for seed in 42 43 44 45 46 47; do
  run "splitfix_seed${seed}" --model ssm-wavenet --discretization zoh --honor-optim \
      --seed "${seed}" --split-seed 42
done
echo "=== pilot E finished $(date '+%a %H:%M')"

# Pilot F: the plateau threshold above the noise floor of the validation curve.
for seed in 42 43 44; do
  run "plateau2e2_seed${seed}" --model ssm-wavenet --discretization zoh --honor-optim \
      --seed "${seed}" --plateau-threshold 0.02
  run "plateau2e2_seed${seed}_repeat" --model ssm-wavenet --discretization zoh --honor-optim \
      --seed "${seed}" --plateau-threshold 0.02
done
echo "=== pilot F finished $(date '+%a %H:%M')"
