#!/bin/bash
# ToneTwist Big Muff benchmark in the nablafx framework, one run at a time
# (each run peaks at 16-17 GiB on the 24 GiB GPU). A run whose result file
# exists is skipped, so the queue can be restarted after a kill.
#
# Second round, after SSM-WaveNet v1 underfit (train loss 0.43 vs 0.31 for
# S4-TF-L-16) and its learned poles lost all memory beyond 20 ms under weight
# decay: both models without weight decay on state-space parameters, SSM-WaveNet
# with S4D zero-order-hold discretization, then with 32 states per channel.
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

# A run started by an earlier queue may still be training.
while pgrep -f "scripts/product_nablafx_benc[h].py" > /dev/null; do
  sleep 30
done

echo "=== smoke N=32 memory check: start $(date '+%a %H:%M')"
uv run python scripts/product_nablafx_bench.py --run-id smoke_ssmzoh32 --no-record \
  --model ssm-wavenet --discretization zoh --honor-optim --state-dim 32 --max-steps 30 2>&1 \
  | grep -E "peak_gpu_gib|out of memory|Error"
rm -rf demo/runs/nablafx_smoke_ssmzoh32
echo "=== smoke N=32 memory check: end $(date '+%a %H:%M')"

run ssmzoh_seed42 --model ssm-wavenet --discretization zoh --honor-optim --seed 42
run s4tfl16nowd_seed42 --model s4-tf-l-16 --honor-optim --seed 42
run ssmzoh32_seed42 --model ssm-wavenet --discretization zoh --honor-optim --state-dim 32 --seed 42
echo "=== queue finished $(date '+%a %H:%M')"
