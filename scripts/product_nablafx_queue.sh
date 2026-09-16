#!/bin/bash
# ToneTwist Big Muff benchmark in the nablafx framework, one run at a time
# (each run peaks at 16-17 GiB on the 24 GiB GPU). A run whose result file
# exists is skipped, so the queue can be restarted after a kill.
#
# Third round. Round 2's SSM-WaveNet (zero-order hold, no weight decay on
# state-space parameters) fitted well but converged to an inverted output (val
# ESR 3.96, 0.02 sign-flipped) and was killed; every run now trains with
# PolarityGuard. S4-TF-L-16 gets the same two changes (no weight decay on its
# state-space parameters, polarity guard) so that architecture is the only
# difference. Seeds 42, 43, 44 in that order, alternating models; every
# finished run is reported.
set -u
export TMPDIR=/fastdata/lavaulta/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /fastdata/lavaulta/neural_amp

run() {
  local run_id=$1
  shift
  # The GPU is shared with ollama, which loads models of up to 21 GiB on demand
  # and unloads them after 5 idle minutes: a run starts once 18.5 GiB have stayed
  # free for 6 minutes, and a run that fails (out of memory) is started again.
  for attempt in 1 2 3 4 5 6; do
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
}

# A run started by an earlier queue may still be training.
while pgrep -f "scripts/product_nablafx_benc[h].py" > /dev/null; do
  sleep 30
done

for seed in 42 43 44; do
  run "ssmzoh_guard_seed${seed}" --model ssm-wavenet --discretization zoh --honor-optim --seed "${seed}"
  run "s4tfl16_nowd_guard_seed${seed}" --model s4-tf-l-16 --honor-optim --seed "${seed}"
done

# Round 4, so that the comparisons with the released training also have three
# seeds: the S4 baseline as the released code trains it, then the SSM-WaveNet
# ablation (learned b, weight decay on state-space parameters, no guard).
for seed in 42 43 44; do
  run "s4tfl16_seed${seed}" --model s4-tf-l-16 --no-polarity-guard --seed "${seed}"
done
for seed in 42 43 44; do
  run "ssmwavenet_seed${seed}" --model ssm-wavenet --no-polarity-guard --seed "${seed}"
done

# Round 5, seed sensitivity: training is not deterministic on the GPU, so a
# second run at the same seed measures the spread that the seed does not set.
# One repeat per seed completes a 3 seeds x 2 runs layout for SSM-WaveNet
# (seed 42 was repeated on its own), then for S4-TF-L-16 as released.
for seed in 43 44; do
  run "ssmzoh_guard_seed${seed}_repeat" --model ssm-wavenet --discretization zoh --honor-optim --seed "${seed}"
done
for seed in 42 43 44; do
  run "s4tfl16_seed${seed}_repeat" --model s4-tf-l-16 --no-polarity-guard --seed "${seed}"
done
echo "=== queue finished $(date '+%a %H:%M')"
