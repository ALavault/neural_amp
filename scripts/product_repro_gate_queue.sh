#!/bin/bash
# Reproducibility gate of 2026-09-24 (diagnosis/seeds/repro_gate.md): the bench and
# ssm_wavenet.py changed since the reference runs, so each path is retrained for a
# few hundred deterministic steps and its metrics.csv compared bit for bit with the
# reference's (diagnosis/seeds/repro_gate.py). Flags copied from the reference records.
source "$(dirname "$0")/product_queue_lib.sh"
gate() {
  local run_id=$1
  shift
  wait_gpu
  echo "=== ${run_id}: start $(date '+%a %d %H:%M')"
  .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${run_id}" --no-record "$@" \
    > "demo/runs/${run_id}.log" 2>&1
  echo "=== ${run_id}: end $(date '+%a %d %H:%M') (exit $?)"
}
# pilotC_decide_seed42 (commit 189474d)
gate repro_gate_ssm_bigmuff --model ssm-wavenet --discretization zoh --honor-optim \
  --deterministic --seed 42 --max-steps 200
# paired_s4_tf_l_16_s925 (commit 6112d6f)
gate repro_gate_s4tf_bigmuff --model s4-tf-l-16 --honor-optim --deterministic --seed 42 \
  --split-seed 925 --max-steps 200
# param_ssm_wavenet_marshall (commit 29a869c); 143 steps per epoch, so 300 steps
# cover two validation points
gate repro_gate_ssm_marshall --model ssm-wavenet --device marshall --discretization zoh \
  --honor-optim --deterministic --seed 42 --split-seed 925 --l1-weight 0.5 \
  --mrstft-weight 0.5 --max-steps 300
echo "=== repro gate finished $(date '+%a %d %H:%M')"
