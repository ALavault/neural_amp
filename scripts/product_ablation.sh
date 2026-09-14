#!/bin/bash
# Run three SSM-WaveNet ablations on the resplit Big Muff, sequentially.
# Each run checkpoints every 500 steps and can be resumed after a kill.
set -eu
export TMPDIR=/fastdata/lavaulta/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /fastdata/lavaulta/neural_amp

echo "=== Run 1/3: baseline + 30k steps ==="
uv run python scripts/product_train_ssm.py \
  --max-steps 30000 --run-id ssm_v2_30k --resume \
  2>&1 | grep -vE "warn\(msg\)|UserWarning|^\s*$"

echo ""
echo "=== Run 2/3: circuit-init + deriv loss ==="
uv run python scripts/product_train_ssm.py \
  --circuit-init --deriv-weight 0.1 \
  --max-steps 15000 --run-id ssm_v2_circuit_deriv --resume \
  2>&1 | grep -vE "warn\(msg\)|UserWarning|^\s*$"

echo ""
echo "=== Run 3/3: sine activation + circuit-init + deriv loss ==="
uv run python scripts/product_train_ssm.py \
  --act-type sine --circuit-init --deriv-weight 0.1 \
  --max-steps 15000 --run-id ssm_v2_sine_circuit_deriv --resume \
  2>&1 | grep -vE "warn\(msg\)|UserWarning|^\s*$"

echo ""
echo "=== Tous les runs terminés ==="
