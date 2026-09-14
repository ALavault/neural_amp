#!/bin/bash
# Goal 1: three SSM-WaveNet diagnostics
set -eu
export TMPDIR=/fastdata/lavaulta/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /fastdata/lavaulta/neural_amp

echo "=== Seed 1 (Big Muff resplit, 15k steps) ==="
uv run python scripts/product_train_ssm.py \
  --max-steps 15000 --seed 1 \
  --run-id ssm_bigmuff_resplit_seed1 --resume \
  2>&1 | grep -vE "warn\(msg\)|UserWarning|^\s*$"

echo ""
echo "=== Seed 2 (Big Muff resplit, 15k steps) ==="
uv run python scripts/product_train_ssm.py \
  --max-steps 15000 --seed 2 \
  --run-id ssm_bigmuff_resplit_seed2 --resume \
  2>&1 | grep -vE "warn\(msg\)|UserWarning|^\s*$"

echo ""
echo "=== Original splits (Big Muff, 15k steps) ==="
uv run python scripts/product_train_ssm.py \
  --max-steps 15000 \
  --data-dir datasets/raw/internal_m4/electro_harmonix_big_muff \
  --run-id ssm_bigmuff_original_splits --resume \
  2>&1 | grep -vE "warn\(msg\)|UserWarning|^\s*$"

echo ""
echo "=== Tous les runs terminés ==="
