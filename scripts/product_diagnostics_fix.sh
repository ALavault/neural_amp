#!/bin/bash
# Rerun the two diverged runs at lr=0.005
set -eu
export TMPDIR=/fastdata/lavaulta/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /fastdata/lavaulta/neural_amp

echo "=== Seed 2 (lr=0.005) ==="
uv run python scripts/product_train_ssm.py \
  --max-steps 15000 --seed 2 --lr 0.005 \
  --run-id ssm_bigmuff_resplit_seed2 --resume \
  2>&1 | grep -vE "warn\(msg\)|UserWarning|^\s*$"

echo ""
echo "=== Original splits (lr=0.005) ==="
uv run python scripts/product_train_ssm.py \
  --max-steps 15000 --lr 0.005 \
  --data-dir datasets/raw/internal_m4/electro_harmonix_big_muff \
  --run-id ssm_bigmuff_original_splits --resume \
  2>&1 | grep -vE "warn\(msg\)|UserWarning|^\s*$"

echo ""
echo "=== Terminé ==="
