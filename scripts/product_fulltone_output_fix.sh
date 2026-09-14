#!/bin/bash
# Two runs on Fulltone to compare output activation fixes.
# Run 1: cosine lr at 0.01 with tanh (the lr adapts itself down)
# Run 2: output_act=none with lr=0.001 (removes the bottleneck entirely)
set -eu
export TMPDIR=/fastdata/lavaulta/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /fastdata/lavaulta/neural_amp
DATA=demo/resplit_fulltone

echo "=== Run 1/2: cosine lr 0.01, tanh output ==="
uv run python scripts/product_train_ssm.py \
  --max-steps 15000 --lr 0.01 --cosine-lr \
  --data-dir "$DATA" \
  --run-id ssm_fulltone_cosine --resume \
  2>&1 | grep -vE "warn\(msg\)|UserWarning|^\s*$"

echo ""
echo "=== Run 2/2: no output activation, lr 0.001 ==="
uv run python scripts/product_train_ssm.py \
  --max-steps 15000 --lr 0.001 --output-act none \
  --data-dir "$DATA" \
  --run-id ssm_fulltone_noact --resume \
  2>&1 | grep -vE "warn\(msg\)|UserWarning|^\s*$"

echo ""
echo "=== Terminé ==="
