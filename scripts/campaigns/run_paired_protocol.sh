#!/bin/bash
# Pre-check of the paired protocol, pre-registered in diagnosis/seeds/paired_protocol.md.
#
# Two architectures x three declared splits, one run per cell, no replicate: at a fixed
# budget, replicating reduces only the error term and never the interaction, which
# dominates here. The splits come from random.Random(20260922).sample(range(1000), 5),
# inscribed in the pre-registration before any run.
#
# --split-seed makes the train/val partition and the batch order identical across
# architectures; without it two architectures at the same --seed share one validation
# segment in twelve. --deterministic throughout, verified working on both architectures
# by the gate0_* smoke runs.
#
# .venv/bin/python3 rather than uv run: the snap uv could resync the venv mid-comparison.
set -u
cd /fastdata/lavaulta/neural_amp
export TMPDIR=/fastdata/lavaulta/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8

LOG=demo/runs/paired_queue.log
SPLITS="925 736 688"
MODELS="ssm-wavenet s4-tf-l-16"

for split in $SPLITS; do
  for model in $MODELS; do
    id="paired_${model//-/_}_s${split}"
    if [ -f "demo/nablafx_bench/${id}.json" ]; then
      echo "=== $id : deja fait, saute" | tee -a "$LOG"
      continue
    fi
    echo "=== $id : debut $(date '+%a %H:%M')" | tee -a "$LOG"
    .venv/bin/python3 scripts/product_nablafx_bench.py \
      --run-id "$id" --model "$model" --seed 42 --split-seed "$split" \
      --deterministic --honor-optim \
      > "demo/runs/${id}.log" 2>&1
    echo "=== $id : fin $(date '+%a %H:%M') (code $?)" | tee -a "$LOG"
  done
done
echo "=== pre-controle apparie termine $(date '+%a %H:%M')" | tee -a "$LOG"
