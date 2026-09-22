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
    # Wait for the GPU to actually be free. A previous run killed mid-flight left 1.7 GiB
    # behind and the next launch died on CUBLAS_STATUS_ALLOC_FAILED within seconds, which
    # the queue happily recorded as "code 0" and moved past.
    for _ in $(seq 60); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -lt 2000 ] && break
      echo "=== $id : GPU occupe (${used} MiB), attente" | tee -a "$LOG"
      sleep 30
    done
    echo "=== $id : debut $(date '+%a %H:%M')" | tee -a "$LOG"
    # --discretization zoh is NOT optional: the flag defaults to "free", and every healthy
    # run in this repository - pilot C, pilot E - used zoh. The first launch of this queue
    # omitted it and trained a different model: test ESR 1.14 against 0.32 on its own best
    # checkpoint, eight polarity flips, stopped at epoch 138 where every prior run reached
    # 408 to 1103. That record is kept under demo/nablafx_bench/void_discretization_free_*.
    .venv/bin/python3 scripts/product_nablafx_bench.py \
      --run-id "$id" --model "$model" --seed 42 --split-seed "$split" \
      --deterministic --honor-optim --discretization zoh \
      > "demo/runs/${id}.log" 2>&1
    code=$?
    if [ ! -f "demo/nablafx_bench/${id}.json" ]; then
      echo "=== $id : ECHEC, aucun enregistrement ecrit (code $code)" | tee -a "$LOG"
      tail -3 "demo/runs/${id}.log" | tr '\r' '\n' | tail -2 | tee -a "$LOG"
      exit 1
    fi
    echo "=== $id : fin $(date '+%a %H:%M') (code $code)" | tee -a "$LOG"
  done
done
echo "=== pre-controle apparie termine $(date '+%a %H:%M')" | tee -a "$LOG"
