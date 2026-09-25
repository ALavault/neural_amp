#!/bin/bash
# FOH training, optional (demo/sr_transfer/predictions_foh_training.md): the three paired
# SSM-WaveNet runs again with --discretization foh, same seed, split and flags.
source "$(dirname "$0")/product_queue_lib.sh"
for split in 925 736 688; do
  run_once "paired_ssm_wavenet_foh_s${split}" --model ssm-wavenet --discretization foh \
    --honor-optim --deterministic --seed 42 --split-seed "${split}"
done
echo "=== FOH training finished $(date '+%a %d %H:%M')"
