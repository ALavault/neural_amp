#!/bin/bash
# After vs100r_gc_film_joint_s925 learned nothing (its ReLU output head closed; RUNS.jsonl),
# four 600-step probes, then one full run (user decision, 2026-09-27). The order and the
# rule are fixed before any probe: the full run _r2 takes the first arm, in this order,
# whose probe leaves the plateau (last validation loss < 2.0 and test ESR < 0.9). All four
# probes run whatever happens, for the diagnosis. The GPU is ours during the pause of
# G2.2, so only the first probe waits for it.
source "$(dirname "$0")/product_queue_lib.sh"
COMMON="--model ssm-wavenet --device vs100r --discretization zoh --honor-optim"
COMMON="${COMMON} --deterministic --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5"
COMMON="${COMMON} --cond film --cond-head joint --cond-hidden 32"
ARMS=(
  "leaky_g1_s42|--head-act leaky --seed 42"
  "relu_g0_s42|--no-polarity-guard --seed 42"
  "leaky_g0_s42|--head-act leaky --no-polarity-guard --seed 42"
  "relu_g1_s43|--seed 43"
)
chosen=""
for arm in "${ARMS[@]}"; do
  name=${arm%%|*}
  flags=${arm#*|}
  run_once "vs100r_probe_${name}" ${COMMON} ${flags} --max-steps 600
  wait_gpu() { :; }
  if .venv/bin/python3 scripts/product_vs100r_probe_check.py "vs100r_probe_${name}"; then
    echo "=== probes: ${name} leaves the plateau"
    [ -z "${chosen}" ] && chosen="${flags}" && echo "=== probes: full run takes ${name}"
  else
    echo "=== probes: ${name} stays on the plateau"
  fi
done
if [ -z "${chosen}" ]; then
  echo "=== probes: no arm leaves the plateau, no full run"; exit 1
fi
run_once "vs100r_gc_film_joint_s925_r2" ${COMMON} ${chosen} \
  || { echo "=== VS100R r2 failed"; exit 1; }
echo "=== VS100R r2 finished $(date '+%a %d %H:%M')"
