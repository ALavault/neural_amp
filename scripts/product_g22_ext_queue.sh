#!/bin/bash
# G2.2 extension to splits 744 and 369 (demo/param/predictions_poles.md, "Élargissement de
# G2.2", pre-registered 2026-09-30). Runs from the G2.2 r2 runner (561eef4, pole floor of
# 2d49ac7), whose lib cd's there: a bitwise gate of the poles and FiLM paths under driver
# 595.91.07, stopping at the first mismatch; then arms A and D, flags of
# product_g22_queue.sh, interleaved by split. A failed run stops the queue.
source /fastdata/lavaulta/neural_amp_runner_g22r2/scripts/product_queue_lib.sh
COMMON="--model ssm-wavenet --device marshall --discretization zoh --honor-optim"
COMMON="${COMMON} --deterministic --seed 42 --l1-weight 0.5 --mrstft-weight 0.5"
A="--cond film --cond-head joint --cond-hidden 32"
D="--cond poles --cond-head additive --cond-hidden 7 7 6 6"
gate() {
  local run_id=$1 reference=$2
  shift 2
  wait_gpu
  echo "=== ${run_id}: start $(date '+%a %d %H:%M')"
  .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${run_id}" --no-record "$@" \
    > "demo/runs/${run_id}.log" 2>&1 \
    || { echo "=== ${run_id}: bench exit $?, queue stopped"; exit 1; }
  GATE="${run_id}" REFERENCE="${reference}" .venv/bin/python3 - <<'EOF' || { echo "=== ${run_id} FAILED: queue stopped"; exit 1; }
import json, os, sys
from datetime import datetime
sys.path.insert(0, "diagnosis/seeds")
import repro_gate
gate, reference = os.environ["GATE"], os.environ["REFERENCE"]
r = repro_gate.compare(gate, reference)
print(f"=== {gate}: {r['values_compared']} values, {r['mismatches']} mismatches, identical {r['identical']}, last step {r['last_step']}")
with open("demo/RUNS.jsonl", "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"run_id": gate, "status": "gate", "reference": reference,
        "identical": r["identical"], "mismatches": r["mismatches"],
        "values_compared": r["values_compared"], "last_step": r["last_step"],
        "first_mismatches": r["first_mismatches"], "environment": repro_gate.environment(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
# The gate must have reached its --max-steps 300 (last logged step 299).
sys.exit(0 if r["identical"] and r["last_step"] >= 299 else 1)
EOF
}
gate repro_gate_g22ext_poles_marshall g2_poles_additive_s925 ${COMMON} --split-seed 925 \
  ${D} --max-steps 300
gate repro_gate_g22ext_film_marshall param_ssm_wavenet_marshall ${COMMON} --split-seed 925 \
  --max-steps 300
for split in 744 369; do
  run_once "g2_film_joint_s${split}" ${COMMON} --split-seed "${split}" ${A} \
    || { echo "=== G2.2 extension stopped: A at ${split} failed"; exit 1; }
  run_once "g2_poles_additive_s${split}" ${COMMON} --split-seed "${split}" ${D} \
    || { echo "=== G2.2 extension stopped: D at ${split} failed"; exit 1; }
done
echo "=== G2.2 extension finished $(date '+%a %d %H:%M')"
