#!/bin/bash
# FFT 288 000 without TF32 against the current setting (demo/fft_size/
# predictions_fast_notf32.md, pre-registered 2026-10-04). Runs from the runner given as
# $1, whose lib cd's there: a bitwise gate of the default path (300 steps of
# vs100r_gc_film_joint_s925_r2), then arm C for seeds 42, 43 and 44, each replaying the
# learning-rate halving steps of the arm-A run of the same seed. A failure stops it.
RUNNER=${1:?usage: product_fft_fast_queue.sh <runner path>}
source "${RUNNER}/scripts/product_queue_lib.sh"
GPU_FREE_MIB=${GPU_FREE_MIB:-21000}
COMMON="--model ssm-wavenet --device vs100r --discretization zoh --honor-optim"
COMMON="${COMMON} --deterministic --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5"
COMMON="${COMMON} --cond film --cond-head joint --cond-hidden 32 --head-act leaky"

halving_steps() {
  .venv/bin/python3 - "$1" <<'EOF'
import csv, sys
path = f"demo/runs/nablafx_{sys.argv[1]}/logs/metrics.csv"
previous, steps = None, []
for row in csv.DictReader(open(path)):
    lr = row.get("lr-AdamW/pg1")
    if lr:
        if previous is not None and lr != previous:
            steps.append(row["step"])
        previous = lr
print(" ".join(steps))
EOF
}

gate=repro_gate_vs100r48_fftsize
wait_gpu
echo "=== ${gate}: start $(date '+%a %d %H:%M')"
.venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${gate}" --no-record \
  ${COMMON} --seed 42 --max-steps 300 > "demo/runs/${gate}.log" 2>&1 \
  || { echo "=== ${gate}: bench exit $?, queue stopped"; exit 1; }
GATE="${gate}" .venv/bin/python3 - <<'EOF' || { echo "=== gate FAILED: queue stopped"; exit 1; }
import json, os, sys
from datetime import datetime
sys.path.insert(0, "diagnosis/seeds")
import repro_gate
gate, reference = os.environ["GATE"], "vs100r_gc_film_joint_s925_r2"
r = repro_gate.compare(gate, reference)
print(f"=== {gate}: {r['values_compared']} values, {r['mismatches']} mismatches, identical {r['identical']}, last step {r['last_step']}")
with open("demo/RUNS.jsonl", "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"run_id": gate, "status": "gate", "reference": reference,
        "identical": r["identical"], "mismatches": r["mismatches"],
        "values_compared": r["values_compared"], "last_step": r["last_step"],
        "first_mismatches": r["first_mismatches"], "environment": repro_gate.environment(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
sys.exit(0 if r["identical"] and r["last_step"] >= 299 else 1)
EOF

for pair in "42 vs100r_gc_film_joint_s925_r2" "43 vs100r48_film_s43" "44 vs100r48_film_s44"; do
  set -- ${pair}
  seed=$1 twin=$2
  steps=$(halving_steps "${twin}")
  echo "=== vs100r48fast_film_s${seed}: halvings of ${twin} at steps [${steps}]"
  run_once "vs100r48fast_film_s${seed}" ${COMMON} --seed "${seed}" --no-early-stopping \
    --fft-size fast --no-tf32 ${steps:+--lr-halving-steps ${steps}} \
    || { echo "=== C s${seed} failed, queue stopped"; exit 1; }
done
echo "=== FFT fast queue finished $(date '+%a %d %H:%M')"
