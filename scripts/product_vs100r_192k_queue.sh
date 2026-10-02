#!/bin/bash
# VS100R trained at 192 kHz against trained at 48 kHz (demo/capture/valvestate/
# predictions_192k.md, pre-registered 2026-10-02). Runs from the runner given as $1
# (scripts/product_runner_worktree.sh), whose lib cd's there:
# 1. bitwise gate of the 48 kHz path (300 steps of vs100r_gc_film_joint_s925_r2);
# 2. 192 kHz preflight, 40 steps without record (peak memory, step time);
# 3. B192 seed 42, then A48 and B192 for seeds 43 and 44. Each B192 run halves its
#    learning rate at the global steps where its A48 twin's plateau detector did.
# A failed step stops the queue.
RUNNER=${1:?usage: product_vs100r_192k_queue.sh <runner path>}
source "${RUNNER}/scripts/product_queue_lib.sh"
# The 48 kHz run peaked at 20.4 GiB reserved.
GPU_FREE_MIB=${GPU_FREE_MIB:-21000}
COMMON="--model ssm-wavenet --discretization zoh --honor-optim --deterministic"
COMMON="${COMMON} --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5"
COMMON="${COMMON} --cond film --cond-head joint --cond-hidden 32 --head-act leaky"

halving_steps() {
  # Global steps at which the twin's logged learning rate changed.
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

gate=repro_gate_vs100r48_192kbench
wait_gpu
echo "=== ${gate}: start $(date '+%a %d %H:%M')"
.venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${gate}" --no-record \
  --device vs100r ${COMMON} --seed 42 --max-steps 300 > "demo/runs/${gate}.log" 2>&1 \
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

preflight=preflight_vs100r192
wait_gpu
echo "=== ${preflight}: start $(date '+%a %d %H:%M')"
.venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${preflight}" --no-record \
  --device vs100r_192k ${COMMON} --seed 42 --max-steps 40 --no-early-stopping \
  --lr-halving-steps 7830 10701 > "demo/runs/${preflight}.log" 2>&1 \
  || { echo "=== ${preflight}: bench exit $?, queue stopped"; exit 1; }
grep -o '"peak_gpu[a-z_]*": [0-9.]*\|"minutes_this_session": [0-9.]*' \
  "demo/runs/${preflight}.log" | sed 's/^/=== preflight /'

b192() {
  local seed=$1 twin=$2 steps
  steps=$(halving_steps "${twin}")
  echo "=== vs100r192_film_s${seed}: halvings of ${twin} at steps [${steps}]"
  run_once "vs100r192_film_s${seed}" --device vs100r_192k ${COMMON} --seed "${seed}" \
    --no-early-stopping ${steps:+--lr-halving-steps ${steps}} \
    || { echo "=== B192 s${seed} failed, queue stopped"; exit 1; }
}
b192 42 vs100r_gc_film_joint_s925_r2
for seed in 43 44; do
  run_once "vs100r48_film_s${seed}" --device vs100r ${COMMON} --seed "${seed}" \
    --no-early-stopping || { echo "=== A48 s${seed} failed, queue stopped"; exit 1; }
  b192 "${seed}" "vs100r48_film_s${seed}"
done
echo "=== VS100R 192 kHz queue finished $(date '+%a %d %H:%M')"
