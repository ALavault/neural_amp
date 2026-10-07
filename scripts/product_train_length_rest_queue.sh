#!/bin/bash
# 2026-10-07, rescheduled on the user's request: the remaining train-length runs (seeds 43
# and 44, arms L10 and L20) after the tracks queue. Same commands as
# product_train_length_queue.sh, with a new gate id (the first gate's run directory is
# never overwritten) and no smokes (their timings were measured on 2026-10-06).
# Training length per take (demo/train_length/predictions.md, pre-registered 2026-10-05).
# Runs from the runner given as $1, whose lib cd's there: a bitwise gate of the default
# path (300 steps of vs100r_gc_film_joint_s925_r2), two timed 300-step smokes (arm C's
# flags, all segments then 10 per take) for the cost of the more frequent validation,
# then arms L10 and L20 for seeds 42, 43 and 44, each replaying the learning-rate halving
# steps of the arm-A run of the same seed, as arm C did. After each run: its validation
# must be arm C's and its training set 280 or 564 segments, written to demo/RUNS.jsonl.
# A failure stops the queue.
RUNNER=${1:?usage: product_train_length_queue.sh <runner path>}
source "${RUNNER}/scripts/product_queue_lib.sh"
GPU_FREE_MIB=21000  # after the lib, which sets its own default
COMMON="--model ssm-wavenet --device vs100r --discretization zoh --honor-optim"
COMMON="${COMMON} --deterministic --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5"
COMMON="${COMMON} --cond film --cond-head joint --cond-hidden 32 --head-act leaky"
FAST="--no-early-stopping --fft-size fast --no-tf32"

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

gate=repro_gate_vs100r48_trainlen_r2
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

check() {
  RUN="$1" REF="$2" EXPECTED="$3" .venv/bin/python3 - <<'EOF'
import json, os, sys
from datetime import datetime
run, ref, expected = os.environ["RUN"], os.environ["REF"], int(os.environ["EXPECTED"])
r = json.load(open(f"demo/nablafx_bench/{run}.json"))
c = json.load(open(f"demo/nablafx_bench/{ref}.json"))
ok = r["val_indices"] == c["val_indices"] and r["train_segments"] == expected
print(f"=== {run}: validation as {ref} {r['val_indices'] == c['val_indices']}, train segments {r['train_segments']} (expected {expected})")
with open("demo/RUNS.jsonl", "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"run_id": run, "status": "check", "reference": ref,
        "val_identical": r["val_indices"] == c["val_indices"],
        "train_segments": r["train_segments"], "expected": expected, "ok": ok,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
sys.exit(0 if ok else 1)
EOF
}

for pair in "43 vs100r48_film_s43" "44 vs100r48_film_s44"; do
  set -- ${pair}
  seed=$1 twin=$2
  steps=$(halving_steps "${twin}")
  for count in 10 20; do
    run="vs100r48len${count}_film_s${seed}"
    expected=$([ "${count}" = 10 ] && echo 280 || echo 564)
    echo "=== ${run}: halvings of ${twin} at steps [${steps}]"
    run_once "${run}" ${COMMON} ${FAST} --seed "${seed}" \
      --train-segments-per-take "${count}" ${steps:+--lr-halving-steps ${steps}} \
      || { echo "=== ${run} failed, queue stopped"; exit 1; }
    check "${run}" "vs100r48fast_film_s${seed}" "${expected}" \
      || { echo "=== ${run}: check FAILED, queue stopped"; exit 1; }
  done
done
echo "=== train length queue finished $(date '+%a %d %H:%M')"
