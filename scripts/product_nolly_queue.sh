#!/bin/bash
# Nolly models (demo/nolly_models/predictions.md, D-0015), from the runner given as $1,
# whose lib cd's there. Launch it once the probe reading passed its gates and
# scripts/product_nolly_dataset.py built the dataset, chained after the queue in
# progress (while kill -0 <pid>), never beside it. Steps: the dataset's counts written
# to demo/RUNS.jsonl; a bitwise gate of the default path (300 steps of
# vs100r_gc_film_joint_s925_r2: it covers the VS100R path only, a 5-input head draws
# other initial values); a 100-step smoke of each arm on Nolly; then per seed C, F3,
# GREY, L, so that a stop leaves complete comparisons. C takes its learning-rate
# halvings from the plateau; the grey boxes replay those of C at the same seed. Each run
# is checked: 15 000 steps, 1 306 training segments, C's split. A failure is written to
# demo/RUNS.jsonl and stops the queue.
RUNNER=${1:?usage: product_nolly_queue.sh <runner path>}
source "${RUNNER}/scripts/product_queue_lib.sh"
NO="--device nolly --discretization zoh --honor-optim --deterministic"
NO="${NO} --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5"
NO="${NO} --cond film --cond-head joint --cond-hidden 32"
NO="${NO} --no-early-stopping --fft-size fast --no-tf32"
C="--model ssm-wavenet --head-act leaky"
GREY="--model greybox"
L="--model deepgreybox --gb-taps 512"
F3="--model deepgreybox --gb-stages 3 --gb-taps 512 --gb-bias --gb-slow 0.05 0.3"
# Run id prefixes of demo/nolly_models/evaluate.py.
declare -A PREFIX=([C]=nolly_C [GREY]=nolly_grey [L]=nolly_gbL [F3]=nolly_gbF3)
# Runs killed without a record, relaunched once as <id>_r2 (evaluate.py reads <id>_r2
# when <id> has no record): list them here, space-separated, before a relaunch.
RELAUNCHED=" "
attempt() {  # attempt <run id>: the id to run under
  case "${RELAUNCHED}" in *" $1 "*) echo "$1_r2" ;; *) echo "$1" ;; esac
}
fresh() {  # fresh <run id>: no earlier attempt left without a record under this id
  if [ -e "demo/runs/nablafx_$1" ] && [ ! -f "demo/nablafx_bench/$1.json" ]; then
    fail "$1" "earlier attempt without a record: relaunch it as $1_r2 (RELAUNCHED)"
  fi
}

fail() {  # fail <run id> <reason>: recorded in demo/RUNS.jsonl, then the queue stops
  printf '{"run_id": "%s", "status": "failed", "reason": "%s", "timestamp": "%s", "log": "demo/runs/%s.log"}\n' \
    "$1" "$2" "$(date -Iseconds)" "$1" >> demo/RUNS.jsonl
  echo "=== $1: $2, queue stopped"
  exit 1
}

halving_steps() {  # steps where C's learning rate changed, from its metrics.csv
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

check() {  # check <run> [<C run> <C's halving steps>]: steps, segment count and, given
  # C, the same split and the halvings replayed (none fixed for C itself: plateau)
  RUN="$1" REF="${2:-}" STEPS="${3:-}" .venv/bin/python3 - <<'EOF'
import json, os, sys
from datetime import datetime
run, ref = os.environ["RUN"], os.environ["REF"]
r = json.load(open(f"demo/nablafx_bench/{run}.json"))
same = None
if ref:
    same = r["val_indices"] == json.load(open(f"demo/nablafx_bench/{ref}.json"))["val_indices"]
halvings = [int(s) for s in os.environ["STEPS"].split()] if ref else None
ok = (r["global_step"] == 15000 and r["train_segments"] == 1306 and same is not False
      and r["lr_halving_steps"] == halvings)
print(f"=== {run}: validation as {ref or '-'} {same}, train segments {r['train_segments']}, step {r['global_step']}, halvings {r['lr_halving_steps']}")
with open("demo/RUNS.jsonl", "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"run_id": run, "status": "check", "reference": ref or None,
        "val_identical": same, "train_segments": r["train_segments"],
        "lr_halving_steps": r["lr_halving_steps"], "expected_halving_steps": halvings,
        "global_step": r["global_step"], "ok": ok,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
sys.exit(0 if ok else 1)
EOF
}

dataset="nolly_dataset${RETRY:-}"
RUN="${dataset}" .venv/bin/python3 - <<'EOF' || fail "${dataset}" "dataset counts"
import json, os, sys
from datetime import datetime
m = json.load(open("datasets/raw/internal/capture_v2/nolly_48k/manifest.json"))
ok = m["counts"] == {"trainval": 29, "test": 8, "test_seen": 29}
print(f"=== {os.environ['RUN']}: counts {m['counts']}, advance {m['advance_48k']} (median lag {m['median_marker_lag_192k']} at 192 kHz)")
with open("demo/RUNS.jsonl", "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"run_id": os.environ["RUN"], "status": "check",
        "counts": m["counts"], "advance_48k": m["advance_48k"],
        "median_marker_lag_192k": m["median_marker_lag_192k"], "ok": ok,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
sys.exit(0 if ok else 1)
EOF

GPU_FREE_MIB=21000
gate="repro_gate_nolly${RETRY:-}"
wait_gpu
echo "=== ${gate}: start $(date '+%a %d %H:%M')"
.venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${gate}" --no-record \
  --model ssm-wavenet --device vs100r --discretization zoh --honor-optim --deterministic \
  --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5 --cond film --cond-head joint \
  --cond-hidden 32 --head-act leaky --seed 42 --max-steps 300 \
  > "demo/runs/${gate}.log" 2>&1 || fail "${gate}" "gate bench exit $?"
GATE="${gate}" .venv/bin/python3 - <<'EOF' || fail "${gate}" "gate not identical"
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

for arm in C F3 GREY L; do
  smoke="smoke_nolly_${arm}${RETRY:-}"
  wait_gpu
  .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${smoke}" --no-record \
    ${!arm} ${NO} --seed 42 --max-steps 100 > "demo/runs/${smoke}.log" 2>&1 \
    || fail "${smoke}" "smoke bench exit $?"
  printf '{"run_id": "%s", "status": "smoke", "ok": true, "max_steps": 100, "timestamp": "%s", "log": "demo/runs/%s.log"}\n' \
    "${smoke}" "$(date -Iseconds)" "${smoke}" >> demo/RUNS.jsonl
  echo "=== ${smoke}: OK $(date '+%a %d %H:%M')"
done

for seed in 42 43 44; do
  c=$(attempt "nolly_C_s${seed}")
  fresh "${c}"
  run_once "${c}" ${C} ${NO} --seed "${seed}" \
    || { echo "=== ${c} failed, queue stopped"; exit 1; }
  check "${c}" || { echo "=== ${c}: check FAILED, queue stopped"; exit 1; }
  steps=$(halving_steps "${c}")
  [ -n "${steps}" ] || fail "nolly_halvings_s${seed}" "no learning-rate halving in ${c}: grey arms not run"
  for arm in F3 GREY L; do
    run=$(attempt "${PREFIX[${arm}]}_s${seed}")
    fresh "${run}"
    echo "=== ${run}: halvings of ${c} at steps [${steps}]"
    run_once "${run}" ${!arm} ${NO} --seed "${seed}" --lr-halving-steps ${steps} \
      || { echo "=== ${run} failed, queue stopped"; exit 1; }
    check "${run}" "${c}" "${steps}" || { echo "=== ${run}: check FAILED, queue stopped"; exit 1; }
  done
done
echo "=== nolly queue finished $(date '+%a %d %H:%M')"
