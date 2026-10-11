#!/bin/bash
# Deeper grey box (demo/greybox_deep/predictions.md, D-0014), from the runner given as
# $1, whose lib cd's there: a bitwise gate of the default path (300 steps of
# vs100r_gc_film_joint_s925_r2, the bench changed since) and, once it exists, of
# vs100r48gbF3_s42 (300 steps, after the driver change of 2026-10-11), a 100-step smoke of each new
# configuration, then the runs, most informative first so that a stop leaves complete
# comparisons: VS100R L and F3 (seeds 42-44, arm C's flags and halving steps), JVM GREY
# and L (partitions of arm A, arm A's schedule), VS100R F1, JVM F3. Each VS100R run is
# checked against arm C's validation afterwards. A failure is written to
# demo/RUNS.jsonl and stops the queue. Launch it chained after the queue in progress
# (while kill -0 <pid>), never beside it.
RUNNER=${1:?usage: product_greybox_deep_queue.sh <runner path>}
source "${RUNNER}/scripts/product_queue_lib.sh"
VS="--device vs100r --discretization zoh --honor-optim --deterministic"
VS="${VS} --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5"
VS="${VS} --cond film --cond-head joint --cond-hidden 32"
VS="${VS} --no-early-stopping --fft-size fast --no-tf32"
# Arm A's JVM schedule (plateau halvings, early stopping allowed, 15 000 steps); the
# grey boxes have no SSM, so --fft-size and --no-tf32 only touch their speed and bits.
JV="--device marshall --honor-optim --deterministic --l1-weight 0.5 --mrstft-weight 0.5"
JV="${JV} --fft-size fast --no-tf32"
L="--model deepgreybox --gb-taps 512"
F1="--model deepgreybox --gb-taps 512 --gb-bias --gb-slow 0.05 0.3"
F3="--model deepgreybox --gb-stages 3 --gb-taps 512 --gb-bias --gb-slow 0.05 0.3"
GREY="--model greybox"
PARTITIONS="925 736 688 744 369"

fail() {  # fail <run id> <reason>: recorded in demo/RUNS.jsonl, then the queue stops
  printf '{"run_id": "%s", "status": "failed", "reason": "%s", "timestamp": "%s", "log": "demo/runs/%s.log"}\n' \
    "$1" "$2" "$(date -Iseconds)" "$1" >> demo/RUNS.jsonl
  echo "=== $1: $2, queue stopped"
  exit 1
}

c_steps() {  # arm C's halving steps as it recorded them; none: stop
  .venv/bin/python3 -c "import json, sys
steps = json.load(open(f'demo/nablafx_bench/vs100r48fast_film_s{sys.argv[1]}.json'))['lr_halving_steps']
assert steps, 'no halving steps'
print(' '.join(map(str, steps)))" "$1"
}

check() {  # VS100R run against arm C's validation split and segment count
  RUN="$1" REF="$2" .venv/bin/python3 - <<'EOF'
import json, os, sys
from datetime import datetime
run, ref = os.environ["RUN"], os.environ["REF"]
r = json.load(open(f"demo/nablafx_bench/{run}.json"))
c = json.load(open(f"demo/nablafx_bench/{ref}.json"))
ok = r["val_indices"] == c["val_indices"] and r["train_segments"] == 1396
print(f"=== {run}: validation as {ref} {r['val_indices'] == c['val_indices']}, train segments {r['train_segments']}, step {r['global_step']}")
with open("demo/RUNS.jsonl", "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"run_id": run, "status": "check", "reference": ref,
        "val_identical": r["val_indices"] == c["val_indices"],
        "train_segments": r["train_segments"], "global_step": r["global_step"], "ok": ok,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
sys.exit(0 if ok else 1)
EOF
}

GPU_FREE_MIB=21000
# Runs killed without a record (reboot of 2026-10-11 03:44): relaunched once as <id>_r2.
RELAUNCHED=" vs100r48gbF3_s43 "
attempt() {  # attempt <run id>: the id to run under
  case "${RELAUNCHED}" in *" $1 "*) echo "$1_r2" ;; *) echo "$1" ;; esac
}

gate() {  # gate <gate id> <reference run> <bench args...>: 300 steps bitwise equal, or stop
  local gate=$1 reference=$2
  shift 2
  wait_gpu
  echo "=== ${gate}: start $(date '+%a %d %H:%M')"
  .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${gate}" --no-record "$@" \
    --max-steps 300 > "demo/runs/${gate}.log" 2>&1 || fail "${gate}" "gate bench exit $?"
  GATE="${gate}" REFERENCE="${reference}" .venv/bin/python3 - <<'EOF' || fail "${gate}" "gate not identical"
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
sys.exit(0 if r["identical"] and r["last_step"] >= 299 else 1)
EOF
}

gate "repro_gate_vs100r48_greybox_deep${RETRY:-}" vs100r_gc_film_joint_s925_r2 \
  --model ssm-wavenet --device vs100r --discretization zoh --honor-optim --deterministic \
  --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5 --cond film --cond-head joint \
  --cond-hidden 32 --head-act leaky --seed 42
# The deep grey box itself, once runs trained before a driver change exist.
if [ -f demo/nablafx_bench/vs100r48gbF3_s42.json ]; then
  gate "repro_gate_vs100r48gbF3${RETRY:-}" vs100r48gbF3_s42 ${F3} ${VS} --seed 42 \
    --lr-halving-steps $(c_steps 42)
fi

for arm in L F1 F3; do
  smoke="smoke_greybox_deep_${arm}${RETRY:-}"
  wait_gpu
  .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${smoke}" --no-record \
    ${!arm} ${VS} --seed 42 --max-steps 100 > "demo/runs/${smoke}.log" 2>&1 \
    || fail "${smoke}" "smoke bench exit $?"
  echo "=== ${smoke}: OK $(date '+%a %d %H:%M')"
done

vs100r() {  # vs100r <arm>: the three seeds
  local arm=$1 seed steps run
  for seed in 42 43 44; do
    steps=$(c_steps "${seed}") || fail "vs100r48gb${arm}_s${seed}" "no arm C halving steps"
    run=$(attempt "vs100r48gb${arm}_s${seed}")
    echo "=== ${run}: halvings of arm C at steps [${steps}]"
    run_once "${run}" ${!arm} ${VS} --seed "${seed}" --lr-halving-steps ${steps} \
      || { echo "=== ${run} failed, queue stopped"; exit 1; }
    check "${run}" "vs100r48fast_film_s${seed}" \
      || { echo "=== ${run}: check FAILED, queue stopped"; exit 1; }
  done
}

jvm() {  # jvm <arm> <run prefix>: the five partitions of arm A, seed 42
  local arm=$1 prefix=$2 p run
  for p in ${PARTITIONS}; do
    run=$(attempt "${prefix}_s${p}")
    run_once "${run}" ${!arm} ${JV} --seed 42 --split-seed "${p}" \
      || { echo "=== ${run} failed, queue stopped"; exit 1; }
  done
}

vs100r L
vs100r F3
jvm GREY jvm_grey
jvm L jvm_gbL
vs100r F1
jvm F3 jvm_gbF3
echo "=== greybox deep queue finished $(date '+%a %d %H:%M')"
