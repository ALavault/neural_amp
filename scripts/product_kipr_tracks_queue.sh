#!/bin/bash
# The two KIPR-inspired tracks (demo/odd_gate/predictions.md, demo/greybox/predictions.md,
# pre-registered 2026-10-06), from the runner given as $1, whose lib cd's there:
# a bitwise gate of the default path (300 steps of vs100r_gc_film_joint_s925_r2), a
# 100-step smoke of each new VS100R model, the KIPR runs (odd and grey tags, protocol
# r2), then the VS100R runs, each replaying the learning-rate halving steps of the arm-A
# run of the same seed as arm C did, and checked against arm C's validation afterwards.
# A failure is written to demo/RUNS.jsonl and stops the queue. Launch it chained after
# the queue in progress (while kill -0 <pid>), never beside it.
RUNNER=${1:?usage: product_kipr_tracks_queue.sh <runner path>}
source "${RUNNER}/scripts/product_queue_lib.sh"
COMMON="--device vs100r --discretization zoh --honor-optim --deterministic"
COMMON="${COMMON} --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5"
COMMON="${COMMON} --cond film --cond-head joint --cond-hidden 32"
COMMON="${COMMON} --no-early-stopping --fft-size fast --no-tf32"
ODD="--model ssm-wavenet --act-type odd --head-act tanh"
GREY="--model greybox"

fail() {  # fail <run id> <reason>: recorded in demo/RUNS.jsonl, then the queue stops
  printf '{"run_id": "%s", "status": "failed", "reason": "%s", "timestamp": "%s", "log": "demo/runs/%s.log"}\n' \
    "$1" "$2" "$(date -Iseconds)" "$1" >> demo/RUNS.jsonl
  echo "=== $1: $2, queue stopped"
  exit 1
}

# Arm C's halving steps as it recorded them (it replayed its arm-A twin's); none: stop.
c_steps() {
  .venv/bin/python3 -c "import json, sys
steps = json.load(open(f'demo/nablafx_bench/vs100r48fast_film_s{sys.argv[1]}.json'))['lr_halving_steps']
assert steps, 'no halving steps'
print(' '.join(map(str, steps)))" "$1"
}

GPU_FREE_MIB=21000  # after the lib, which sets its own default
# RETRY (e.g. _r2) suffixes the gate and smoke ids of a relaunch, so that no run id is
# reused (2026-10-07: the first odd smoke ran out of GPU memory before the gate
# recomputation fix); the VS100R run ids are unchanged, none of them had started.
gate=repro_gate_vs100r48_kiprtracks${RETRY:-}
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

for arm in odd grey; do
  flags=${ODD}
  [ "${arm}" = grey ] && flags=${GREY}
  smoke="smoke_kiprtracks_${arm}${RETRY:-}"
  wait_gpu
  .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${smoke}" --no-record \
    ${flags} ${COMMON} --seed 42 --max-steps 100 > "demo/runs/${smoke}.log" 2>&1 \
    || fail "${smoke}" "smoke bench exit $?"
  echo "=== ${smoke}: OK $(date '+%a %d %H:%M')"
done

GPU_FREE_MIB=6000
for tag in odd grey; do
  for seed in 42 43 44; do
    for config in sym asym; do
      run="kipr_${config}_s${seed}_${tag}"
      if [ -f "demo/runs/${run}/record.json" ]; then
        echo "=== ${run}: done, skipped"
        continue
      fi
      wait_gpu
      echo "=== ${run}: start $(date '+%a %d %H:%M')"
      .venv/bin/python3 demo/kipr_known_answer/known_answer.py train "${config}" \
        "${seed}" --tag "${tag}" > "demo/runs/${run}.log" 2>&1
      code=$?
      echo "=== ${run}: end $(date '+%a %d %H:%M') (exit ${code})"
      if [ ! -f "demo/runs/${run}/record.json" ]; then
        printf '{"run_id": "%s", "status": "failed", "exit_code": %d, "timestamp": "%s", "log": "demo/runs/%s.log"}\n' \
          "${run}" "${code}" "$(date -Iseconds)" "${run}" >> demo/RUNS.jsonl
        echo "=== ${run}: FAILED, queue stopped"
        exit 1
      fi
    done
  done
done

check() {
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
for seed in 42 43 44; do
  steps=$(c_steps "${seed}") || fail "vs100r48_tracks_s${seed}" "no arm C halving steps"
  for arm in odd grey; do
    flags=${ODD}
    run="vs100r48odd_film_s${seed}"
    if [ "${arm}" = grey ]; then
      flags=${GREY}
      run="vs100r48grey_s${seed}"
    fi
    echo "=== ${run}: halvings of arm C at steps [${steps}]"
    run_once "${run}" ${flags} ${COMMON} --seed "${seed}" --lr-halving-steps ${steps} \
      || { echo "=== ${run} failed, queue stopped"; exit 1; }
    check "${run}" "vs100r48fast_film_s${seed}" \
      || { echo "=== ${run}: check FAILED, queue stopped"; exit 1; }
  done
done
echo "=== KIPR tracks queue finished $(date '+%a %d %H:%M')"
