#!/bin/bash
# G2.1 (demo/param/predictions_poles.md): after the pilot C extension decide queue and
# the S4 redo, preflight both pole arms for 200 steps (memory and time gates), then the
# three new arms at split 925 in the pre-registered order C, D, B. Run from its own
# runner worktree (scripts/product_runner_worktree.sh <commit> <path>).
source "$(dirname "$0")/product_queue_lib.sh"
until grep -q "=== queue finished" demo/runs/pilotC_ext_queue.log; do sleep 300; done
# Flags of param_ssm_wavenet_marshall (arm A); only the conditioning changes.
COMMON="--model ssm-wavenet --device marshall --discretization zoh --honor-optim"
COMMON="${COMMON} --deterministic --seed 42 --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5"
C="--cond poles --cond-head joint --cond-hidden 25"
D="--cond poles --cond-head additive --cond-hidden 7 7 6 6"
B="--cond film --cond-head additive --cond-hidden 8 8 8 8"

preflight() {
  local id=$1
  shift
  wait_gpu
  echo "=== ${id}: preflight start $(date '+%a %d %H:%M')"
  .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${id}" --no-record \
    --max-steps 200 ${COMMON} "$@" > "demo/runs/${id}.log" 2>&1
  echo "=== ${id}: preflight end $(date '+%a %d %H:%M') (exit $?)"
  # Gates: peak <= 21.5 GiB and 15,000 steps in <= 12 h, extrapolated from 200 steps
  # (data loading included, so pessimistic). A crash leaves no record: gate failed.
  .venv/bin/python3 - "${id}" <<'EOF'
import json, re, sys
from datetime import datetime
run = sys.argv[1]
text = open(f"demo/runs/{run}.log").read()
peaks = re.findall(r'"peak_gpu_gib": ([0-9.]+)', text)
minutes = re.findall(r'"minutes_this_session": ([0-9.]+)', text)
peak = float(peaks[-1]) if peaks else None
hours = float(minutes[-1]) * 15_000 / 200 / 60 if minutes else None
ok = peak is not None and hours is not None and peak <= 21.5 and hours <= 12
print(f"=== {run}: peak {peak} GiB, extrapolated {hours} h -> {'PASS' if ok else 'FAIL'}")
with open("demo/RUNS.jsonl", "a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "run_id": run, "status": "preflight", "passed": ok, "peak_gpu_gib": peak,
        "extrapolated_hours": hours, "log": f"demo/runs/{run}.log",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "preregistration": "demo/param/predictions_poles.md",
    }) + "\n")
sys.exit(0 if ok else 1)
EOF
}

preflight g2pre_poles_joint ${C} || { echo "=== G2.1 stopped by the preflight gate"; exit 1; }
preflight g2pre_poles_additive ${D} || { echo "=== G2.1 stopped by the preflight gate"; exit 1; }
run_once g2_poles_joint_s925 ${COMMON} ${C}
run_once g2_poles_additive_s925 ${COMMON} ${D}
run_once g2_film_additive_s925 ${COMMON} ${B}
echo "=== G2.1 finished $(date '+%a %d %H:%M')"
