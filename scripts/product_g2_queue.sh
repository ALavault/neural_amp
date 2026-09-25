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
  # A preflight that passed and was recorded is not redone after a relaunch of the queue.
  if grep -q "\"run_id\": \"${id}\", \"status\": \"preflight\", \"passed\": true" \
      demo/RUNS.jsonl && [ -f "demo/runs/${id}.reserved_mib" ]; then
    echo "=== ${id}: preflight already passed, not redone"
    return 0
  fi
  # Almost the whole idle GPU (23 000 MiB, 22.5 GiB; 23.6 GiB free when idle): OOM is then
  # the model's, not the contention's (ollama, RACLETTE).
  GPU_FREE_MIB=23000 wait_gpu
  echo "=== ${id}: preflight start $(date '+%a %d %H:%M')"
  .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${id}" --no-record \
    --max-steps 200 ${COMMON} "$@" > "demo/runs/${id}.log" 2>&1
  local code=$?
  echo "=== ${id}: preflight end $(date '+%a %d %H:%M') (exit ${code})"
  # Gates: reserved peak (allocator cache included) <= 21.5 GiB and 15,000 steps in
  # <= 12 h, extrapolated from 200 steps (data loading included, so pessimistic). A
  # crash leaves no record: gate failed. Writes the reserved peak for the next wait.
  .venv/bin/python3 - "${id}" "${code}" <<'EOF'
import json, re, sys
from datetime import datetime
run, code = sys.argv[1], int(sys.argv[2])
text = open(f"demo/runs/{run}.log").read()
oom = "out of memory" in text.lower()
peaks = re.findall(r'"peak_gpu_reserved_gib": ([0-9.]+)', text)
minutes = re.findall(r'"minutes_this_session": ([0-9.]+)', text)
peak = float(peaks[-1]) if peaks else None
hours = float(minutes[-1]) * 15_000 / 200 / 60 if minutes else None
ok = (
    code == 0 and not oom and peak is not None and hours is not None
    and peak <= 21.5 and hours <= 12
)
cause = (
    "out of memory with 22.5 GiB free" if oom
    else f"exit {code}" if code != 0
    else "no record" if peak is None
    else "measured"
)
print(f"=== {run}: reserved peak {peak} GiB, extrapolated {hours} h ({cause}) -> {'PASS' if ok else 'FAIL'}")
if peak is not None:
    open(f"demo/runs/{run}.reserved_mib", "w").write(str(int(peak * 1024)))
with open("demo/RUNS.jsonl", "a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "run_id": run, "status": "preflight", "passed": ok, "cause": cause,
        "peak_gpu_reserved_gib": peak,
        "extrapolated_hours": hours, "log": f"demo/runs/{run}.log",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "preregistration": "demo/param/predictions_poles.md",
    }) + "\n")
sys.exit(0 if ok else 1)
EOF
}

preflight g2pre_poles_joint ${C} || { echo "=== G2.1 stopped by the preflight gate"; exit 1; }
preflight g2pre_poles_additive ${D} || { echo "=== G2.1 stopped by the preflight gate"; exit 1; }
# Each run then waits for the larger reserved peak plus 1 GiB of CUDA context free.
# One value per line before taking the maximum: the files have no trailing newline, and
# a plain cat glued 13701 and 13701 into one number on 2026-09-25 (the queue then waited
# for 1.37e9 MiB and was stopped by hand).
reserved=$(for f in demo/runs/g2pre_poles_joint.reserved_mib \
    demo/runs/g2pre_poles_additive.reserved_mib; do cat "${f}"; echo; done | sort -n | tail -1)
GPU_FREE_MIB=$(( reserved + 1024 > 18500 ? reserved + 1024 : 18500 ))
if [ "${GPU_FREE_MIB}" -gt 23500 ]; then
  echo "=== G2.1 stopped: waiting for ${GPU_FREE_MIB} MiB free would never end"; exit 1
fi
echo "=== G2.1 runs wait for ${GPU_FREE_MIB} MiB free"
# A failed run stops G2.1: it is recorded by run_once, and relaunched by hand as _r2.
run_once g2_poles_joint_s925 ${COMMON} ${C} || { echo "=== G2.1 stopped: C failed"; exit 1; }
run_once g2_poles_additive_s925 ${COMMON} ${D} || { echo "=== G2.1 stopped: D failed"; exit 1; }
run_once g2_film_additive_s925 ${COMMON} ${B} || { echo "=== G2.1 stopped: B failed"; exit 1; }
echo "=== G2.1 finished $(date '+%a %d %H:%M')"
