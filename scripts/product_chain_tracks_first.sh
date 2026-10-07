#!/bin/bash
# 2026-10-07, user's request ("reschedule"): the tracks queue (odd gate and grey box VS100R
# runs) before the rest of the train-length runs. The previous chain
# (product_chain_kipr_first.sh) and its train-length queue were stopped at 07:08, while
# vs100r48len20_film_s42 (PID $1) was finishing on its own; this chain waits for it,
# writes the end and check lines the stopped queue would have written, then runs the
# tracks queue, then the remaining train-length runs (seeds 43 and 44). One queue at a time.
# Second use, same day: the tracks queue stopped on the odd smoke (out of GPU memory, fixed
# in 7ca8ced); relaunched with RETRY=_r2 after vs100r48len10_film_s43, the length rest with a
# new gate id. Arguments: <pid> <run id> <arm C reference> <expected train segments>.
# Detach with: nohup setsid bash <this> <pid> <run> <ref> <expected> \
#   >> demo/runs/chain_tracks_first.log 2>&1 < /dev/null &
set -u
LAST=${1:?usage: product_chain_tracks_first.sh <pid> <run id> <reference> <expected>}
RUN=${2:?run id}
REF=${3:?arm C reference}
EXPECTED=${4:?expected train segments}
LOGS=/fastdata/lavaulta/neural_amp/demo/runs
TRACKS=/fastdata/lavaulta/neural_amp_runner_tracks
LENGTH=/fastdata/lavaulta/neural_amp_runner_len
MAIN=/fastdata/lavaulta/neural_amp

echo "=== chain tracks first: waiting for ${RUN} (PID ${LAST}) $(date '+%a %d %H:%M')"
while kill -0 "${LAST}" 2>/dev/null; do sleep 60; done
cd "${LENGTH}"
if [ -f "demo/nablafx_bench/${RUN}.json" ]; then
  echo "=== ${RUN}: end $(date '+%a %d %H:%M') (queue stopped before; finished on its own)" \
    >> "${LOGS}/train_length_queue.log"
  RUN="${RUN}" REF="${REF}" EXPECTED="${EXPECTED}" .venv/bin/python3 - \
    >> "${LOGS}/train_length_queue.log" 2>&1 <<'EOF'
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
  echo "=== ${RUN}: check exit $?"
else
  printf '{"run_id": "%s", "status": "failed", "reason": "no bench record after the run ended (queue stopped before)", "timestamp": "%s", "log": "demo/runs/%s.log"}\n' \
    "${RUN}" "$(date -Iseconds)" "${RUN}" >> "${MAIN}/demo/RUNS.jsonl"
  echo "=== ${RUN}: no bench record, failure written to demo/RUNS.jsonl"
fi

echo "=== tracks queue: start $(date '+%a %d %H:%M')"
RETRY=_r2 bash "${TRACKS}/scripts/product_kipr_tracks_queue.sh" "${TRACKS}" \
  >> "${LOGS}/kipr_tracks_queue.log" 2>&1
echo "=== tracks queue: exit $? $(date '+%a %d %H:%M')"
echo "=== train-length rest queue: start $(date '+%a %d %H:%M')"
GATE_ID=repro_gate_vs100r48_trainlen_r3 bash "${MAIN}/scripts/product_train_length_rest_queue.sh" "${LENGTH}" \
  >> "${LOGS}/train_length_queue.log" 2>&1
echo "=== train-length rest queue: exit $? $(date '+%a %d %H:%M')"
