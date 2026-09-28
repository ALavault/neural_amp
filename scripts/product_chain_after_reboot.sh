#!/bin/bash
# Resumes the chain killed by the 2026-09-28 21:46 reboot (NVIDIA driver 595.84 ->
# 595.91.07). Runs from the G2 runner (a0a4a4f), whose lib cd's there: a bitwise gate of
# the three paths of product_repro_gate_queue.sh under the new driver, stopping at the
# first mismatch; then the F-Fuzz fix with the killed SSM run relaunched once as _r2
# (FIX_ffuzz.md section 5), then the FOH training queue unchanged. Code changed between
# a950cb0 (last s4tf and marshall gates) and a0a4a4f, so a mismatch on those two paths
# does not separate code from driver.
source /fastdata/lavaulta/neural_amp_runner_g2/scripts/product_queue_lib.sh
gate() {
  local run_id=$1 reference=$2
  shift 2
  wait_gpu
  echo "=== ${run_id}: start $(date '+%a %d %H:%M')"
  .venv/bin/python3 scripts/product_nablafx_bench.py --run-id "${run_id}" --no-record "$@" \
    > "demo/runs/${run_id}.log" 2>&1
  GATE="${run_id}" REFERENCE="${reference}" .venv/bin/python3 - <<'EOF' || { echo "=== ${run_id} FAILED: queue stopped"; exit 1; }
import json, os, sys
from datetime import datetime
sys.path.insert(0, "diagnosis/seeds")
import repro_gate
gate, reference = os.environ["GATE"], os.environ["REFERENCE"]
r = repro_gate.compare(gate, reference)
print(f"=== {gate}: {r['values_compared']} values, {r['mismatches']} mismatches, identical {r['identical']}")
with open("demo/RUNS.jsonl", "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"run_id": gate, "status": "gate", "reference": reference,
        "identical": r["identical"], "mismatches": r["mismatches"],
        "values_compared": r["values_compared"], "last_step": r["last_step"],
        "first_mismatches": r["first_mismatches"], "environment": repro_gate.environment(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
sys.exit(0 if r["identical"] else 1)
EOF
}
gate repro_gate_drv59591_ssm_bigmuff pilotC_decide_seed42 --model ssm-wavenet \
  --discretization zoh --honor-optim --deterministic --seed 42 --max-steps 200
gate repro_gate_drv59591_s4tf_bigmuff paired_s4_tf_l_16_s925 --model s4-tf-l-16 \
  --honor-optim --deterministic --seed 42 --split-seed 925 --max-steps 200
gate repro_gate_drv59591_ssm_marshall param_ssm_wavenet_marshall --model ssm-wavenet \
  --device marshall --discretization zoh --honor-optim --deterministic --seed 42 \
  --split-seed 925 --l1-weight 0.5 --mrstft-weight 0.5 --max-steps 300
COMMON="--seed 42 --split-seed 925 --deterministic --l1-weight 0.5 --mrstft-weight 0.5"
COMMON="${COMMON} --device ffuzz --target-advance 4"
run_once param_ssm_wavenet_ffuzz_adv4_r2 ${COMMON} --model ssm-wavenet --discretization zoh --honor-optim
run_once param_s4_ffuzz_adv4 ${COMMON} --model s4-param
echo "=== F-Fuzz fix finished $(date '+%a %d %H:%M')"
bash /fastdata/lavaulta/neural_amp_runner_g2/scripts/product_foh_training_queue.sh
echo "=== chain after reboot finished $(date '+%a %d %H:%M')"
