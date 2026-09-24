#!/bin/bash
# Paired protocol extension (diagnosis/seeds/paired_protocol.md, section "Élargissement
# aux partitions 744 et 369"): a bitwise gate that the training-order recording changes
# nothing, then both architectures at splits 744 and 369 with the order logged at
# epochs 1, 2 and 50. Flags of the six existing paired runs.
source "$(dirname "$0")/product_queue_lib.sh"
wait_gpu
echo "=== sampler gate: start $(date '+%a %d %H:%M')"
.venv/bin/python3 scripts/product_nablafx_bench.py --run-id repro_gate_sampler_ssm_bigmuff \
  --no-record --model ssm-wavenet --discretization zoh --honor-optim --deterministic \
  --seed 42 --max-steps 200 --log-sampler-epochs 0 1 > demo/runs/repro_gate_sampler_ssm_bigmuff.log 2>&1
.venv/bin/python3 - <<'EOF' || { echo "=== sampler gate FAILED: queue stopped"; exit 1; }
import json, sys
from datetime import datetime
sys.path.insert(0, "diagnosis/seeds")
import repro_gate
r = repro_gate.compare("repro_gate_sampler_ssm_bigmuff", "pilotC_decide_seed42")
print(f"=== sampler gate: {r['values_compared']} values, {r['mismatches']} mismatches, identical {r['identical']}")
with open("demo/RUNS.jsonl", "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"run_id": "repro_gate_sampler_ssm_bigmuff", "status": "gate",
        "reference": "pilotC_decide_seed42", "identical": r["identical"],
        "mismatches": r["mismatches"], "values_compared": r["values_compared"],
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
sys.exit(0 if r["identical"] else 1)
EOF
for split in 744 369; do
  run_once "paired_ssm_wavenet_s${split}" --model ssm-wavenet --discretization zoh \
    --honor-optim --deterministic --seed 42 --split-seed "${split}" --log-sampler-epochs 0 1 49
  run_once "paired_s4_tf_l_16_s${split}" --model s4-tf-l-16 --honor-optim --deterministic \
    --seed 42 --split-seed "${split}" --log-sampler-epochs 0 1 49
done
echo "=== paired extension finished $(date '+%a %d %H:%M')"
