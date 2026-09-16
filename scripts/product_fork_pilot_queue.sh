#!/bin/bash
# Pilot A (diagnosis/butterfly/pilot_A.md), one run at a time on the shared GPU. A run
# whose record exists in demo/butterfly is skipped, so the queue can be restarted.
# 1. The parent to epoch 100, then its twin: the fork checkpoints must be identical bit
#    for bit (determinism at batch size 16), or the queue stops.
# 2. Fork 100, k = 0 "decide", then k = 0 "replay", which must reproduce it bit for bit.
# 3. Fork 100, k = 1..4 in both arms; fork 5, k = 0..4 "decide".
# 4. The S4-TF-L-16 same-seed repeats of diagnosis/seeds/hypotheses_nested.md.
set -u
export TMPDIR=/fastdata/lavaulta/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /fastdata/lavaulta/neural_amp

wait_for_gpu() {
  # ollama loads models of up to 21 GiB on demand and unloads them after 5 idle minutes.
  local free_minutes=0
  while [ "${free_minutes}" -lt 6 ]; do
    if [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 18500 ]; then
      free_minutes=$((free_minutes + 1))
    else
      free_minutes=0
    fi
    sleep 60
  done
}

pilot() {
  local name=$1
  shift
  for attempt in 1 2 3; do
    if [ -f "demo/butterfly/${name}.json" ]; then
      echo "=== ${name}: done, skipped"
      return
    fi
    wait_for_gpu
    echo "=== ${name}: start $(date '+%a %H:%M') (attempt ${attempt})"
    uv run python scripts/product_fork_pilot.py "$@" 2>&1 \
      | grep -vE "warn\(msg\)|UserWarning|warnings.warn|is differentiable|^\s*$"
    echo "=== ${name}: end $(date '+%a %H:%M') (exit ${PIPESTATUS[0]})"
  done
  [ -f "demo/butterfly/${name}.json" ] || { echo "=== ${name}: failed 3 times, queue stopped"; exit 1; }
}

pilot butterfly_ssm_seed42_parent parent
pilot butterfly_ssm_seed42_parent_twin parent --tag _twin
uv run python - <<'EOF' || { echo "=== parent twins differ, queue stopped"; exit 1; }
import sys
import torch
runs = "demo/runs/nablafx_butterfly_ssm_seed42_parent{}/checkpoints/fork_epoch{}.ckpt"
for epoch in (5, 100):
    a, b = (torch.load(runs.format(tag, epoch), map_location="cpu", weights_only=False)
            for tag in ("", "_twin"))
    same = all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    same &= all(torch.equal(s[m], b["optimizer_states"][0]["state"][i][m])
                for i, s in a["optimizer_states"][0]["state"].items()
                for m in ("exp_avg", "exp_avg_sq"))
    print(f"=== parent twins, fork epoch {epoch}: identical {same}")
    if not same:
        sys.exit(1)
EOF

pilot butterfly_ssm_seed42_f100_decide_k0 child --fork 100 --arm decide --k 0
pilot butterfly_ssm_seed42_f100_replay_k0 child --fork 100 --arm replay --k 0
grep -q '"identical_to_decide_k0": true' demo/butterfly/butterfly_ssm_seed42_f100_replay_k0.json \
  || { echo "=== replay k0 differs from decide k0, queue stopped"; exit 1; }
echo "=== replay k0 identical to decide k0"

for k in 1 2 3 4; do
  pilot "butterfly_ssm_seed42_f100_decide_k${k}" child --fork 100 --arm decide --k "${k}"
  pilot "butterfly_ssm_seed42_f100_replay_k${k}" child --fork 100 --arm replay --k "${k}"
done
for k in 0 1 2 3 4; do
  pilot "butterfly_ssm_seed42_f5_decide_k${k}" child --fork 5 --arm decide --k "${k}"
done
echo "=== pilot finished $(date '+%a %H:%M')"

bash scripts/product_nablafx_queue.sh
