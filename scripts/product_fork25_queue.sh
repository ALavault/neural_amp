#!/bin/bash
# Pilot D (diagnosis/butterfly/pilot_D_fork25.md), one run at a time on the shared GPU.
# The fork at epoch 25 sits after the parent's last polarity flip (epoch 9) and before its
# first learning-rate halving (epoch 113), so the children inherit a complete and common
# guard history while still being early in training.
# 1. The parent again, tagged, saving fork checkpoints at epochs 5, 25 and 100; the ones
#    at 5 and 100 must be identical bit for bit to the parent already run, or the queue
#    stops: without that, the fork 25 checkpoint belongs to no known trajectory.
# 2. Fork 25 "decide" k = 0..4, then "replay" k = 0..4, k = 0 reproducing "decide" k = 0.
set -u
export TMPDIR=/fastdata/lavaulta/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /fastdata/lavaulta/neural_amp

wait_for_gpu() {
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

pilot butterfly_ssm_seed42_parent_f25 parent --tag _f25 --fork-epochs 5 25 100

uv run python - <<'EOF' || { echo "=== fork 25: parent differs from the original, queue stopped"; exit 1; }
import sys
import torch

original = "demo/runs/nablafx_butterfly_ssm_seed42_parent/checkpoints"
again = "demo/runs/nablafx_butterfly_ssm_seed42_parent_f25/checkpoints"
for epoch in (5, 100):
    a = torch.load(f"{original}/fork_epoch{epoch}.ckpt", map_location="cpu", weights_only=False)
    b = torch.load(f"{again}/fork_epoch{epoch}.ckpt", map_location="cpu", weights_only=False)
    same = all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    print(f"=== fork {epoch} checkpoint identical: {same}")
    if not same:
        sys.exit(1)
EOF

# The children read the untagged parent directory, so the new checkpoint goes there.
cp demo/runs/nablafx_butterfly_ssm_seed42_parent_f25/checkpoints/fork_epoch25.ckpt \
   demo/runs/nablafx_butterfly_ssm_seed42_parent/checkpoints/fork_epoch25.ckpt

for k in 0 1 2 3 4; do
  pilot "butterfly_ssm_seed42_f25_decide_k${k}" child --fork 25 --arm decide --k "${k}"
done

pilot butterfly_ssm_seed42_f25_replay_k0 child --fork 25 --arm replay --k 0
if grep -q '"identical_to_decide_k0": true' demo/butterfly/butterfly_ssm_seed42_f25_replay_k0.json; then
  echo "=== fork 25 replay k0 identical to decide k0"
  for k in 1 2 3 4; do
    pilot "butterfly_ssm_seed42_f25_replay_k${k}" child --fork 25 --arm replay --k "${k}"
  done
else
  echo "=== fork 25 replay k0 differs from decide k0, fork 25 replay arm stopped"
fi
echo "=== pilot D finished $(date '+%a %H:%M')"
