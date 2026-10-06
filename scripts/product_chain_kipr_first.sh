#!/bin/bash
# 2026-10-06, user's order: the KIPR runs of the two KIPR-inspired tracks first (3.8 GiB
# each, they fit beside the other GPU user), then the train-length queue, then the
# tracks queue, which skips the KIPR runs already done. One queue at a time. A KIPR
# failure stops the chain, so that no run is retried under the same id; the tracks
# queue runs whatever the train-length queue's outcome.
# Detach with: nohup setsid bash <this> >> demo/runs/chain_kipr_first.log 2>&1 < /dev/null &
set -u
LOGS=/fastdata/lavaulta/neural_amp/demo/runs
TRACKS=/fastdata/lavaulta/neural_amp_runner_tracks
LENGTH=/fastdata/lavaulta/neural_amp_runner_len
for tag in odd grey; do
  echo "=== KIPR ${tag}: start $(date '+%a %d %H:%M')"
  bash "${TRACKS}/scripts/product_kipr_queue.sh" "${TRACKS}" "${tag}" \
    >> "${LOGS}/kipr_queue.log" 2>&1 \
    || { echo "=== KIPR ${tag} queue failed, chain stopped $(date '+%a %d %H:%M')"; exit 1; }
  echo "=== KIPR ${tag}: done $(date '+%a %d %H:%M')"
done
echo "=== train-length queue: start $(date '+%a %d %H:%M')"
bash "${LENGTH}/scripts/product_train_length_queue.sh" "${LENGTH}" \
  >> "${LOGS}/train_length_queue.log" 2>&1
echo "=== train-length queue: exit $? $(date '+%a %d %H:%M')"
echo "=== tracks queue: start $(date '+%a %d %H:%M')"
bash "${TRACKS}/scripts/product_kipr_tracks_queue.sh" "${TRACKS}" \
  >> "${LOGS}/kipr_tracks_queue.log" 2>&1
echo "=== tracks queue: exit $? $(date '+%a %d %H:%M')"
