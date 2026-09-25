#!/bin/bash
# After the pilot C extension fixe arms: G2.2, the paired extension, the F-Fuzz latency
# fix, then the optional FOH training, one after the other, each from this runner. A
# queue that stops on a failure does not stop the next one: they are independent.
cd "$(dirname "$0")/.."
until grep -q "=== pilot C extension, fixe arms finished" demo/runs/pilotC_ext_fixe_queue.log; do
  sleep 300
done
for queue in g22 paired_ext ffuzz_fix foh_training; do
  echo "=== chain: ${queue} start $(date '+%a %d %H:%M')"
  bash "scripts/product_${queue}_queue.sh"
  echo "=== chain: ${queue} end $(date '+%a %d %H:%M') (exit $?)"
done
echo "=== chain finished $(date '+%a %d %H:%M')"
