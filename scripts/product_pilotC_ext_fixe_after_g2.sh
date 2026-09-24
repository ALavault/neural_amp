#!/bin/bash
# Pilot C extension, phase 3, once G2.1 has left the GPU, whether it finished or was
# stopped by its preflight gate. Runs the fixe queue from the runner pinned at the
# commit of the decide arms (2874c6a), so both arms of the extension share their code.
cd /fastdata/lavaulta/neural_amp
until grep -qE "=== G2.1 (finished|stopped)" demo/runs/g2_queue.log; do sleep 300; done
exec bash /fastdata/lavaulta/neural_amp_runner/scripts/product_pilotC_ext_fixe_queue.sh
