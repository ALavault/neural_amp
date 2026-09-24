#!/bin/bash
# Native cost of SSM-WaveNet against NAM A2 lite and full, like for like: both builds
# (IEEE -O3 and fast-math), the three engines measured back to back, one JSON line per
# measurement with the build and the machine load. Models: the Big Muff A2 of
# scripts/product_report.py and paired_ssm_wavenet_s925 exported to FOH.
# Usage: product_ssm_cost.sh <out.jsonl>
set -eu
cd "$(dirname "$0")/.."
OUT=$1
WORK=${TMPDIR:-/fastdata/lavaulta/tmp}/ssm_cost
mkdir -p "${WORK}"
.venv/bin/python3 scripts/product_ssm_export.py --run-id paired_ssm_wavenet_s925 \
  --out "${WORK}/ssm.json" > /dev/null
A2=demo/runs/product_a2_electro_harmonix_big_muff_seed0_v2
for build in ieee fastmath; do
  if [ "${build}" = ieee ]; then fast=OFF; flags=""; else fast=ON; flags="-ffast-math"; fi
  cmake -S cpp -B "${WORK}/${build}" -DCMAKE_BUILD_TYPE=Release -DNAM_TOOLS_FAST_MATH=${fast} \
    -DCMAKE_CXX_FLAGS="${flags}" > /dev/null
  cmake --build "${WORK}/${build}" --target nam_benchmark ssm_wavenet_benchmark -j8 > /dev/null
  for block in 32 64 128 256; do
    for model in lite full; do
      "${WORK}/${build}/nam_benchmark" "${A2}/model_${model}.nam" "${block}" 2 5 \
        | tail -1 | sed "s/^{/{\"engine\":\"a2_${model}\",\"build\":\"${build}\",\"load\":\"$(cut -d' ' -f1 /proc/loadavg)\",/" >> "${OUT}"
    done
  done
  "${WORK}/${build}/ssm_wavenet_benchmark" "${WORK}/ssm.json" \
    | sed "s/^{/{\"engine\":\"ssm_wavenet\",\"build\":\"${build}\",\"load\":\"$(cut -d' ' -f1 /proc/loadavg)\",/" >> "${OUT}"
done
