#!/bin/bash
# Create (or move) the queue runner: a detached worktree of this repository pinned at a
# commit, sharing with the main tree everything that is not code - the venv, the raw
# data, the nablafx and auraloss submodules, and the run outputs, which land in the main
# tree. Records then carry the pinned commit. Usage: product_runner_worktree.sh <commit>
set -eu
MAIN=/fastdata/lavaulta/neural_amp
RUNNER=/fastdata/lavaulta/neural_amp_runner
commit=$1
if [ -d "${RUNNER}" ]; then
  if pgrep -f "${RUNNER}/scripts/product_nablafx_bench.py" > /dev/null; then
    echo "a run is using the runner; refusing to move it"; exit 1
  fi
  git -C "${RUNNER}" checkout -q --detach "${commit}"
else
  git -C "${MAIN}" worktree add -q --detach "${RUNNER}" "${commit}"
fi
cd "${RUNNER}"
for shared in .venv datasets/raw third_party/nablafx third_party/auraloss demo/runs \
    demo/nablafx_bench demo/RUNS.jsonl; do
  if [ ! -L "${shared}" ]; then
    rm -rf "${shared}"
    ln -s "${MAIN}/${shared}" "${shared}"
  fi
done
git -C "${RUNNER}" log --oneline -1
