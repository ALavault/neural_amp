#!/bin/bash
# Create (or move) the queue runner: a detached worktree of this repository pinned at a
# commit, sharing with the main tree everything that is not code - the venv, the raw
# data, the nablafx and auraloss submodules, and the run outputs, which land in the main
# tree. Records then carry the pinned commit.
# Usage: product_runner_worktree.sh <commit> [runner path]. One runner per concurrent
# queue: moving a runner changes the code of the queue still running from it.
set -eu
MAIN=/fastdata/lavaulta/neural_amp
commit=$1
RUNNER=${2:-/fastdata/lavaulta/neural_amp_runner}
if [ -d "${RUNNER}" ]; then
  # Queues are launched by absolute path; their runs use a relative one after cd, so
  # the guard looks for the queue script, waiting or running.
  if pgrep -f "${RUNNER}/scripts/" > /dev/null; then
    echo "a queue is running from the runner; stop it before moving it"; exit 1
  fi
  # Unlink the shared paths first (the links, never their targets), so that checkout
  # can restore the runner's own tracked copies without going through a link into the
  # main tree; they are linked again below.
  for shared in .venv datasets/raw third_party/nablafx third_party/auraloss demo/runs \
      demo/nablafx_bench demo/RUNS.jsonl; do
    if [ -L "${RUNNER}/${shared}" ]; then rm -- "${RUNNER}/${shared}"; fi
  done
  git -C "${RUNNER}" checkout -q -f --detach "${commit}"
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
