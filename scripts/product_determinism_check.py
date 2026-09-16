#!/usr/bin/env python3
"""Can a Big Muff training run be repeated bit for bit? Prerequisite of the fork pilot.

For each model, with batch size 2 so that it fits next to other GPU jobs:
- two fresh nondeterministic runs (the benchmark as run so far): do they differ?
- two fresh runs with --deterministic: identical weights and Adam moments?
- one deterministic run stopped at step 100, copied twice, both copies resumed to
  step 200: identical to each other? (Forks from a checkpoint need this.)
Every run goes through scripts/product_nablafx_bench.py with --no-record; the
training minutes it prints are kept. Writes diagnosis/butterfly/determinism.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "demo/runs"
OUT = ROOT / "diagnosis/butterfly/determinism.json"
MODELS = {
    "ssm": ["--model", "ssm-wavenet", "--discretization", "zoh", "--honor-optim"],
    "s4": ["--model", "s4-tf-l-16", "--no-polarity-guard"],
}


def run(
    run_id: str,
    model_args: list[str],
    steps: int,
    deterministic: bool,
    resume: bool = False,
) -> dict:
    run_dir = RUNS / f"nablafx_{run_id}"
    if not resume and run_dir.exists():
        shutil.rmtree(run_dir)
    command = [
        sys.executable,
        "scripts/product_nablafx_bench.py",
        "--run-id",
        run_id,
        "--seed",
        "42",
        "--max-steps",
        str(steps),
        "--batch-size",
        "2",
        "--no-record",
        *model_args,
        *(["--deterministic"] if deterministic else []),
        *(["--resume"] if resume else []),
    ]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stdout.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    result = {"run_id": run_id, "returncode": done.returncode}
    start = done.stdout.rfind('{\n  "run_id"')
    if done.returncode == 0 and start >= 0:
        record = json.loads(done.stdout[start:])
        result.update(
            minutes=record["minutes_this_session"], global_step=record["global_step"]
        )
    else:
        result["error"] = (done.stderr or done.stdout).strip().splitlines()[-15:]
    print(json.dumps(result), flush=True)
    return result


def last_checkpoint(run_id: str) -> dict:
    """The latest "last" checkpoint. A resumed copy keeps the copied last.ckpt and
    saves its own as last-v1.ckpt, so the one with the highest step is taken."""
    paths = (RUNS / f"nablafx_{run_id}/checkpoints").glob("last*.ckpt")
    checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    return max(checkpoints, key=lambda c: c["global_step"])


def compare(a: str, b: str) -> dict:
    """Bitwise comparison of weights and Adam moments in two runs' last checkpoints."""
    x, y = last_checkpoint(a), last_checkpoint(b)
    weights_equal = all(
        torch.equal(x["state_dict"][k], y["state_dict"][k]) for k in x["state_dict"]
    )
    max_weight_diff = max(
        float((x["state_dict"][k].double() - y["state_dict"][k].double()).abs().max())
        for k in x["state_dict"]
    )
    moments_equal = all(
        torch.equal(state[name], y["optimizer_states"][0]["state"][index][name])
        for index, state in x["optimizer_states"][0]["state"].items()
        for name in ("exp_avg", "exp_avg_sq")
    )
    return {
        "pair": [a, b],
        "steps": [x["global_step"], y["global_step"]],
        "weights_equal": weights_equal,
        "max_weight_diff": max_weight_diff,
        "adam_moments_equal": moments_equal,
    }


PAIRS = [
    ("nondet_a", "nondet_b"),
    ("det_a", "det_b"),
    ("resume_a", "resume_b"),
    ("det_a", "resume_a"),
]


def main() -> None:
    if sys.argv[1:] == ["--compare-only"]:
        # Recompute the comparisons of an earlier invocation from its checkpoints.
        out = json.loads(OUT.read_text(encoding="utf-8"))
        for name, entry in out.items():
            ok = {r["run_id"] for r in entry["runs"] if r["returncode"] == 0}
            entry["comparisons"] = [
                compare(f"determinism_{name}_{a}", f"determinism_{name}_{b}")
                for a, b in PAIRS
                if {f"determinism_{name}_{a}", f"determinism_{name}_{b}"} <= ok
            ]
            for c in entry["comparisons"]:
                print(json.dumps(c), flush=True)
        OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        return
    out: dict = {}
    for name, model_args in MODELS.items():
        prefix = f"determinism_{name}"
        runs = [
            run(f"{prefix}_nondet_a", model_args, 200, False),
            run(f"{prefix}_nondet_b", model_args, 200, False),
            run(f"{prefix}_det_a", model_args, 200, True),
            run(f"{prefix}_det_b", model_args, 200, True),
            run(f"{prefix}_det_half", model_args, 100, True),
        ]
        if runs[-1]["returncode"] == 0:
            for copy in ("resume_a", "resume_b"):
                target = RUNS / f"nablafx_{prefix}_{copy}"
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(RUNS / f"nablafx_{prefix}_det_half", target)
                runs.append(run(f"{prefix}_{copy}", model_args, 200, True, resume=True))
        ok = {
            r["run_id"].removeprefix(f"{prefix}_") for r in runs if r["returncode"] == 0
        }
        comparisons = [
            compare(f"{prefix}_{a}", f"{prefix}_{b}")
            for a, b in PAIRS
            if a in ok and b in ok
        ]
        for c in comparisons:
            print(json.dumps(c), flush=True)
        out[name] = {"runs": runs, "comparisons": comparisons}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
