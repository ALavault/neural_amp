#!/usr/bin/env python3
"""Verdict of pilot A against the predictions in diagnosis/butterfly/pilot_A.md.

Read-only, CPU. For each fork and arm: test ESR of the last checkpoint of k = 0..4,
s = standard deviation of log ESR over the five runs, deltas against k = 0, the
learning-rate halvings and final step of each run, and the ESR between each child's
output and the k = 0 output on the test input. Writes
diagnosis/butterfly/pilot_A_results.json.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import product_nablafx_bench as bench  # noqa: E402
import torch  # noqa: E402

RECORDS = ROOT / "demo/butterfly"
OUT = ROOT / "diagnosis/butterfly/pilot_A_results.json"
ESR = "metric/test/esr"


def run_id(fork: int, arm: str, k: int) -> str:
    return f"butterfly_ssm_seed42_f{fork}_{arm}_k{k}"


def outputs(name: str, inputs: torch.Tensor) -> torch.Tensor:
    processor = bench.build_processor(
        SimpleNamespace(
            model="ssm-wavenet",
            num_blocks=8,
            channels=16,
            state_dim=4,
            output_act="tanh",
            discretization="zoh",
        )
    )
    state = torch.load(
        bench.RUNS_DIR / f"nablafx_{name}/checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    processor.load_state_dict(
        {
            key.removeprefix("model.processor."): value
            for key, value in state.items()
            if key.startswith("model.processor.")
        }
    )
    processor.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs), 8):
            processor.reset_states()
            predictions.append(processor(inputs[start : start + 8]))
    return torch.cat(predictions)


def halvings(lr_by_epoch: dict[str, float]) -> list[int]:
    epochs = sorted(lr_by_epoch, key=int)
    return [
        int(b)
        for a, b in itertools.pairwise(epochs)
        if lr_by_epoch[b] != lr_by_epoch[a]
    ]


def main() -> None:
    torch.set_num_threads(16)
    data = bench.data_module("test")
    data.setup("test")
    inputs = torch.stack([x for x, _ in data.test_dataset])
    out: dict = {"groups": {}}
    for fork, arm in ((100, "decide"), (100, "replay"), (5, "decide")):
        names = [run_id(fork, arm, k) for k in range(5)]
        if not all((RECORDS / f"{n}.json").exists() for n in names):
            continue
        records = [json.loads((RECORDS / f"{n}.json").read_text()) for n in names]
        log_esr = [math.log(r["test_last"][ESR]) for r in records]
        reference = outputs(names[0], inputs)
        output_esr = [
            float(
                (outputs(n, inputs) - reference).pow(2).sum() / reference.pow(2).sum()
            )
            for n in names[1:]
        ]
        group = {
            "test_esr": [r["test_last"][ESR] for r in records],
            "s": statistics.stdev(log_esr),
            "delta": [v - log_esr[0] for v in log_esr[1:]],
            "output_esr_vs_k0": output_esr,
            "global_step": [r["global_step"] for r in records],
            "halvings": [halvings(r["lr_by_epoch"]) for r in records],
            "nudge": [r["nudge"] for r in records],
            "polarity_flips": [r["polarity_flips"] for r in records],
        }
        if arm == "replay":
            group["identical_to_decide_k0"] = records[0].get("identical_to_decide_k0")
        out["groups"][f"fork{fork}_{arm}"] = group
        print(
            f"fork {fork:3d} {arm:6s} ESR {[round(v, 4) for v in group['test_esr']]}"
            f" s {group['s']:.3f} delta {[round(d, 3) for d in group['delta']]}"
            f" steps {group['global_step']}"
        )
        print(f"   output ESR vs k0 {[f'{v:.2e}' for v in output_esr]}")
        print(f"   halvings {group['halvings']}")

    groups = out["groups"]
    verdict = {}
    if "fork100_replay" in groups:
        verdict["replay_valid"] = groups["fork100_replay"]["identical_to_decide_k0"]
    if "fork5_decide" in groups:
        verdict["P0_positive_control"] = groups["fork5_decide"]["s"] >= 0.10
    if {"fork100_decide", "fork100_replay"} <= groups.keys():
        decide, replay = groups["fork100_decide"], groups["fork100_replay"]
        mean_decide = statistics.fmean(abs(d) for d in decide["delta"])
        mean_replay = statistics.fmean(abs(d) for d in replay["delta"])
        ratio = mean_decide / mean_replay if mean_replay > 0 else math.inf
        verdict.update(
            s_decide=decide["s"],
            s_replay=replay["s"],
            mean_abs_delta_ratio=ratio,
            A_supported=decide["s"] >= 0.10
            and replay["s"] <= decide["s"] / 2
            and ratio >= 2,
            A_refuted_dynamics=replay["s"] >= 0.10 and ratio < 2,
            A_refuted_stability=decide["s"] < 0.10 and replay["s"] < 0.10,
        )
    out["verdict"] = verdict
    print(json.dumps(verdict, indent=1))
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
