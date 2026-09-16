#!/usr/bin/env python3
"""Evaluate the preregistered M4 spline-grid diagnostic against its thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from summarize_m4_memory import LABELS, evaluate_run, median

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/m4_grid.yaml"
MANIFEST_PATH = ROOT / "datasets/manifests/m4_internal.json"
OUTPUT_DIR = ROOT / "experiments/summaries/m4_grid"
REPORT_PATH = ROOT / "reports/M4_GRID.md"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def decide(rows: list[dict]) -> dict:
    decision = CONFIG["decision"]
    model = decision["primary_model"]
    primary = decision["primary_device"]
    grid = [r for r in rows if r["phase"] == "M4_GRID" and r["model"] == model]
    memory = [r for r in rows if r["phase"] == "M4_MEMORY" and r["model"] == model]
    result: dict = {"primary_model": model, "primary_device": primary}
    fulltone = [r for r in grid if r["device"] == primary]
    if not fulltone:
        result["verdict"] = "no_grid_runs"
        return result
    keys = ("test_esr", "band_error", "train_window_esr", "transfer_loss")
    med = {key: median([r[key] for r in fulltone]) for key in keys}
    reference = [r for r in memory if r["device"] == primary]
    ref = (
        {key: median([r[key] for r in reference]) for key in keys}
        if reference
        else decision["reference_medians"]
    )
    confirmed = decision["confirmed"]
    decrease = ref["train_window_esr"] - med["train_window_esr"]
    if (
        med["train_window_esr"] <= confirmed["train_window_esr_max"]
        and med["test_esr"] <= confirmed["test_esr_max"]
        and med["transfer_loss"] <= confirmed["transfer_loss_max"]
    ):
        verdict = "confirmed"
    elif decrease < decision["refuted"]["train_window_esr_min_decrease"]:
        verdict = "refuted"
    else:
        verdict = "partial"
    result.update(
        {
            "seeds_present": sorted(r["seed"] for r in fulltone),
            "medians": med,
            "reference_medians": ref,
            "train_window_esr_decrease": decrease,
            "verdict": verdict,
        }
    )
    control = decision["regression"]["device"]
    control_new = [r["test_esr"] for r in grid if r["device"] == control]
    control_ref = [r["test_esr"] for r in memory if r["device"] == control]
    if control_new and control_ref:
        increase = median(control_new) - median(control_ref)
        result["regression_control"] = {
            "device": control,
            "median_test_esr_new": median(control_new),
            "median_test_esr_reference": median(control_ref),
            "median_increase": increase,
            "passes": (
                increase <= decision["regression"]["median_test_esr_max_increase"]
            ),
        }
    return result


def default_run_ids() -> list[str]:
    ids = []
    for device in CONFIG["devices"]:
        for model in CONFIG["models"]:
            for seed in CONFIG["seeds"]:
                run_id = (
                    f"m4_grid_{LABELS[device]}_{model.lower()}"
                    f"t{CONFIG['taps']}_seed{seed}_v1"
                )
                if (ROOT / "experiments/runs" / run_id).exists():
                    ids.append(run_id)
        for model in ("S3", "B0"):
            ids.extend(CONFIG["references"][device][model])
    return ids


def report(rows: list[dict], decision: dict) -> str:
    lines = [
        "# M4 spline-grid diagnostic (S3, 33-tap FIRs, knots over +-0.4)",
        "",
        "| Run | Model | Device | Seed | Taps | Grid | Params | Test ESR | "
        "100-300 Hz | Train-window ESR | Transfer | Residual ratio |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["taps"] is None:
            grid = "-"
        else:
            grid = f"+-{r.get('spline_range') or 2.0:g}"
        lines.append(
            f"| {r['run_id']} | {r['model']} | {LABELS[r['device']]} | {r['seed']} | "
            f"{r['taps'] if r['taps'] is not None else '-'} | {grid} | "
            f"{r['parameters']} | "
            f"{r['test_esr']:.4f} | {r['band_error']:.4f} | "
            f"{r['train_window_esr']:.4f} | {r['transfer_loss']:+.4f} | "
            f"{r['residual_energy_ratio']:.2e} |"
        )
    lines += ["", "```json", json.dumps(decision, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [
        evaluate_run(run_id, manifest, device)
        for run_id in (args.run_id or default_run_ids())
    ]
    decision = decide(rows)
    summary = {
        "schema_version": 1,
        "config": CONFIG,
        "rows": rows,
        "decision": decision,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(rows, decision), encoding="utf-8")
    print(report(rows, decision))


if __name__ == "__main__":
    main()
