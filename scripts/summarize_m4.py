#!/usr/bin/env python3
"""Aggregate the fixed M4 matrix and write its required failure autopsy."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/m4_smoke.yaml"
SUMMARY_DIR = ROOT / "experiments/summaries/m4_smoke"
CPU_PATH = ROOT / "experiments/runs/m4_python_cpu_seed0_v2/metrics.json"
REPORT_PATH = ROOT / "reports/M4_CORE.md"
LABELS = {
    "fulltone_full_drive_2": "Fulltone Full Drive 2",
    "electro_harmonix_big_muff": "Electro-Harmonix Big Muff",
}
SLUGS = {
    "fulltone_full_drive_2": "fulltone",
    "electro_harmonix_big_muff": "bigmuff",
}


def run_id(device: str, model: str, seed: int) -> str:
    return f"m4_{SLUGS[device]}_{model.lower()}_seed{seed}_v1"


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = []
    failures = []
    for device in config["devices"]:
        for model in config["models"]:
            for seed in config["seeds"]:
                identifier = run_id(device, model, seed)
                path = ROOT / "experiments/runs" / identifier
                status = json.loads((path / "status.json").read_text(encoding="utf-8"))
                metrics = json.loads(
                    (path / "metrics.json").read_text(encoding="utf-8")
                )
                if status["status"] != "completed":
                    failures.append(identifier)
                rows.append(
                    {
                        "run_id": identifier,
                        "device": device,
                        "model": model,
                        "seed": seed,
                        "status": status["status"],
                        "best_step": metrics["best_step"],
                        "parameters": metrics["parameters"],
                        "latency_samples": metrics["latency_samples"],
                        **metrics["test"],
                    }
                )
    if failures or len(rows) != 24:
        raise RuntimeError(f"M4 matrix is incomplete: {failures}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["device"], row["model"])].append(row)
    metric_names = (
        "esr",
        "mae",
        "gain_error",
        "correlation",
        "mrstft",
        "magnitude_error",
        "log_spectral_distance_db",
        "residual_energy_ratio",
    )
    aggregates = []
    for device in config["devices"]:
        baseline_esr = median([item["esr"] for item in grouped[(device, "B0")]])
        for model in config["models"]:
            items = grouped[(device, model)]
            entry = {
                "device": device,
                "model": model,
                "seeds": len(items),
                "parameters": items[0]["parameters"],
                "latency_samples": items[0]["latency_samples"],
            }
            for metric in metric_names:
                entry[f"median_{metric}"] = median(
                    [float(item[metric]) for item in items]
                )
            entry["relative_esr_vs_b0"] = (
                entry["median_esr"] - baseline_esr
            ) / baseline_esr
            aggregates.append(entry)

    by_key = {(item["device"], item["model"]): item for item in aggregates}
    quality = all(
        min(
            by_key[(device, "S3")]["median_esr"],
            by_key[(device, "S4")]["median_esr"],
        )
        < by_key[(device, "B0")]["median_esr"]
        for device in config["devices"]
    )
    noninferior = all(
        min(
            by_key[(device, "S3")]["median_esr"],
            by_key[(device, "S4")]["median_esr"],
        )
        <= 1.05 * by_key[(device, "B0")]["median_esr"]
        for device in config["devices"]
    )
    antialiasing = False
    cpu = json.loads(CPU_PATH.read_text(encoding="utf-8"))
    cpu64 = {
        code: next(
            item["median_ns_per_sample"]
            for item in result["blocks"]
            if item["block_size"] == 64
        )
        for code, result in cpu["python_streaming"].items()
    }
    a2_cpu64 = next(
        item["median_ns_per_sample"]
        for item in cpu["official_a2_core_cpp_reference"]
        if item["block_size"] == 64
    )
    summary = {
        "schema_version": 1,
        "run_count": len(rows),
        "failed_runs": failures,
        "aggregates": aggregates,
        "gate": {
            "quality": quality,
            "efficiency": noninferior and False,
            "antialiasing": antialiasing,
            "passed": quality or (noninferior and False) or antialiasing,
            "notes": [
                "Efficiency cannot pass because fidelity is not non-inferior.",
                (
                    "FSSR C++ inference is unavailable, so Python/C++ timing is "
                    "diagnostic."
                ),
                "No physical high-rate reference identifies parasite energy for H3.",
            ],
        },
        "cpu_block64": {
            "python_streaming_ns_per_sample": cpu64,
            "official_a2_core_cpp_ns_per_sample": a2_cpu64,
        },
    }
    autopsy = {
        "ordered_checks": [
            {
                "order": 1,
                "name": "alignment",
                "finding": (
                    "Fulltone marker grids are exact. Big Muff keeps the published "
                    "pair because nonlinear correlation is inconsistent. Failure on "
                    "Fulltone "
                    "rules out alignment as the sole cause."
                ),
            },
            {
                "order": 2,
                "name": "gain_and_calibration",
                "finding": (
                    "Median Big Muff gain error is about -0.93 for B2/S3/S4, showing "
                    "collapse toward a low-energy output; A2 is -0.45."
                ),
            },
            {
                "order": 3,
                "name": "data_leakage",
                "finding": (
                    "No path crosses a released source file. Fulltone source "
                    "identities "
                    "are disjoint; Big Muff session identity remains unpublished."
                ),
            },
            {
                "order": 4,
                "name": "context_length",
                "finding": (
                    "Every model receives 6,346 warm-up samples. This covers A2 and "
                    "provides 99 slow-state updates, but FSSR's fast receptive field "
                    "is "
                    "much shorter than A2 and may limit delayed nonlinear behavior."
                ),
            },
            {
                "order": 5,
                "name": "state_stability",
                "finding": (
                    "All outputs and gradients are finite; alternate block differences "
                    "are below 2e-5 for every run."
                ),
            },
            {
                "order": 6,
                "name": "numeric_clipping",
                "finding": (
                    "Prepared inputs and targets contain no samples at or above 0.999 "
                    "and no non-finite values."
                ),
            },
            {
                "order": 7,
                "name": "residual_energy",
                "finding": (
                    "The residual remains negligible on Fulltone and near 0.2% of "
                    "output "
                    "energy on Big Muff; it does not repair the structured core."
                ),
            },
            {
                "order": 8,
                "name": "undertraining",
                "finding": (
                    "Big Muff FSSR validation stays near ESR 1 despite 200 steps, "
                    "consistent with a low-energy local solution."
                ),
            },
            {
                "order": 9,
                "name": "overtraining",
                "finding": (
                    "Checkpoint selection rejects late regressions. Fulltone has a "
                    "source-generalization gap, but Big Muff fails on validation and "
                    "test."
                ),
            },
            {
                "order": 10,
                "name": "budget_allocation",
                "finding": (
                    "S3/S4 have 1,244 parameters versus A2 Full's 12,145. M4 tested an "
                    "efficiency-sized FSSR, not the equal-cost H1 point."
                ),
            },
            {
                "order": 11,
                "name": "metric_listening_conflict",
                "finding": (
                    "Correlation, magnitude error, log spectral distance, and ESR "
                    "agree "
                    "on the ranking; no metric conflict is currently evident."
                ),
            },
            {
                "order": 12,
                "name": "real_cost",
                "finding": (
                    "At block 64, S3/S4 Python require about 21.5/24.4 microseconds "
                    "per sample and are not real-time. A2 Core C++ is 2.86 "
                    "microseconds, but "
                    "cross-engine values are not final H2 evidence."
                ),
            },
        ],
        "primary_diagnosis": [
            "FSSR output-energy collapse on the hard-clipping device.",
            "A tenfold parameter-budget mismatch leaves H1 equal-cost untested.",
            (
                "The short fast receptive field may be marginal for the published "
                "Big Muff pair."
            ),
        ],
        "authorized_next_experiment": (
            "One-seed, two-device S3 diagnostic with residual width 31, matching A2 "
            "parameter count while keeping the architecture and loss fixed. Stop if it "
            "does not materially close the ESR gap."
        ),
    }

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (SUMMARY_DIR / "autopsy.json").write_text(
        json.dumps(autopsy, indent=2) + "\n", encoding="utf-8"
    )
    with (SUMMARY_DIR / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (SUMMARY_DIR / "aggregates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=aggregates[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(aggregates)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for axis, device in zip(axes, config["devices"], strict=True):
        values = [by_key[(device, model)]["median_esr"] for model in config["models"]]
        axis.bar(config["models"], values)
        axis.set_title(LABELS[device])
        axis.set_ylabel("median test ESR")
    figure.tight_layout()
    figure.savefig(SUMMARY_DIR / "esr_by_device.png", dpi=140)
    plt.close(figure)

    table_lines = [
        "| Device | Model | Median ESR | Relative vs B0 | Median gain error |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        table_lines.append(
            f"| {LABELS[item['device']]} | {item['model']} | "
            f"{item['median_esr']:.6f} | "
            f"{100 * item['relative_esr_vs_b0']:+.1f}% | "
            f"{item['median_gain_error']:+.3f} |"
        )
    report = f"""# M4 - Principal Smoke Test

## Outcome

The fixed 24-run matrix completed with three seeds, two physical devices,
four models, and no failed run. **M4 did not pass.** A2 Full has the lowest
median test ESR on both devices; neither FSSR variant is non-inferior, and
S4 does not show a clear advantage over S3.

{chr(10).join(table_lines)}

These are bounded `INTERNAL_DEV` smoke results, not final H1-H3 evidence.

## Gate evaluation

- Quality: failed. FSSR does not beat A2 on either device.
- Efficiency: failed on fidelity before cost can establish non-inferiority.
- Antialiasing: not established. S4 and S3 are nearly tied, and no physical
  high-rate reference identifies parasite energy.

## Cost diagnostic

At block 64 and one CPU thread, the Python streaming paths measure
{cpu64["B2"] / 1000:.2f} µs/sample for B2, {cpu64["S3"] / 1000:.2f} for S3,
and {cpu64["S4"] / 1000:.2f} for S4. The pinned official A2 Core C++
reference is {a2_cpu64 / 1000:.2f} µs/sample. This cross-engine comparison
is diagnostic only; FSSR C++ inference does not yet exist.

## Ordered autopsy

The machine-readable autopsy is
`experiments/summaries/m4_smoke/autopsy.json`. The strongest findings are
output-energy collapse on Big Muff, a tenfold parameter-budget mismatch versus
A2 Full, and a possibly marginal fast receptive field. Alignment uncertainty
cannot explain the Fulltone failure; clipping, non-finite state, leakage across
known source files, and block inconsistency were not observed.

## Bounded recovery decision

Before any maturation matrix, run one seed on both devices with S3 residual
width 31. This matches A2's parameter budget while leaving data, loss, context,
and training budget fixed. Stop and retain the negative-result/benchmark pivot
if it does not materially close the ESR gap.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()
