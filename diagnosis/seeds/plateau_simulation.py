#!/usr/bin/env python3
"""Would another plateau threshold fire more consistently? (Read-only, logs only.)

Pilot F tested one threshold, 2e-2, and it neither reduced the dispersion nor spared the
quality. Before spending GPU on another value, this replays the validation curves
already written through a faithful simulation of ReduceLROnPlateau and asks, for each
threshold, how much the epoch of the first halving varies between runs.

The measurement is honest about what it is. A different threshold would change the real
curve, so this cannot predict the outcome of a real run. What it does measure is the
sensitivity of the trigger itself: given curves that differ only by a one-float32-step
perturbation, how differently would the same rule fire? If that dispersion stays large
at every threshold, no fixed threshold can make the schedule reproducible, and only an
adaptive rule is worth GPU time.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "demo/runs"
THRESHOLDS = (1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 5e-2)
PATIENCES = (20, 40)

GROUPS = {
    "enfants f100 (un pas float32)": [
        f"butterfly_ssm_seed42_f100_decide_k{k}" for k in range(5)
    ],
    "graines du banc": [
        f"ssmzoh_guard_seed{s}{r}" for s in (42, 43, 44) for r in ("", "_repeat")
    ],
    "chaine de donnees commune": [f"splitfix_seed{s}" for s in range(42, 48)],
}


def validation(run: str) -> list[float]:
    with (RUNS / f"nablafx_{run}/logs/metrics.csv").open() as stream:
        return [
            float(row["loss/val/tot"])
            for row in csv.DictReader(stream)
            if row.get("loss/val/tot") and row.get("epoch")
        ]


def halvings(curve: list[float], threshold: float, patience: int) -> list[int]:
    """torch.optim.lr_scheduler.ReduceLROnPlateau, mode min, threshold_mode rel."""
    best, bad, out = float("inf"), 0, []
    for epoch, value in enumerate(curve):
        if value < best * (1 - threshold):
            best, bad = value, 0
        else:
            bad += 1
        if bad > patience:
            out.append(epoch)
            bad = 0
    return out


def main() -> None:
    out: dict = {}
    for name, runs in GROUPS.items():
        curves = {r: validation(r) for r in runs}
        shortest = min(len(c) for c in curves.values())
        print(f"\n=== {name} ({len(runs)} runs, {shortest} epoques communes)")
        for patience in PATIENCES:
            print(f"  patience {patience}")
            for threshold in THRESHOLDS:
                firsts, counts = [], []
                for curve in curves.values():
                    events = halvings(curve[:shortest], threshold, patience)
                    firsts.append(events[0] if events else shortest)
                    counts.append(len(events))
                spread = statistics.stdev(firsts) if len(firsts) > 1 else 0.0
                out[f"{name}|{patience}|{threshold}"] = {
                    "first": firsts,
                    "sd_first": spread,
                    "count": counts,
                }
                print(
                    f"    seuil {threshold:.0e} : premieres divisions {firsts}"
                    f"  ecart-type {spread:6.1f}  nombre {counts}"
                )

    (Path(__file__).parent / "plateau_simulation.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
