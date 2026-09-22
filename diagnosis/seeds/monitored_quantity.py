#!/usr/bin/env python3
"""Which quantity should ReduceLROnPlateau watch? (Read-only, logs only, no GPU.)

Tests a prediction inscribed in quiet_share_exploration.md before this ran: monitoring
metric/val/esr instead of loss/val/tot should fire EARLIER and MORE erratically, the
validation ESR fluctuating about ten times as much as the loss. Cheap to check, and it
closes the cheapest-looking way out of the plateau problem if it holds.

The control comes first and it is what makes the rest trustworthy: replayed on the
quantity the protocol actually watches, the simulation must reproduce the real halving
epochs of the eight runs. It does, to the epoch, eight times out of eight.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

from plateau_simulation import halvings

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(42, 50))
THRESHOLD, PATIENCE = 1e-4, 20
WATCHED = ("loss/val/tot", "loss/val/mrstft", "loss/val/l1", "metric/val/esr")
# The real epochs, read from the lr column of each run's metrics.csv.
OBSERVED = {42: 285, 43: 252, 44: 161, 45: 234, 46: 255, 47: 164, 48: 167, 49: 141}


def series(seed: int, column: str) -> list[float]:
    path = ROOT / f"demo/runs/nablafx_pilotC_decide_seed{seed}/logs/metrics.csv"
    with path.open() as stream:
        return [
            float(row[column])
            for row in csv.DictReader(stream)
            if row.get(column) and row.get("epoch")
        ]


def first_halvings(column: str) -> list[int]:
    out = []
    for seed in SEEDS:
        fired = halvings(series(seed, column), THRESHOLD, PATIENCE)
        out.append(fired[0] if fired else -1)
    return out


def main() -> None:
    simulated = first_halvings("loss/val/tot")
    exact = sum(s == OBSERVED[seed] for seed, s in zip(SEEDS, simulated, strict=True))
    print("Controle : la simulation reproduit-elle les vraies dates de division ?")
    print(f"  reelles  {[OBSERVED[s] for s in SEEDS]}")
    print(f"  simulees {simulated}")
    print(f"  exactes a l'epoque pres : {exact}/8\n")

    print("Prediction inscrite : surveiller l'ESR declenche plus tot,")
    print("et plus erratiquement.")
    print(f"{'surveillee':20s} {'mediane':>8} {'ecart-type':>11} {'etendue':>14}")
    for column in WATCHED:
        firsts = first_halvings(column)
        print(
            f"{column:20s} {statistics.median(firsts):8.0f}"
            f" {statistics.stdev(firsts):11.1f}"
            f" {f'{min(firsts)} a {max(firsts)}':>14}"
        )


if __name__ == "__main__":
    sys.exit(main())
