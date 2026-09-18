#!/usr/bin/env python3
"""By what margin is a plateau declared? (Read-only, logs only.)

validation_noise.py ruled out the sampling of the twelve validation segments as the cause
of divergent decisions: the segments are fixed within a run, so their draw cancels in the
plateau detection. What remains is the epoch-to-epoch fluctuation at fixed segments.

ReduceLROnPlateau halves the learning rate after 20 epochs without an improvement of more
than 1e-4 in relative terms over the best value so far. So for each halving, two numbers
decide everything: how close the run came to setting a new best during the twenty epochs
that triggered it, and how large the ordinary epoch-to-epoch fluctuation is. If the first
is smaller than the second, the halving epoch is set by optimisation noise, and two runs
differing by one float32 step have no reason to halve at the same time.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "demo/runs"
BUTTERFLY = ROOT / "demo/butterfly"
PATIENCE = 20
THRESHOLD = 1e-4


def validation(run: str) -> dict[int, float]:
    with (RUNS / f"nablafx_{run}/logs/metrics.csv").open() as stream:
        return {
            int(row["epoch"]): float(row["loss/val/tot"])
            for row in csv.DictReader(stream)
            if row["loss/val/tot"]
        }


def halvings(run: str) -> list[int]:
    lr = json.loads((BUTTERFLY / f"{run}.json").read_text())["lr_by_epoch"]
    epochs = sorted(lr, key=int)
    return [int(b) for a, b in zip(epochs, epochs[1:]) if lr[a] != lr[b]]


def main() -> None:
    out = {}
    for arm in ("decide",):
        for k in range(5):
            run = f"butterfly_ssm_seed42_f100_{arm}_k{k}"
            losses = validation(run)
            epochs = sorted(losses)
            jumps = [
                abs(losses[b] - losses[a]) / losses[a] for a, b in zip(epochs, epochs[1:])
            ]
            fluctuation = statistics.median(jumps)
            rows = []
            for halving in halvings(run)[:3]:
                window = [e for e in epochs if halving - PATIENCE <= e < halving]
                earlier = [losses[e] for e in epochs if e < halving - PATIENCE]
                if not window or not earlier:
                    continue
                best = min(earlier)
                # How close the window came to beating the best, in relative terms.
                approach = (min(losses[e] for e in window) - best) / best
                rows.append({"halving": halving, "closest_approach": approach})
            out[run] = {
                "median_relative_jump": fluctuation,
                "threshold": THRESHOLD,
                "halvings": rows,
            }
            print(
                f"{run[-12:]}: fluctuation mediane {fluctuation:.2e},"
                f" seuil {THRESHOLD:.0e}, rapport {fluctuation / THRESHOLD:.0f}"
            )
            for row in rows:
                print(
                    f"    division a l'epoque {row['halving']:4d} :"
                    f" le meilleur a ete manque de {row['closest_approach']:+.2e}"
                    f" ({row['closest_approach'] / fluctuation:+.2f} fluctuation)"
                )

    (Path(__file__).parent / "plateau_margin.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
