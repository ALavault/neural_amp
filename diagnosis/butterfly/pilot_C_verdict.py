#!/usr/bin/env python3
"""Verdict of pilot C. (Read-only, logs and records only, no GPU.)

Pre-registration: diagnosis/butterfly/pilot_C_common_schedule.md, thresholds written
2026-09-17 and untouched since. Refuses to conclude before the sixteen runs exist.

Three things are printed because the pre-registration asks for them: the ESR of both arms
seed by seed, the standard deviation of the log per arm and their ratio, and the check
that the "fixe" arm halved the learning rate at exactly the eight declared epochs and
nowhere else. That last one is the gate: a fixed schedule that fired anywhere else would
mean the plateau detector was still running.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "demo/nablafx_bench"
RUNS = ROOT / "demo/runs"
SEEDS = list(range(42, 50))
DECLARED = [203, 265, 422, 486, 565, 608, 661, 722]
STEPS_PER_EPOCH = 7
SUPPORTED, REFUTED = 0.50, 0.75


def esr(arm: str, seed: int) -> float | None:
    path = BENCH / f"pilotC_{arm}_seed{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["test_last"]["metric/test/esr"]


def halving_epochs(run: str) -> list[int]:
    """Steps at which the learning rate changed, converted to epochs."""
    with (RUNS / f"nablafx_{run}/logs/metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    lr = [
        (int(r["step"]), float(r["lr-AdamW/pg1"]))
        for r in rows
        if r.get("lr-AdamW/pg1") and r.get("step")
    ]
    return [
        round(step / STEPS_PER_EPOCH)
        for (step, before), (_, after) in zip(lr, lr[1:], strict=False)
        if before != after
    ]


def main() -> None:
    table = {arm: {s: esr(arm, s) for s in SEEDS} for arm in ("decide", "fixe")}
    missing = [
        f"{arm}_seed{s}" for arm in table for s in SEEDS if table[arm][s] is None
    ]
    if missing:
        print(f"{16 - len(missing)}/16 runs; il manque {', '.join(missing)}.")
        return

    print("graine   decide     fixe      ecart en log")
    for s in SEEDS:
        d, f = table["decide"][s], table["fixe"][s]
        print(f"  {s}    {d:.4f}   {f:.4f}    {math.log(f / d):+.3f}")
    spreads = {
        arm: statistics.stdev(math.log(table[arm][s]) for s in SEEDS) for arm in table
    }
    ratio = spreads["fixe"] / spreads["decide"]
    print(
        f"\ns(decide) = {spreads['decide']:.3f}   s(fixe) = {spreads['fixe']:.3f}"
        f"   rapport = {ratio:.2f}"
    )
    print(
        f"precondition du pre-enregistrement, s(decide) >= 0,20 : "
        f"{spreads['decide'] >= 0.20}"
    )

    print("\nporte : le calendrier fixe a-t-il divise aux huit epoques declarees ?")
    gate = True
    for s in SEEDS:
        run = f"pilotC_fixe_seed{s}"
        observed = halving_epochs(run)
        ok = observed == DECLARED
        gate = gate and ok
        print(f"  {run:22s} {observed} {'conforme' if ok else 'DIFFERE'}")

    verdict = (
        "soutenue"
        if ratio <= SUPPORTED
        else "refutee"
        if ratio >= REFUTED
        else "intermediaire, rapportee telle quelle"
    )
    print(f"\nverdict : {verdict} (soutenue <= {SUPPORTED}, refutee >= {REFUTED})")

    (Path(__file__).parent / "pilot_C_results.json").write_text(
        json.dumps(
            {
                "esr": table,
                "s": spreads,
                "ratio": ratio,
                "gate_schedule_exact": gate,
                "verdict": verdict,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
