#!/usr/bin/env python3
"""Verdict of pilot G. (Read-only, CPU, no GPU.)

Refuses to run before the eight "decide" runs of pilot C exist. The pre-registration
in diagnosis/seeds/pilot_G_split_property.md names eight seeds, and computing the
correlation on six would mean looking early at the number it exists to protect.

Primary property, named in advance: the minimum RMS of the twelve validation segments.
Everything else is exploratory and reported as such, with the reminder that across seven
properties the strongest exceeds 0.7 about half the time by chance alone.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [str(s) for s in range(42, 50)]
PRIMARY = "val_rms_min"
SUPPORTED, REFUTED = 0.74, 0.50


def main() -> None:
    properties = json.loads((Path(__file__).parent / "split_property.json").read_text())
    esr = {}
    for seed in SEEDS:
        path = ROOT / f"demo/nablafx_bench/pilotC_decide_seed{seed}.json"
        if not path.exists():
            print(f"graine {seed} sans run : le verdict attend les huit.")
            return
        esr[seed] = json.loads(path.read_text())["test_last"]["metric/test/esr"]

    outcome = [math.log(esr[s]) for s in SEEDS]
    rows = {}
    for name in sorted(k for k in properties[SEEDS[0]] if k != "val_indices"):
        if not all(name in properties[s] for s in SEEDS):
            continue
        values = [properties[s][name] for s in SEEDS]
        if len(set(values)) < 3:
            rows[name] = {"rho": None, "note": "moins de trois valeurs distinctes"}
            continue
        result = stats.spearmanr(values, outcome)
        rows[name] = {"rho": float(result.statistic), "p": float(result.pvalue)}

    primary = rows.get(PRIMARY, {})
    rho = primary.get("rho")
    verdict = (
        "porte ouverte, propriete degeneree"
        if rho is None
        else "H2 soutenue"
        if abs(rho) >= SUPPORTED
        else "H2 refutee"
        if abs(rho) < REFUTED
        else "indecidable"
    )

    print("graine  ESR      " + PRIMARY)
    for seed in SEEDS:
        print(f"  {seed}   {esr[seed]:.4f}   {properties[seed][PRIMARY]:.5f}")
    print()
    for name, row in rows.items():
        mark = " <- primaire" if name == PRIMARY else ""
        if row.get("rho") is None:
            print(f"  {name:22s} {row['note']}{mark}")
        else:
            print(f"  {name:22s} rho = {row['rho']:+.2f}  p = {row['p']:.3f}{mark}")
    print(f"\nverdict sur la propriete primaire : {verdict}")

    (Path(__file__).parent / "split_property_verdict.json").write_text(
        json.dumps({"esr": esr, "correlations": rows, "verdict": verdict}, indent=1)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
