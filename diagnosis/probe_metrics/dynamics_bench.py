#!/usr/bin/env python3
"""Bench and pre-registered verdict of the envelope-dynamics score E2 (NOTE.md).

run:     every defect as a variant of the DYNAMIC chain, "steps" excitation, strong
         levels -12/-9/-6 dBFS RMS, seeds 100-102 (implementation check) and 200-202
         (reading); release and attack scores; writes dynamics_bench.csv.
verdict: applies the pre-registered criteria to seeds 200-202; writes
         dynamics_verdict.json.
CPU only.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

from fssr_nam.product import defect_bench as db
from fssr_nam.product.dynamics import dynamic_deviation

HERE = Path(__file__).resolve().parent
FS = 48_000.0
CO_RESPONDERS = {"prefilter_hz", "echo_beyond_memory"}
TEST_SEEDS = (200, 201, 202)


def pairs():
    noisy = replace(db.DYNAMIC, noise_db=-80.0)
    yield "repeat_floor", -80.0, noisy, noisy
    for defect, (_, _, severities) in db.DEFECTS.items():
        for severity in severities:
            yield (defect, severity, *db.variant(defect, severity, base=db.DYNAMIC))


def run() -> None:
    rows = []
    for seed in (100, 101, 102, *TEST_SEEDS):
        for level in db.LEVELS_DB["strong_rms"]:
            x, boundaries = db.steps_signal(level, seed, FS)
            for defect, severity, base, changed in pairs():
                out = dynamic_deviation(
                    x,
                    db.apply(base, x, FS, 1),
                    db.apply(changed, x, FS, 2),
                    FS,
                    boundaries,
                )
                rows.append(
                    {
                        "seed": seed,
                        "level_db": level,
                        "defect": defect,
                        "severity": severity,
                        "release_db": out["release"]["score_db"],
                        "attack_db": out["attack"]["score_db"],
                    }
                )
            print(seed, level, flush=True)
    with (HERE / "dynamics_bench.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def medians(rows) -> dict:
    """(level, defect, severity) -> median over test seeds of (release, attack)."""
    cells = {}
    for r in rows:
        if int(r["seed"]) in TEST_SEEDS:
            key = (float(r["level_db"]), r["defect"], float(r["severity"]))
            cells.setdefault(key, []).append(
                (float(r["release_db"]), float(r["attack_db"]))
            )
    return {
        k: (statistics.median(a for a, _ in v), statistics.median(b for _, b in v))
        for k, v in cells.items()
    }


def per_level(cells: dict, level: float) -> dict:
    release = {s: cells[(level, "release_ms", s)][0] for s in (60.0, 100.0, 200.0)}
    floor = cells[(level, "repeat_floor", -80.0)][0]
    others = {
        d: max(v[0] for (lv, dd, _), v in cells.items() if lv == level and dd == d)
        for (_, d, _) in cells
        if d not in CO_RESPONDERS | {"release_ms", "repeat_floor"}
    }
    worst = max(others, key=others.get)
    return {
        "release": release,
        "floor": floor,
        "monotone": floor < release[60.0] < release[100.0] < release[200.0],
        "sensitive_60ms": release[60.0] >= 3 * floor,
        "worst_other": [worst, others[worst]],
        "specific": release[200.0] >= 3 * others[worst],
        "attack_of_release": cells[(level, "release_ms", 200.0)][1],
        "control": cells[(level, "release_ms", 200.0)][1] <= release[200.0] / 3,
        "co_responders": {
            d: max(v[0] for (lv, dd, _), v in cells.items() if lv == level and dd == d)
            for d in CO_RESPONDERS
        },
    }


def verdict() -> None:
    rows = list(csv.DictReader((HERE / "dynamics_bench.csv").open()))
    cells = medians(rows)
    levels = {lv: per_level(cells, lv) for lv in db.LEVELS_DB["strong_rms"]}
    count = {
        k: sum(v[k] for v in levels.values())
        for k in ("monotone", "sensitive_60ms", "specific", "control")
    }
    result = {
        "levels": {str(k): v for k, v in levels.items()},
        "criteria_met": {k: v >= 2 for k, v in count.items() if k != "control"}
        | {"control": count["control"] == len(levels)},
        "counts": count,
    }
    (HERE / "dynamics_verdict.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["criteria_met"], indent=1))


if __name__ == "__main__":
    {"run": run, "verdict": verdict}[sys.argv[1]]()
