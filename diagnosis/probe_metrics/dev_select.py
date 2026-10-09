#!/usr/bin/env python3
"""Development choice of the probe hyper-parameters (diagnosis/probe_metrics/NOTE.md).

Static reference only, no defect read: noise and synchronized sweep, three levels,
seeds 100-102 (estimation) and seed + 50 (reading). For each probe (FIR, Hammerstein
3, Hammerstein 5): taps by capacity at ridge 1e-6, then the largest ridge that keeps
capacity within 1 %. Capacity = median r_ref / RMS(y_ref) on the reading segment.
Writes dev_select.json next to this file. CPU only.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np

from fssr_nam.product import defect_bench as db
from fssr_nam.product.probes import (
    Probe,
    branches,
    fit,
    normal_equations,
    regularized,
    render,
)

HERE = Path(__file__).resolve().parent
FS = 48_000.0
SKIP = round(0.05 * FS)
FAMILIES = ("noise", "sweep")
SEEDS = (100, 101, 102)
TAPS = (16, 32, 64, 128, 256)
RIDGES = (1e-8, 1e-6, 1e-4, 1e-2)
DEGREES = {"fir": 1, "hammerstein_3": 3, "hammerstein_5": 5}


def conditions() -> list[dict]:
    out = []
    for family in FAMILIES:
        for level in db.LEVELS_DB[db.LEVEL_KIND[family]]:
            for seed in SEEDS:
                xa = db.excitation(family, level, 4.0, seed, FS)
                xb = db.excitation(family, level, 4.0, seed + 50, FS)
                out.append(
                    {
                        "family": family,
                        "level_db": level,
                        "seed": seed,
                        "xa": xa,
                        "ya": db.apply(db.STATIC, xa, FS),
                        "xb": xb,
                        "yb": db.apply(db.STATIC, xb, FS),
                    }
                )
    return out


def capacity(probe: Probe, case: dict) -> tuple[float, float]:
    """r_ref / RMS(y_ref) on the reading segment, and the condition number."""
    result = fit(probe, case["xa"], case["ya"])
    q = render(result, case["xb"])
    residual = (case["yb"] - q)[SKIP:]
    ratio = np.sqrt(np.mean(residual**2) / np.mean(case["yb"][SKIP:] ** 2))
    u, _ = branches(case["xa"], probe.degree)
    gram, _ = normal_equations(probe, u, None)
    return float(ratio), float(np.linalg.cond(regularized(probe, gram)))


def summary(probe: Probe, cases: list[dict]) -> dict:
    values = [capacity(probe, case) for case in cases]
    by_seed = {}
    for case, (ratio, _) in zip(cases, values, strict=True):
        by_seed.setdefault((case["family"], case["level_db"]), []).append(ratio)
    return {
        "median_ratio": statistics.median(v[0] for v in values),
        "median_condition": statistics.median(v[1] for v in values),
        "max_seed_sd_of_ratio": max(statistics.stdev(v) for v in by_seed.values()),
    }


def choose(name: str, degree: int, cases: list[dict]) -> dict:
    by_taps = {m: summary(Probe(degree, m, 1e-6), cases) for m in TAPS}
    best = min(v["median_ratio"] for v in by_taps.values())
    taps = min(m for m, v in by_taps.items() if v["median_ratio"] <= 1.10 * best)
    base = Probe(degree, taps, RIDGES[0])
    by_ridge = {r: summary(replace(base, ridge=r), cases) for r in RIDGES}
    first = by_ridge[RIDGES[0]]["median_ratio"]
    ridge = max(r for r, v in by_ridge.items() if v["median_ratio"] <= 1.01 * first)
    print(f"{name}: taps {taps}, ridge {ridge:g}", flush=True)
    return {
        "taps": taps,
        "ridge": ridge,
        "by_taps": {str(k): v for k, v in by_taps.items()},
        "by_ridge": {f"{k:g}": v for k, v in by_ridge.items()},
    }


def main() -> None:
    cases = conditions()
    result = {name: choose(name, degree, cases) for name, degree in DEGREES.items()}
    (HERE / "dev_select.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
