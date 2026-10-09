#!/usr/bin/env python3
"""Development of the Volterra-Laguerre and Wiener-Hammerstein probes (NOTE.md, v2).

Same development conditions and capacity criterion as dev_select.py. Also measures
nesting between the Hammerstein-5 probe frozen in v1 and the chosen Volterra-Laguerre
probe: the share of each probe's columns outside the span of the other's, on x_A.
Writes dev_select_v2.json. CPU only.
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from fssr_nam.product.probes import (
    VolterraLaguerre,
    WienerHammerstein,
    branches,
    fit_volterra,
    fit_wh,
    laguerre,
    monomials,
    start_spread,
    volterra_contributions,
    volterra_features,
    wh_render,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dev_select import FS, SKIP, conditions  # noqa: E402

SIZES = (4, 8)
POLES = (0.5, 0.8, 0.9, 0.95)
RIDGES = (1e-8, 1e-6, 1e-4, 1e-2)
NEST_SAMPLES = 32_768


def ratio(y: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - q)[SKIP:] ** 2) / np.mean(y[SKIP:] ** 2)))


def volterra_capacity(probe: VolterraLaguerre, cases: list[dict]) -> float:
    values = []
    for case in cases:
        result = fit_volterra(probe, case["xa"], case["ya"])
        q = sum(volterra_contributions(result, case["xb"]).values())
        values.append(ratio(case["yb"], q))
    return statistics.median(values)


def choose_volterra(order: int, cases: list[dict]) -> dict:
    grid = {
        (size, pole): volterra_capacity(VolterraLaguerre(size, pole, order), cases)
        for size in SIZES
        for pole in POLES
    }
    best = min(grid.values())
    size = min(
        s for s in SIZES if min(v for (k, _), v in grid.items() if k == s) <= 1.1 * best
    )
    pole = min((v, p) for (k, p), v in grid.items() if k == size)[1]
    base = VolterraLaguerre(size, pole, order, RIDGES[0])
    by_ridge = {r: volterra_capacity(replace(base, ridge=r), cases) for r in RIDGES}
    ridge = max(r for r, v in by_ridge.items() if v <= 1.01 * by_ridge[RIDGES[0]])
    print(f"volterra {order}: size {size}, pole {pole}, ridge {ridge:g}", flush=True)
    return {
        "size": size,
        "pole": pole,
        "ridge": ridge,
        "grid": {f"{k[0]}_{k[1]}": v for k, v in grid.items()},
        "by_ridge": {f"{k:g}": v for k, v in by_ridge.items()},
    }


def wiener_hammerstein(cases: list[dict]) -> dict:
    probe = WienerHammerstein(starts=3, iterations=20)
    second = round(FS)
    rows = []
    for case in cases:
        if case["family"] != "noise" or case["level_db"] != -20.0:
            continue
        xa, ya = case["xa"][:second], case["ya"][:second]
        xb, yb = case["xb"][:second], case["yb"][:second]
        result = fit_wh(probe, xa, ya)
        scale = np.sqrt(np.mean(yb[SKIP:] ** 2))
        rows.append(
            {
                "seed": case["seed"],
                "capacity": ratio(yb, wh_render(result.states[0], probe, xb)),
                "start_spread": start_spread(result, xb) / scale,
            }
        )
    capacity = statistics.median(r["capacity"] for r in rows)
    spread = statistics.median(r["start_spread"] for r in rows)
    print(
        f"wiener-hammerstein: capacity {capacity:.3f}, spread {spread:.3f}", flush=True
    )
    return {
        "rows": rows,
        "median_capacity": capacity,
        "median_start_spread": spread,
        "unreliable_before_defects": spread > 0.5 * capacity,
    }


def outside_share(columns: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Per column, the RMS share left after least-squares projection on `basis`."""
    coef = np.linalg.lstsq(basis, columns, rcond=None)[0]
    residual = columns - basis @ coef
    return np.linalg.norm(residual, axis=0) / np.linalg.norm(columns, axis=0)


def nesting(case: dict, chosen: dict) -> dict:
    x = case["xa"][:NEST_SAMPLES]
    lags = (0, 1, 4, 16, 64, 255)
    u, _ = branches(x, 5)
    hammerstein = np.column_stack(
        [np.r_[np.zeros(m), s[: len(s) - m]] for s in u for m in range(256)]
    )
    probe = VolterraLaguerre(chosen["size"], chosen["pole"], 3)
    volterra = volterra_features(
        laguerre(x, probe.size, probe.pole), monomials(probe.size, 3)
    )
    picked = np.column_stack(
        [np.r_[np.zeros(m), s[: len(s) - m]] for s in u for m in lags]
    )
    h_in_v = outside_share(picked, np.hstack((np.ones((len(x), 1)), volterra)))
    v_in_h = outside_share(volterra, np.hstack((np.ones((len(x), 1)), hammerstein)))
    return {
        "hammerstein_columns": [f"x^{k} lag {m}" for k in range(1, 6) for m in lags],
        "hammerstein_outside_volterra": h_in_v.tolist(),
        "volterra_outside_hammerstein_median": float(np.median(v_in_h)),
        "volterra_outside_hammerstein_max": float(np.max(v_in_h)),
    }


def main() -> None:
    cases = conditions()
    result = {f"volterra_laguerre_{o}": choose_volterra(o, cases) for o in (2, 3)}
    result["wiener_hammerstein"] = wiener_hammerstein(cases)
    reference = next(c for c in cases if c["family"] == "noise" and c["seed"] == 100)
    result["nesting"] = nesting(reference, result["volterra_laguerre_3"])
    (HERE / "dev_select_v2.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
