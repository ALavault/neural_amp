#!/usr/bin/env python3
"""Post-hoc exploration of the quiet-share lead. (Read-only, CPU, no GPU.)

Reproduces every number in quiet_share_exploration.md. This is NOT a pre-registered
test: pilot G designated the lead without concluding, and about fifteen associations
were examined here on eight points. No verdict label is printed, on purpose.

Two measurement rules keep it honest. Every curve statistic is computed on the FIXED
window 20-140 epochs, common to the eight runs - the earliest halving falls at 141, so
no window depends on the outcome it serves to explain. And the one statistic that did
depend on a variable-length window (the "relative slope before the first halving") is
absent: a short window is dominated by the steep initial descent, so any run halving
early looked mechanically steeper. It was an artefact, not an observation.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(42, 50))
LO, HI = 20, 140
STEPS_PER_EPOCH = 7
TESTED_PROPERTIES = 6  # the six of the seven declared that had enough distinct values


def curve(seed: int) -> dict[str, np.ndarray]:
    """Validation series of a decide run, as (epoch, value) arrays."""
    path = ROOT / f"demo/runs/nablafx_pilotC_decide_seed{seed}/logs/metrics.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    keep = ("loss/val/tot", "loss/val/l1", "loss/val/mrstft", "metric/val/esr")
    series = {
        name: np.array(
            [(int(r["epoch"]), float(r[name])) for r in rows if r.get(name)]
        )
        for name in keep
    }
    lr = [
        (int(r["step"]), float(r["lr-AdamW/pg1"]))
        for r in rows
        if r.get("lr-AdamW/pg1") and r.get("step")
    ]
    changes = [step for (step, a), (_, b) in itertools.pairwise(lr) if a != b]
    series["first_halving"] = np.array([round(changes[0] / STEPS_PER_EPOCH)])
    return series


def window(arr: np.ndarray) -> np.ndarray:
    return arr[(arr[:, 0] >= LO) & (arr[:, 0] <= HI)]


def route_stats(series: dict[str, np.ndarray]) -> dict[str, float]:
    """The four candidate routes, on the fixed window."""
    out = {}
    for name in ("loss/val/tot", "loss/val/l1", "loss/val/mrstft", "metric/val/esr"):
        w = window(series[name])
        trend = np.polyval(np.polyfit(w[:, 0], w[:, 1], 1), w[:, 0])
        out[f"cv({name})"] = float((w[:, 1] - trend).std() / w[:, 1].mean())
    w = window(series["loss/val/tot"])
    slope, _ = np.polyfit(w[:, 0], w[:, 1], 1)
    level = w[:, 1].mean()
    out["level"] = float(level)
    # Drift over one patience window, measured in units of the fluctuation: the trigger
    # is a record-setting process, so this is the quantity that should govern its date.
    out["drift_over_noise"] = float(
        (-slope * 20 / level) / out["cv(loss/val/tot)"]
    )
    mrstft = window(series["loss/val/mrstft"])[:, 1].mean()
    out["mrstft_share"] = float(0.1 * mrstft / level)
    return out


def exact_permutation_p(x: list[float], y: list[float]) -> float:
    """Two-sided p by exhaustive permutation. n = 8 means 40320 orderings."""
    observed = abs(stats.spearmanr(x, y).statistic)
    hits = sum(
        abs(stats.spearmanr(x, perm).statistic) >= observed - 1e-12
        for perm in itertools.permutations(y)
    )
    return hits / math.factorial(len(y))


def partial(x: list[float], y: list[float], z: list[float]) -> float:
    """Spearman partial correlation of x and y given z, through ranks."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    rxy, rxz, ryz = (
        stats.pearsonr(a, b)[0] for a, b in ((rx, ry), (rx, rz), (ry, rz))
    )
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz**2) * (1 - ryz**2))


def show(label: str, x: list[float], y: list[float]) -> None:
    r = stats.spearmanr(x, y)
    print(f"  {label:44s} rho = {r.statistic:+.3f}  p = {r.pvalue:.4f}")


def main() -> None:
    props = json.loads((Path(__file__).parent / "split_property.json").read_text())
    quiet = [props[str(s)]["val_quiet_share_mean"] for s in SEEDS]
    esr, first, routes = [], [], {}
    for seed in SEEDS:
        record = ROOT / f"demo/nablafx_bench/pilotC_decide_seed{seed}.json"
        test = json.loads(record.read_text())["test_last"]
        esr.append(math.log(test["metric/test/esr"]))
        series = curve(seed)
        first.append(int(series["first_halving"][0]))
        routes[seed] = route_stats(series)

    print("La piste designee par le pilote G, et son association la plus forte :")
    p_exact = exact_permutation_p(quiet, esr)
    show("part calme -> log ESR", quiet, esr)
    print(
        f"  {'':44s} p exact {p_exact:.4f}, Sidak sur "
        f"{TESTED_PROPERTIES} = {1 - (1 - p_exact) ** TESTED_PROPERTIES:.3f}"
    )
    show("part calme -> 1re division (une issue du run)", quiet, first)
    show("1re division -> log ESR", first, esr)

    print("\nUn seul axe a n = 8 : les partielles sont symetriques.")
    print(f"  calme -> ESR, division constante  {partial(quiet, esr, first):+.3f}")
    print(f"  division -> ESR, calme constant   {partial(first, esr, quiet):+.3f}")

    print("\nQuatre routes candidates, fenetre fixe 20-140 :")
    for key in ("cv(loss/val/tot)", "drift_over_noise", "level", "mrstft_share"):
        values = [routes[s][key] for s in SEEDS]
        if len(set(round(v, 6) for v in values)) < 3:
            print(f"  {key:44s} constante, {min(values):.3f} a {max(values):.3f}")
            continue
        show(f"part calme -> {key}", quiet, values)
        show(f"{key} -> 1re division", values, first)

    print("\nCe qui n'est pas une correlation : le bruit face au seuil.")
    for name in ("loss/val/tot", "metric/val/esr"):
        values = [routes[s][f"cv({name})"] for s in SEEDS]
        print(
            f"  CV de {name:18s} moyenne {np.mean(values):.3f}"
            f"  ({min(values):.3f} a {max(values):.3f})"
        )
    print("  seuil relatif de ReduceLROnPlateau : 1e-4")


if __name__ == "__main__":
    sys.exit(main())
