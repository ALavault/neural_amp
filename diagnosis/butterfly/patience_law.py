#!/usr/bin/env python3
"""First-passage law of the patience scheduler (patience_law.md). Read-only, logs, CPU.

Pre-registered 2026-10-02 (patience_law.md, commit 2b447f3) before any measurement.
Replay gate, then the variance ratio of the epoch-to-epoch fluctuations of the
validation loss, then the predicted law of the first halving epoch under the retained
noise model, compared with the observed epochs. Writes patience_law.json.
"""

from __future__ import annotations

import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diagnosis/seeds"))
from plateau_simulation import halvings  # noqa: E402

RUNS = ROOT / "demo/runs"
PATIENCE, THRESHOLD = 20, 1e-4
WINDOW, NOISE_TAIL, K = 120, 60, 10
SIMS, HORIZON = 2000, 300
CHILDREN = [f"butterfly_ssm_seed42_f100_decide_k{k}" for k in range(5)]
PILOT_C = [f"pilotC_decide_seed{s}" for s in range(42, 58)]
PARENT = "butterfly_ssm_seed42_parent"


def rows(run: str) -> list[dict]:
    with (RUNS / f"nablafx_{run}/logs/metrics.csv").open() as stream:
        return list(csv.DictReader(stream))


def validation(run: str) -> list[float]:
    by_epoch = {
        int(r["epoch"]): float(r["loss/val/tot"])
        for r in rows(run)
        if r["loss/val/tot"]
    }
    return [by_epoch[e] for e in sorted(by_epoch)]


def observed_first(run: str) -> int:
    """Epoch at which the logged learning rate first changes, in the run's epochs."""
    if run in CHILDREN:
        lr = json.loads((ROOT / f"demo/butterfly/{run}.json").read_text())[
            "lr_by_epoch"
        ]
        keys = sorted(lr, key=int)
        return next(int(b) for a, b in itertools.pairwise(keys) if lr[a] != lr[b])
    # lr rows carry a step but no epoch: the epoch is that of the first row, at or
    # after that step, which carries one.
    table = rows(run)
    stamped = sorted(
        (int(r["step"]), int(r["epoch"])) for r in table if r["epoch"] and r["step"]
    )
    first = None
    for r in table:
        if r.get("lr-AdamW/pg1"):
            value = float(r["lr-AdamW/pg1"])
            first = value if first is None else first
            if value != first:
                step = int(r["step"])
                return next(e for s, e in stamped if s >= step)
    raise ValueError(f"{run}: no learning-rate change")


def curve(run: str) -> tuple[list[float], int]:
    """The validation curve the scheduler saw, and the offset of the run's epoch 0."""
    if run in CHILDREN:
        parent = validation(PARENT)
        return parent + validation(run), len(parent)
    return validation(run), 0


def trend_fit(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.polyfit(np.log(t + 1.0), y, 2)


def predict(
    full: list[float],
    start: int,
    stop: int,
    threshold: float,
    model: str,
    rng: np.random.Generator,
) -> list[int]:
    """First halving (absolute epoch) of SIMS curves: real curve before `start`, then
    trend fitted on [start, stop] plus simulated noise."""
    t = np.arange(start, stop + 1, dtype=float)
    y = np.log(np.asarray(full[start : stop + 1]))
    coef = trend_fit(t, y)
    resid = y - np.polyval(coef, np.log(t + 1.0))
    tail = resid[-NOISE_TAIL:]
    future = np.arange(start, stop + HORIZON + 1, dtype=float)
    base = np.polyval(coef, np.log(future + 1.0))
    head = list(full[:start])
    out = []
    for _ in range(SIMS):
        if model == "independent":
            noise = rng.normal(0.0, tail.std(ddof=1), len(future))
        else:
            noise = np.cumsum(rng.normal(0.0, np.diff(tail).std(ddof=1), len(future)))
        simulated = head + list(np.exp(base + noise))
        fired = halvings(simulated, threshold, PATIENCE)
        out.append(fired[0] if fired else len(simulated))
    return out


def main() -> None:
    runs = CHILDREN + PILOT_C
    info = {}
    for run in runs:
        full, offset = curve(run)
        info[run] = {"curve": full, "offset": offset, "observed": observed_first(run)}
    # Replay gate: one convention (0 or 1 epoch), the same for every run.
    gate = {}
    for shift in (0, 1):
        exact = 0
        for run in runs:
            fired = halvings(info[run]["curve"], THRESHOLD, PATIENCE)
            replay = fired[0] - info[run]["offset"] + shift if fired else None
            exact += replay == info[run]["observed"]
        gate[shift] = exact
    shift = max(gate, key=gate.get)
    print(f"porte de rejeu : {gate} -> decalage {shift}, {gate[shift]}/21 exacts")
    result = {
        "gate": {"exact_by_shift": gate, "shift": shift, "valid": gate[shift] >= 19}
    }
    if gate[shift] < 19:
        Path(__file__).with_suffix(".json").write_text(
            json.dumps(result, indent=1) + "\n"
        )
        print("porte echouee : mesure non calculable comme specifiee")
        return
    # Step 1: variance ratio of detrended residuals before the observed first halving.
    vr = {}
    for run in runs:
        full, offset = info[run]["curve"], info[run]["offset"]
        stop = info[run]["observed"] + offset - shift
        start = max(stop - WINDOW, 20 if run in PILOT_C else 0)
        info[run]["window"] = (start, stop)
        t = np.arange(start, stop + 1, dtype=float)
        y = np.log(np.asarray(full[start : stop + 1]))
        r = y - np.polyval(trend_fit(t, y), np.log(t + 1.0))
        vr[run] = float(np.var(r[K:] - r[:-K]) / (K * np.var(np.diff(r))))
    median_vr = statistics.median(vr.values())
    model = (
        "independent"
        if median_vr <= 0.3
        else "cumulative"
        if median_vr >= 0.7
        else "mixed"
    )
    print(f"VR({K}) mediane {median_vr:.3f} -> {model}")
    models = ["independent", "cumulative"] if model == "mixed" else [model]
    # Step 2: predicted law, coverage, threshold independence.
    rng = np.random.default_rng(20261002)
    predictions = {}
    for m in models:
        per_run, medians = {}, {th: [] for th in (1e-3, 1e-4, 1e-5)}
        for run in runs:
            full, offset = info[run]["curve"], info[run]["offset"]
            start, stop = info[run]["window"]
            row = {}
            for th in (1e-3, 1e-4, 1e-5):
                firsts = (
                    np.asarray(predict(full, start, stop, th, m, rng)) - offset + shift
                )
                q10, q50, q90 = (float(np.percentile(firsts, q)) for q in (10, 50, 90))
                row[str(th)] = {"q10": q10, "q50": q50, "q90": q90}
                medians[th].append(q50)
            obs = info[run]["observed"]
            row["observed"] = obs
            row["covered"] = row["0.0001"]["q10"] <= obs <= row["0.0001"]["q90"]
            per_run[run] = row
        coverage = sum(r["covered"] for r in per_run.values()) / len(runs)
        shifts = {
            str(th): statistics.median(medians[th]) - statistics.median(medians[1e-4])
            for th in (1e-3, 1e-5)
        }
        label = (
            "soutenue"
            if coverage >= 0.60
            else "refutee"
            if coverage <= 0.40
            else "intermediaire"
        )
        independent = all(abs(v) <= 5 for v in shifts.values())
        predictions[m] = {
            "per_run": per_run,
            "coverage": coverage,
            "coverage_label": label,
            "median_shift_vs_1e-4": shifts,
            "threshold_independence": independent,
        }
        print(
            f"[{m}] couverture {coverage:.2f} ({label}) ;"
            f" decalage des medianes {shifts}"
            f" -> independance au seuil {independent}"
        )
    result |= {
        "variance_ratio": {"k": K, "per_run": vr, "median": median_vr, "model": model},
        "predictions": predictions,
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
