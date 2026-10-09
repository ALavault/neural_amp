#!/usr/bin/env python3
"""Every number quoted in REPORT.md, recomputed from the versioned artefacts.

Inputs: table_test.csv (summarize.py), calibration_test_*.csv, real_pairs_*.csv,
dev_select*.json. Output: report_numbers.json. The report quotes this file, so a
number can always be traced back. CPU, seconds.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
PROBES = (
    "fir",
    "hammerstein_3",
    "hammerstein_5",
    "volterra_laguerre_2",
    "volterra_laguerre_3",
)


def table() -> list[dict]:
    rows = list(csv.DictReader((HERE / "table_test.csv").open()))
    for r in rows:
        for k in ("level_db", "severity", "detection_ratio", "log_sd_seeds"):
            r[k] = float(r[k])
    return rows


def extremes(rows: list[dict]) -> tuple[dict, dict, dict]:
    strongest, weakest = defaultdict(lambda: -math.inf), defaultdict(lambda: math.inf)
    for r in rows:
        strongest[r["defect"]] = max(strongest[r["defect"]], r["severity"])
        weakest[r["defect"]] = min(weakest[r["defect"]], r["severity"])
    levels = defaultdict(set)
    for r in rows:
        levels[(r["metric"], r["family"])].add(r["level_db"])
    middle = {k: sorted(v)[len(v) // 2] for k, v in levels.items()}
    return strongest, weakest, middle


def readings(rows, strongest, middle) -> dict:
    out = defaultdict(Counter)
    for r in rows:
        if (
            r["severity"] == strongest[r["defect"]]
            and r["level_db"] == middle[(r["metric"], r["family"])]
        ):
            out[r["metric"]][r["reading"]] += 1
    return {m: dict(c) for m, c in out.items()}


def weakest_detection(rows, weakest, middle) -> dict:
    out = defaultdict(dict)
    for r in rows:
        if (
            r["severity"] == weakest[r["defect"]]
            and r["level_db"] == middle[(r["metric"], r["family"])]
        ):
            out[r["defect"]].setdefault(r["metric"], [0, 0])
            out[r["defect"]][r["metric"]][0] += r["detection_ratio"] >= 10
            out[r["defect"]][r["metric"]][1] += 1
    return {d: {m: f"{a}/{b}" for m, (a, b) in v.items()} for d, v in out.items()}


def captured(rows, strongest, middle) -> dict:
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (
            r["captured_share"]
            and r["severity"] == strongest[r["defect"]]
            and r["level_db"] == middle[(r["metric"], r["family"])]
        ):
            out[r["defect"]][r["metric"]].append(
                (float(r["captured_share"]), float(r["unrepresented_share"]))
            )
    return {
        d: {
            m: {
                "captured_median": round(statistics.median(a for a, _ in v), 3),
                "unrepresented_median": round(statistics.median(b for _, b in v), 3),
                "families": len(v),
            }
            for m, v in metrics.items()
        }
        for d, metrics in out.items()
    }


def repeatability(rows) -> dict:
    out = defaultdict(list)
    for r in rows:
        if r["defect"] != "noise_db":
            out[r["metric"]].append(r["log_sd_seeds"])
    return {
        m: {
            "median": round(statistics.median(v), 4),
            "p90": round(sorted(v)[int(0.9 * len(v))], 4),
        }
        for m, v in out.items()
    }


def bound_rows() -> dict:
    out = defaultdict(lambda: [0, 0])
    for path in HERE.glob("calibration_test_*.csv"):
        for r in csv.DictReader(path.open()):
            if r.get("bound_holds"):
                out[r["metric"]][0] += r["bound_holds"] != "True"
                out[r["metric"]][1] += 1
    return {m: {"violations": a, "rows": b} for m, (a, b) in out.items()}


def broken_sweeps() -> set:
    path = HERE / "sweep_integrity.json"
    if not path.exists():
        return set()
    return {
        (r["order"], r["section"])
        for r in json.loads(path.read_text())
        if not r["intact"]
    }


def measurable(row: dict, broken: set) -> bool:
    """Harmonic rows whose sweep holds a slip in either take are not measurable."""
    if not row["metric"].startswith("harmonic"):
        return True
    orders = (int(row["ref"].split("#")[1]), int(row["eval"].split("#")[1]))
    return all((o, row["segment"]) not in broken for o in orders)


def real_vs100r() -> dict:
    broken = broken_sweeps()
    rows = [
        r
        for r in csv.DictReader((HERE / "real_pairs_vs100r.csv").open())
        if measurable(r, broken)
    ]
    for r in rows:
        r["score"] = float(r["score"]) ** (0.5 if r["metric"] == "esr" else 1)
    out = {}
    for seg in (
        "di_test",
        "pink_noise_-30dBFS_rms",
        "sweep_-20dBFS",
        "sweep_-10dBFS",
        "sweep_-3dBFS",
    ):
        for m in (
            "fir",
            "hammerstein_3",
            "hammerstein_5",
            "esr",
            "mrstft",
            "harmonic_total",
            "harmonic_order_2",
            "harmonic_order_3",
        ):
            sel = [r for r in rows if r["segment"] == seg and r["metric"] == m]
            if not sel:
                continue
            for common_only in (False, True):
                use = [
                    r
                    for r in sel
                    if not common_only or r.get("input_common") != "False"
                ]
                pos = [
                    r["score"]
                    for r in use
                    if r["kind"] == "positive" and r["low_signal"] != "True"
                ]
                neg = [r["score"] for r in use if r["kind"] == "negative_control"]
                rep = [r["score"] for r in sel if r["kind"] == "repeat"]
                floor = statistics.median(rep)
                test = mannwhitneyu(pos, neg, alternative="greater", method="exact")
                out[f"{seg}|{m}|{'common' if common_only else 'all'}"] = {
                    "auc": round(test.statistic / (len(pos) * len(neg)), 3),
                    "p_exact": round(float(test.pvalue), 5),
                    "n_pos": len(pos),
                    "n_neg": len(neg),
                    "repeats_over_floor": [round(v / floor, 2) for v in rep],
                    "negative_median_over_floor": round(
                        statistics.median(neg) / floor, 3
                    ),
                    "negative_max_over_floor": round(max(neg) / floor, 3),
                    "positive_over_floor": sorted(round(v / floor, 2) for v in pos),
                }
    return out


def real_jvm() -> dict:
    rows = list(csv.DictReader((HERE / "real_pairs_jvm.csv").open()))
    out = {}
    for dev in sorted({r["device"] for r in rows}):
        rhos = []
        for knob in "GBMT":
            for side in (1, -1):
                for m in ("fir", "hammerstein_3", "hammerstein_5", "esr", "mrstft"):
                    sel = []
                    for r in rows:
                        if r["device"] != dev or r["knob"] != knob or r["metric"] != m:
                            continue
                        value = int(r["eval"].split("_")["BMTG".index(knob)][1:])
                        if (value - 50) * side > 0:
                            sel.append((abs(value - 50), float(r["score"])))
                    if len(sel) > 2:
                        rhos.append(spearmanr(*zip(*sel, strict=True)).statistic)
        h5 = [r for r in rows if r["device"] == dev and r["metric"] == "hammerstein_5"]
        out[dev] = {
            "ladder_spearman_min": round(min(rhos), 3),
            "ladders": len(rhos),
            "h5_captured_median": round(
                statistics.median(
                    float(r["score"]) / float(r["d_audio_norm"]) for r in h5
                ),
                3,
            ),
            "h5_unrepresented_median": round(
                statistics.median(
                    float(r["unrepresented"]) / float(r["d_audio_norm"]) for r in h5
                ),
                3,
            ),
            "bound_violations": sum(
                r.get("bound_holds") == "False" for r in rows if r["device"] == dev
            ),
        }
    return out


def wiener_hammerstein() -> dict:
    path = HERE / "calibration_test_wh_noise.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["defect"], float(r["severity"]))].append(r)
    out = {}
    for (defect, severity), rs in cells.items():
        score = statistics.median(float(r["normalized"]) for r in rs)
        spread = statistics.median(float(r["start_spread"]) for r in rs)
        out[f"{defect}|{severity:g}"] = {
            "d_norm": score,
            "start_spread": spread,
            "spread_over_score": spread / score if score else math.inf,
            "unreliable": spread > 0.5 * score,
        }
    return out


def main() -> None:
    rows = table()
    strongest, weakest, middle = extremes(rows)
    numbers = {
        "readings_strongest_middle": readings(rows, strongest, middle),
        "weakest_detected": weakest_detection(rows, weakest, middle),
        "captured_strongest_middle": captured(rows, strongest, middle),
        "repeatability": repeatability(rows),
        "bound": bound_rows(),
        "dev": json.loads((HERE / "dev_select.json").read_text()),
        "dev_v2": json.loads((HERE / "dev_select_v2.json").read_text()),
        "real_vs100r": real_vs100r(),
        "real_jvm": real_jvm(),
        "wiener_hammerstein": wiener_hammerstein(),
    }
    numbers["dev"] = {
        k: {"taps": v["taps"], "ridge": v["ridge"]} for k, v in numbers["dev"].items()
    }
    (HERE / "report_numbers.json").write_text(
        json.dumps(numbers, indent=1, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
