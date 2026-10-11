#!/usr/bin/env python3
"""Every number quoted in REPORT.md, recomputed from the versioned artefacts.

Inputs: table_test.csv (summarize.py), calibration_test_*.csv, real_pairs_*.csv,
dev_select*.json, nolly_reading.json and nolly_marker_drift.json. Output:
report_numbers.json. The report quotes this file, so a number can always be traced
back. CPU, seconds.
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


def real_nolly() -> dict:
    """Frozen Nolly reading (nolly_reading.json) and the post hoc marker check
    (nolly_marker_drift.json)."""
    reading, drift = (
        json.loads((HERE / f"nolly_{name}.json").read_text())
        for name in ("reading", "marker_drift")
    )
    gates, takes = reading["gates"], reading["gates"]["takes"]
    sheet = HERE.parents[1] / "demo/capture/v2/settings.csv"
    knobs = {
        r["order"]: (float(r["G"]), float(r["P"])) for r in csv.DictReader(sheet.open())
    }
    excluded = sorted((o for o, g in takes.items() if not g["pass"]), key=int)
    pairs = reading["pairs"]
    one_knob = [
        p["over_floor"]["esr|di_test"] for p in pairs if p["kind"] == "one_knob"
    ]
    h5 = [p["probes"]["hammerstein_5"]["di_test"]["partial_signature"] for p in pairs]
    strong = [o for o, (g, p) in knobs.items() if p == 10 and g >= 5]
    others = [o for o in drift if o not in strong]

    def span(key: str, keys: list) -> list:
        values = [drift[o][key] for o in keys]
        return [round(min(values), 2), round(max(values), 2)]

    readings = reading["readings"]
    return {
        "loopback_max_diff": gates["loopback"]["max_diff"],
        "stereo_max_lr": gates["stereo"]["max_lr"],
        "takes": len(takes),
        "usable": len(takes) - len(excluded),
        "excluded": excluded,
        "excluded_only_same_lag": all(
            g["lag_start"] != g["lag_end"]
            and abs(g["lag_start"]) < 4096
            and abs(g["lag_end"]) < 4096
            and g["markers_over_silence_db"] >= 20
            and g["quiet_step_min_db"] > -100
            for o, g in takes.items()
            if o in excluded
        ),
        "median_lag_192k": gates["median_lag"],
        "quiet_step_min_db": round(
            min(g["quiet_step_min_db"] for g in takes.values()), 1
        ),
        "markers_over_silence_min_db": round(
            min(g["markers_over_silence_db"] for g in takes.values()), 1
        ),
        "anchors_identical": reading["determinism"]["identical"],
        "anchor_esr_di_test": [
            p["baselines"]["di_test"]["esr"] for p in pairs if p["kind"] == "repeat"
        ],
        "one_knob_esr_over_floor_di_test": [round(min(one_knob)), round(max(one_knob))],
        "h5_partial_signature_pairs": [sum(h5), len(h5)],
        "H1": {k: readings["H1_monotonicity"][k] for k in ("signs", "of", "verdict")}
        | {"unmeasured": len(readings["H1_monotonicity"]["unmeasured"])},
        "H2": {
            probe: {g: round(v, 2) for g, v in r["by_gain"].items()}
            for probe, r in readings["H2_capacity"].items()
            if isinstance(r, dict)
        }
        | {"increases_with_gain": readings["H2_capacity"]["increases_with_gain"]},
        "H3": readings["H3_cab_linearity"]["verdict"],
        "H3_rho_descriptive": {
            on: {k: round(v, 4) if k != "late_energy" else v for k, v in r.items()}
            for on, r in reading["cab_linearity"].items()
        },
        "H4": readings["H4_e2_specificity"],
        "drift_lag_all_zero": all(d["lag"] == 0 for d in drift.values()),
        "drift_p10_gain5_takes": strong,
        "drift_p10_gain5_db": span("difference_db", strong),
        "drift_p10_gain5_peak_hz": span("difference_peak_hz", strong),
        "drift_p10_gain5_share_below_2k": span("difference_share_below_2k", strong),
        "drift_anchors_db": span("difference_db", ["1", "20", "39"]),
        "drift_others_db": span("difference_db", others),
        "drift_others_above_minus_40_db": {
            o: round(drift[o]["difference_db"], 1)
            for o in others
            if drift[o]["difference_db"] > -40
        },
        "drift_take18_db": round(drift["18"]["difference_db"], 1),
        "silence_rms_max_dbfs": round(
            max(d["silence_rms_dbfs"] for d in drift.values()), 1
        ),
    }


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
        "real_nolly": real_nolly(),
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
