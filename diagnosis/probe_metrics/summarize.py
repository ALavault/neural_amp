#!/usr/bin/env python3
"""Readings of the known-defect calibration, as fixed in NOTE.md before the numbers.

Input: calibration_test_*.csv. Output: summary_test.json (floors, detection ratios,
repeatability, observed monotonicity, level and content dependence, blind spots,
redundancy, unrepresented share and bound) and table_test.csv, the main table
metric x defect x severity x condition. CPU, seconds.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DETECTED, MISSED = 10.0, 3.0
# energy ratios (squared errors), read in amplitude so that every score scales like an
# RMS difference and one detection threshold means the same for all of them
SQUARED = {"esr", "imd"}


def load() -> list[dict]:
    rows = []
    for path in sorted(HERE.glob("calibration_test_*.csv")):
        rows += list(csv.DictReader(path.open()))
    for row in rows:
        row["score"] = float(row["normalized"] or row["value"])
        if row["metric"] in SQUARED:
            row["score"] = math.sqrt(max(row["score"], 0.0))
        row["severity"] = float(row["severity"])
        row["level_db"] = float(row["level_db"])
    return rows


def with_audio(rows: list[dict]) -> list[dict]:
    """d_audio (normalised RMS of the output difference) as one more metric, taken
    from the FIR rows, so blind spots are read against the raw difference."""
    extra = [
        {**r, "metric": "d_audio", "score": float(r["d_audio_norm"])}
        for r in rows
        if r["metric"] == "fir"
    ]
    return rows + extra


def grouped(rows: list[dict]) -> dict:
    """(metric, family, level, defect, severity) -> {seed: score}."""
    out = defaultdict(dict)
    for r in rows:
        key = (r["metric"], r["family"], r["level_db"], r["defect"], r["severity"])
        out[key][r["seed"]] = r["score"]
    return out


def label(ratio: float) -> str:
    if ratio >= DETECTED:
        return "détecté"
    return "non détecté" if ratio < MISSED else "limite"


def shares(raw: list[dict]) -> dict:
    """(metric, family, level, defect, severity) -> median over seeds of the captured
    share d_probe / d_audio and of the unrepresented share, for the probes."""
    out = defaultdict(lambda: ([], []))
    for r in raw:
        audio = float(r.get("d_audio_norm") or 0.0)
        if not r.get("unrepresented") or audio <= 0:
            continue
        key = (r["metric"], r["family"], r["level_db"], r["defect"], r["severity"])
        out[key][0].append(r["score"] / audio)
        out[key][1].append(float(r["unrepresented"]) / audio)
    return {
        k: (statistics.median(a), statistics.median(b)) for k, (a, b) in out.items()
    }


def table(cells: dict, captured: dict | None = None) -> list[dict]:
    floors = {
        k[:3]: statistics.median(v.values())
        for k, v in cells.items()
        if k[3] == "repeat_floor"
    }
    out = []
    for key, seeds in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        metric, family, level, defect, severity = key
        if defect == "repeat_floor":
            continue
        median = statistics.median(seeds.values())
        floor = floors[(metric, family, level)]
        ratio = median / floor if floor > 0 else math.inf
        logs = [math.log(max(v, 1e-300)) for v in seeds.values()]
        out.append(
            {
                "metric": metric,
                "family": family,
                "level_db": level,
                "defect": defect,
                "severity": severity,
                "median_score": median,
                "floor": floor,
                "detection_ratio": ratio,
                "reading": label(ratio),
                "log_sd_seeds": statistics.stdev(logs) if len(logs) > 1 else 0.0,
                "captured_share": (captured or {}).get(key, (None, None))[0],
                "unrepresented_share": (captured or {}).get(key, (None, None))[1],
            }
        )
    return out


def monotonicity(cells: dict) -> dict:
    """Share of (family, level, seed) where the score rises strictly with severity."""
    by = defaultdict(lambda: defaultdict(dict))
    for (metric, family, level, defect, severity), seeds in cells.items():
        for seed, value in seeds.items():
            by[(metric, defect)][(family, level, seed)][severity] = value
    out = {}
    for (metric, defect), conditions in by.items():
        if defect == "repeat_floor":
            continue
        rising = [
            all(a < b for a, b in pairwise(v))
            for v in ([s[k] for k in sorted(s)] for s in conditions.values())
        ]
        out[f"{metric}|{defect}"] = sum(rising) / len(rising)
    return out


def blind_spots(rows: list[dict]) -> list[dict]:
    """At the strongest severity of each defect (for noise: its loudest level), the
    metrics that miss a difference that d_audio detects."""
    strongest = defaultdict(lambda: -math.inf)
    for r in rows:
        strongest[r["defect"]] = max(strongest[r["defect"]], r["severity"])
    top = defaultdict(dict)
    for r in rows:
        if r["severity"] == strongest[r["defect"]]:
            top[(r["family"], r["level_db"], r["defect"])][r["metric"]] = r
    out = []
    for (family, level, defect), metrics in top.items():
        audio = metrics.get("d_audio")
        if audio is None or audio["reading"] != "détecté":
            continue
        out += [
            {
                "family": family,
                "level_db": level,
                "defect": defect,
                "metric": metric,
                "detection_ratio": r["detection_ratio"],
            }
            for metric, r in metrics.items()
            if r["reading"] == "non détecté"
        ]
    return out


def redundancy(rows: list[dict]) -> dict:
    values = defaultdict(dict)
    for r in rows:
        cell = (r["family"], r["level_db"], r["defect"], r["severity"])
        values[r["metric"]][cell] = math.log(max(r["median_score"], 1e-300))
    out = {}
    for a, b in combinations(sorted(values), 2):
        shared = sorted(set(values[a]) & set(values[b]))
        if len(shared) > 5:
            rho = spearmanr(
                [values[a][c] for c in shared], [values[b][c] for c in shared]
            )
            out[f"{a}|{b}"] = float(rho.statistic)
    return out


def profiles(rows: list[dict]) -> dict:
    """What each metric emphasises beyond the raw difference: per cell, log(score /
    floor) - log(d_audio / its floor); Spearman between metrics on these profiles, and
    for each pair the defects where their profiles differ most. Added to the
    preregistered reading because the plain redundancy is dominated by severity."""
    detect = defaultdict(dict)
    for r in rows:
        cell = (r["family"], r["level_db"], r["defect"], r["severity"])
        detect[r["metric"]][cell] = math.log(max(r["detection_ratio"], 1e-300))
    audio = detect.pop("d_audio")
    profile = {
        m: {c: v - audio[c] for c, v in cells.items() if c in audio}
        for m, cells in detect.items()
    }
    out = {}
    for a, b in combinations(sorted(profile), 2):
        shared = sorted(set(profile[a]) & set(profile[b]))
        if len(shared) <= 5:
            continue
        offset = statistics.median(profile[a][c] - profile[b][c] for c in shared)
        gap = defaultdict(list)  # centred: a constant floor offset is not a difference
        for c in shared:
            gap[c[2]].append(profile[a][c] - profile[b][c] - offset)
        by_defect = {d: statistics.median(v) for d, v in gap.items()}
        top = sorted(by_defect, key=lambda d: -abs(by_defect[d]))[:3]
        rho = spearmanr(
            [profile[a][c] for c in shared], [profile[b][c] for c in shared]
        )
        out[f"{a}|{b}"] = {
            "spearman": float(rho.statistic),
            "separating_defects": {d: round(by_defect[d], 2) for d in top},
        }
    return out


def probe_quality(raw: list[dict]) -> dict:
    out = {}
    for metric in ("fir", "hammerstein_3", "hammerstein_5"):
        mine = [
            r for r in raw if r["metric"] == metric and r["defect"] != "repeat_floor"
        ]
        if not mine:
            continue
        out[metric] = {
            "median_unrepresented": statistics.median(
                float(r["unrepresented"]) for r in mine
            ),
            "median_r_ref": statistics.median(float(r["r_ref"]) for r in mine),
            "bound_violations": sum(r["bound_holds"] != "True" for r in mine),
            "rows": len(mine),
        }
    return out


def dependence(rows: list[dict]) -> dict:
    """Level: highest / lowest level median, at the middle severity; content: spread
    of log medians across families at the middle level and severity."""
    out = {}
    by = defaultdict(dict)
    for r in rows:
        severities = sorted({x["severity"] for x in rows if x["defect"] == r["defect"]})
        if r["severity"] == severities[len(severities) // 2]:
            by[(r["metric"], r["defect"])][(r["family"], r["level_db"])] = r[
                "median_score"
            ]
    for (metric, defect), cells in by.items():
        levels = defaultdict(dict)
        for (family, level), value in cells.items():
            levels[family][level] = value
        ratios = [
            v[max(v)] / v[min(v)]
            for v in levels.values()
            if len(v) > 1 and v[min(v)] > 0
        ]
        middle = [sorted(v.items())[len(v) // 2][1] for v in levels.values()]
        logs = [math.log(max(m, 1e-300)) for m in middle]
        out[f"{metric}|{defect}"] = {
            "level_ratio_median": statistics.median(ratios) if ratios else None,
            "content_log_sd": float(np.std(logs)) if len(logs) > 1 else None,
        }
    return out


def main() -> None:
    raw = load()
    rows = table(grouped(with_audio(raw)), shares(raw))
    with (HERE / "table_test.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "monotonicity": monotonicity(grouped(with_audio(raw))),
        "blind_spots": blind_spots(rows),
        "redundancy_spearman": redundancy(rows),
        "profiles_beyond_d_audio": profiles(rows),
        "probe_quality": probe_quality(raw),
        "dependence": dependence(rows),
        "families": sorted({r["family"] for r in raw}),
    }
    (HERE / "summary_test.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
