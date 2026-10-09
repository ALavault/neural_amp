#!/usr/bin/env python3
"""Diagnostic figures of the probe metrics (reads table_test.csv, real_pairs_*.csv).

- detection_<family>.png: metric x defect, strongest severity, middle level, colour =
  log10(score / repeat floor) on one sequential hue; ✓ detected (>= 10), ~ limit,
  ✗ missed (< 3) written in each cell, so the reading never rests on colour alone.
- sensitivity_<family>.png: one small panel per defect, score / floor against the
  three severities, one line and marker per metric, thresholds 3 and 10 dashed.
- real_vs100r.png: VS100R real pairs, score / median of the three centre repeats,
  per metric, repeats / negative controls / real changes.
Writes into figures/. CPU, seconds.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
METRICS = [
    "fir",
    "hammerstein_3",
    "hammerstein_5",
    "volterra_laguerre_2",
    "harmonic_total",
    "esr",
    "mrstft",
    "d_audio",
]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def load_table() -> list[dict]:
    rows = list(csv.DictReader((HERE / "table_test.csv").open()))
    for r in rows:
        for k in ("level_db", "severity", "median_score", "floor", "detection_ratio"):
            r[k] = float(r[k])
    return rows


def middle_level(rows: list[dict], family: str) -> float:
    levels = sorted({r["level_db"] for r in rows if r["family"] == family})
    return levels[len(levels) // 2]


def detection_map(rows: list[dict], family: str) -> None:
    level = middle_level(rows, family)
    strongest = defaultdict(float)
    for r in rows:
        strongest[r["defect"]] = max(strongest[r["defect"]], r["severity"])
    picked = [
        r
        for r in rows
        if r["family"] == family
        and r["level_db"] == level
        and r["severity"] == strongest[r["defect"]]
    ]
    metrics = [m for m in METRICS if any(r["metric"] == m for r in picked)]
    defects = sorted({r["defect"] for r in picked})
    grid = np.full((len(metrics), len(defects)), np.nan)
    for r in picked:
        if r["metric"] in metrics:
            ratio = max(r["detection_ratio"], 1e-3)
            grid[metrics.index(r["metric"]), defects.index(r["defect"])] = math.log10(
                min(ratio, 1e9)
            )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(
        figsize=(1.0 + 0.62 * len(defects), 0.9 + 0.42 * len(metrics))
    )
    image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=6, aspect="auto")
    for i in range(len(metrics)):
        for j in range(len(defects)):
            value = grid[i, j]
            if np.isnan(value):
                continue
            mark = "✓" if value >= 1 else ("✗" if value < math.log10(3) else "~")
            ax.text(
                j,
                i,
                mark,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > 3.5 else INK,
            )
    ax.set_xticks(range(len(defects)), defects, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(metrics)), metrics, fontsize=8)
    ax.set_title(
        f"{family}, {level:g} dBFS, sévérité maximale : log10(score / plancher)",
        fontsize=9,
        color=INK,
        loc="left",
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / f"detection_{family}.png", dpi=150)
    plt.close(fig)


def sensitivity(rows: list[dict], family: str) -> None:
    level = middle_level(rows, family)
    mine = [r for r in rows if r["family"] == family and r["level_db"] == level]
    defects = sorted({r["defect"] for r in mine})
    cols = 4
    fig, axes = plt.subplots(
        math.ceil(len(defects) / cols), cols, figsize=(11, 8.5), sharey=True
    )
    for ax, defect in zip(axes.flat, defects, strict=False):
        style(ax)
        severities = sorted({r["severity"] for r in mine if r["defect"] == defect})
        for k, metric in enumerate(METRICS):
            values = {
                r["severity"]: r["detection_ratio"]
                for r in mine
                if r["defect"] == defect and r["metric"] == metric
            }
            if not values:
                continue
            ys = [max(values.get(s, np.nan), 1e-2) for s in severities]
            ax.plot(
                range(1, len(severities) + 1),
                ys,
                color=SERIES[k],
                linewidth=2,
                marker=MARKERS[k],
                markersize=6,
                label=metric,
            )
        for threshold in (3, 10):
            ax.axhline(threshold, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(severities) + 1), [f"{s:g}" for s in severities])
        ax.set_title(defect, fontsize=8, color=INK, loc="left")
    for ax in list(axes.flat)[len(defects) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(
        f"{family}, {level:g} dBFS : score / plancher de répétition (log)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / f"sensitivity_{family}.png", dpi=150)
    plt.close(fig)


def real_vs100r() -> None:
    path = HERE / "real_pairs_vs100r.csv"
    if not path.exists():
        return
    rows = [r for r in csv.DictReader(path.open()) if r["metric"] != "excluded"]
    segment = {"harmonic": "sweep_-10dBFS"}
    scores = defaultdict(lambda: defaultdict(list))
    for r in rows:
        wanted = (
            segment["harmonic"] if r["metric"].startswith("harmonic") else "di_test"
        )
        if r["segment"] == wanted and r["metric"] in METRICS:
            scores[r["metric"]][r["kind"]].append(float(r["score"]))
    metrics = [m for m in METRICS if m in scores]
    kinds = ["repeat", "negative_control", "positive"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    style(ax)
    for k, kind in enumerate(kinds):
        for i, metric in enumerate(metrics):
            floor = statistics.median(scores[metric]["repeat"])
            values = [v / floor for v in scores[metric][kind]]
            ax.scatter(
                i + 0.22 * (k - 1) + np.zeros(len(values)),
                values,
                s=36,
                color=SERIES[k],
                marker=MARKERS[k],
                edgecolor="white",
                linewidth=0.8,
                label=kind if i == 0 else None,
                zorder=3,
            )
    for threshold in (3, 10):
        ax.axhline(threshold, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_xticks(range(len(metrics)), metrics, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score / médiane des répétitions", fontsize=8, color=INK)
    ax.set_title(
        "VS100R, envoi FX, di_test (signature : sinus -10 dBFS)",
        fontsize=9,
        color=INK,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "real_vs100r.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = load_table()
    for family in sorted({r["family"] for r in rows}):
        detection_map(rows, family)
        sensitivity(rows, family)
    real_vs100r()


if __name__ == "__main__":
    main()
