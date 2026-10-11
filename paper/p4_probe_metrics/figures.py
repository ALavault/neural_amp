#!/usr/bin/env python3
"""Figures of the P4 dossier (probes as fidelity metrics), single-column width.

- figures/captured_share.pdf: captured share d_probe/d_audio by probe and known defect
  (strongest severity, middle level, median over the six excitation families); keys
  capt_<defect>_<probe> of numbers.json.
- figures/vs100r_separation.pdf: VS100R real pairs on di_test, score over the real
  floor (median of the three centre repeats) for Hammerstein 3 and ESR (amplitude),
  repeats / negative controls / real changes (gain 0, low signal, left out as in the
  AUC), hollow when the input is not common;
  read from diagnosis/probe_metrics/real_pairs_vs100r.csv like numbers.py; AUC keys
  vs_auc_h3_all, vs_auc_h3_common, vs_auc_esr_all, vs_auc_esr_common in the titles.
- figures/jvm_residual.pdf: JVM410H OD1, reading residual r/RMS(y) of every probe
  class tried (keys jvm_*).
Agg backend, no creation date (paper/numbers_lib.savefig): same bytes at every run.

    uv run python paper/p4_probe_metrics/figures.py
"""

import os
import sys

# This file sits next to numbers.py: drop its directory from sys.path so that the
# standard library module `numbers` is not shadowed, and import paper/numbers_lib.py.
sys.path[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

import csv  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numbers_lib import ROOT, savefig  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
WIDTH = 3.4
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
PROBES = ("fir", "h3", "h5", "vl2", "vl3")
PROBE_LABELS = ("FIR", "H3", "H5", "VL2", "VL3")
DEFECTS = (
    ("gain", "gain"),
    ("color", "coloration"),
    ("prefilter", "préfiltre"),
    ("redistrib", "redistribution H2/H3"),
    ("delay", "retard fractionnaire"),
    ("allpass", "passe-tout"),
    ("release", "relâchement"),
    ("cubic", "cubique préfiltré"),
    ("quad", "quadratique préfiltré"),
    ("cross", "terme croisé"),
    ("u7", "terme u⁷"),
    ("echo", "écho 30 ms"),
    ("noise", "bruit additif"),
)

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.formatter.use_locale": False,
        "pdf.fonttype": 42,
    }
)


def numbers() -> dict:
    return json.loads((HERE / "numbers.json").read_text())


def style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def comma(x: float, digits: int) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def captured_share(n: dict) -> None:
    grid = np.array([[n[f"capt_{d}_{p}"]["value"] for p in PROBES] for d, _ in DEFECTS])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3), layout="constrained")
    image = ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            ax.text(
                j,
                i,
                comma(value, 2),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value > 0.6 else INK,
            )
    ax.set_xticks(range(len(PROBES)), PROBE_LABELS)
    ax.set_yticks(range(len(DEFECTS)), [label for _, label in DEFECTS])
    ax.set_xlabel("sonde (gelée sur le développement)")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
    bar.set_label("part captée d_probe / d_audio (sans unité)")
    bar.ax.tick_params(labelsize=6)
    bar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: comma(v, 1))
    )
    savefig(fig, OUT / "captured_share.pdf")
    plt.close(fig)


def amplitude(r: dict) -> float:
    """ESR is an energy ratio: read in amplitude like the probes (NOTE.md)."""
    return float(r["score"]) ** (0.5 if r["metric"] == "esr" else 1)


def vs100r_separation(n: dict) -> None:
    rows = list(
        csv.DictReader((ROOT / "diagnosis/probe_metrics/real_pairs_vs100r.csv").open())
    )
    metrics = (
        ("hammerstein_3", "Hammerstein 3", "h3"),
        ("esr", "ESR (amplitude)", "esr"),
    )
    kinds = (
        ("repeat", "répétitions", MUTED, "o"),
        ("negative_control", "contrôles négatifs", BLUE, "s"),
        ("positive", "changements réels", ORANGE, "^"),
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(WIDTH, 2.4), sharey=True, layout="constrained"
    )
    for ax, (metric, label, tag) in zip(axes, metrics, strict=True):
        sel = [
            r
            for r in rows
            if r["segment"] == "di_test"
            and r["metric"] == metric
            and r["low_signal"] != "True"
        ]
        floor = statistics.median(amplitude(r) for r in sel if r["kind"] == "repeat")
        for k, (kind, _, colour, marker) in enumerate(kinds):
            use = sorted(
                (r for r in sel if r["kind"] == kind),
                key=lambda r: (r["eval"], r["ref"]),
            )
            for i, r in enumerate(use):
                common = r["input_common"] != "False"
                x = k + 0.5 * (i / max(len(use) - 1, 1) - 0.5)
                ax.scatter(
                    x,
                    amplitude(r) / floor,
                    s=12,
                    marker=marker,
                    facecolors=colour if common else "none",
                    edgecolors=colour,
                    linewidths=0.8,
                    zorder=3,
                )
        for threshold in (3, 10):
            ax.axhline(threshold, color=MUTED, linewidth=0.6, linestyle="--")
        ax.set_yscale("log")
        ax.set_xticks(range(3), ["rép.", "contr.", "chang."])
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(
            f"{label}\nAUC {n[f'vs_auc_{tag}_all']['text']} ; "
            f"commune {n[f'vs_auc_{tag}_common']['text']}"
        )
        style(ax)
    axes[0].set_ylabel("score / plancher réel (sans unité)")
    axes[0].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: comma(v, 2 if v < 0.1 else (1 if v < 1 else 0))
        )
    )
    handles = [
        matplotlib.lines.Line2D(
            [], [], marker=marker, color=colour, linestyle="", markersize=4, label=name
        )
        for _, name, colour, marker in kinds
    ]
    handles.append(
        matplotlib.lines.Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgecolor=MUTED,
            linestyle="",
            markersize=4,
            label="creux : entrée non commune",
        )
    )
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    savefig(fig, OUT / "vs100r_separation.pdf")
    plt.close(fig)


def jvm_residual(n: dict) -> None:
    bars = (
        ("H5, gain 0 (repère)", "jvm_h5_res_g0", GREEN),
        ("H5, gain 5", "jvm_h5_res_g5", BLUE),
        ("H5, gain 10", "jvm_h5_res_g10", BLUE),
        ("H5 à 1 024 taps, gain 5", "jvm_h5_1024_res_g5", BLUE),
        ("base tanh, gain 5", "jvm_tanh_res_g5", BLUE),
        ("base tanh, gain 10", "jvm_tanh_res_g10", BLUE),
        ("WH parallèle Laguerre (min)", "jvm_par_read_min", BLUE),
        ("WH parallèle Laguerre (max)", "jvm_par_read_max", BLUE),
        ("WH ajusté sur 1 s", "jvm_wh_read", ORANGE),
    )
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3), layout="constrained")
    values = [n[key]["value"] for _, key, _ in bars]
    y = np.arange(len(bars))[::-1]
    ax.barh(y, values, color=[c for _, _, c in bars], height=0.6)
    for yy, (_, key, _) in zip(y, bars, strict=True):
        ax.text(n[key]["value"] + 0.02, yy, n[key]["text"], va="center", fontsize=6)
    ax.set_yticks(y, [label for label, _, _ in bars])
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "0,25", "0,5", "0,75", "1"])
    ax.set_xlabel("résidu en lecture r / RMS(y) (sans unité)")
    style(ax)
    ax.grid(axis="y", visible=False)
    savefig(fig, OUT / "jvm_residual.pdf")
    plt.close(fig)


def main() -> None:
    n = numbers()
    captured_share(n)
    vs100r_separation(n)
    jvm_residual(n)


if __name__ == "__main__":
    main()
