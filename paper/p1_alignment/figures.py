#!/usr/bin/env python3
"""Figures of dossier P1, from numbers.json and the records numbers.py reads.

Agg backend, single-column width (3.4 in), no training, no inference. Writes
paper/p1_alignment/figures/<name>.pdf with numbers_lib.savefig (same bytes every run).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Drop this directory from the import path: numpy imports the standard `numbers`,
# which numbers.py next to this file would shadow.
HERE = Path(__file__).resolve().parent
sys.path[0] = str(HERE.parent)

import json  # noqa: E402
import statistics  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numbers_lib import load, savefig  # noqa: E402

NUM = json.loads((HERE / "numbers.json").read_text())
FIG = HERE / "figures"
WIDTH = 3.4
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#8a8984"
plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "pdf.fonttype": 42,
    }
)


def v(key: str) -> float:
    return NUM[key]["value"]


def by_setting(rows: list[dict], field: str) -> tuple[list[int], list[float]]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["setting"], []).append(r[field])
    names = sorted(groups)
    return [int(n[1:4]) for n in names], [statistics.mean(groups[n]) for n in names]


def ffuzz_lag() -> None:
    """ESR by fuzz setting: raw, realigned post hoc, retrained on advanced targets."""
    seg = load("diagnosis/ffuzz/segments.json")
    adv = load("diagnosis/ffuzz/segments_adv4.json")
    held = [
        int(s[1:4])
        for s in load("demo/nablafx_bench/param_ssm_wavenet_ffuzz_r2.json")[
            "held_out_settings"
        ]
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    for x in held:
        ax.axvline(x, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=0)
    arms = (
        (
            "SSM-WaveNet-FiLM",
            BLUE,
            "o",
            "param_ssm_wavenet_ffuzz_r2",
            "param_ssm_wavenet_ffuzz_adv4_r2",
        ),
        ("S4-param", ORANGE, "s", "param_s4_ffuzz_r2", "param_s4_ffuzz_adv4"),
    )
    states = (
        ("-", "brut (retard 0)"),
        ("--", "réaligné a posteriori"),
        (":", "réentraîné, cibles avancées"),
    )
    for _name, color, marker, r2, a4 in arms:
        series = (
            by_setting(seg[r2], "esr_interior_lag0"),
            by_setting(seg[r2], "esr_interior_best_lag"),
            by_setting(adv[a4], "esr_interior_lag0"),
        )
        for (x, y), (ls, _) in zip(series, states, strict=True):
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                ls=ls,
                mfc="white" if ls == "--" else color,
            )
    arm_handles = [
        plt.Line2D([], [], color=c, marker=m, ls="none", label=n)
        for n, c, m, *_ in arms
    ]
    state_handles = [
        plt.Line2D([], [], color=INK, ls=ls, label=lab) for ls, lab in states
    ]
    first = ax.legend(
        handles=arm_handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.64),
        frameon=False,
    )
    ax.add_artist(first)
    ax.legend(
        handles=state_handles,
        loc="center right",
        bbox_to_anchor=(1.0, 0.64),
        frameon=False,
        handlelength=2.4,
    )
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 0.6)
    ax.set_xlabel("Réglage fuzz (%) ; pointillés verticaux : non vus")
    ax.set_ylabel("ESR intérieure de test (sans unité)")
    ax.set_xticks(range(0, 101, 20))
    fig.tight_layout()
    savefig(fig, FIG / "ffuzz_lag.pdf")
    plt.close(fig)


def vs100r_oracle() -> None:
    """Per take: raw ESR against ESR after an oracle delay, A48 and B192, 3 seeds."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    lim = (1e-3, 0.5)
    ax.plot(lim, lim, color=MUTED, lw=0.8)
    for k, ls in ((2, "--"), (10, ":")):
        ax.plot(lim, [lim[0] / k, lim[1] / k], color=MUTED, lw=0.6, ls=ls)
        ax.text(
            0.33,
            0.33 / k / 1.25,
            f"÷{k}",
            fontsize=6,
            color=MUTED,
            ha="right",
            va="top",
        )
    paths = ("measures", "measures_s43", "measures_s44")
    for arm, color, marker in (("A48", BLUE, "o"), ("B192", ORANGE, "s")):
        xs, ys = [], []
        for p in paths:
            for d in load(f"diagnosis/vs100r_192k/{p}.json")["takes"].values():
                xs.append(d[arm]["esr"])
                ys.append(d[arm]["esr_after_delay"])
        label = (
            "entraîné à 48 kHz (A48)" if arm == "A48" else "entraîné à 192 kHz (B192)"
        )
        ax.scatter(
            xs,
            ys,
            s=12,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.4,
            label=label,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("ESR brute par prise (sans unité)")
    ax.set_ylabel("ESR après retard oracle (sans unité)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    savefig(fig, FIG / "vs100r_oracle.pdf")
    plt.close(fig)


def delay_curves() -> None:
    """Median ESR on di_validation against the common delay tau, nine runs."""
    aligned = load("demo/train_length/eval_train_length_aligned.json")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    styles = {
        "C": (INK, "-", "C, 150 s"),
        "L10": (BLUE, "--", "L10, 30 s"),
        "L20": (ORANGE, ":", "L20, 60 s"),
    }
    seen = set()
    for run in aligned["runs"].values():
        curve = run["median_esr_curve"]
        grid = np.linspace(-2, 3, len(curve))
        assert (
            len(curve) == 101 and abs(grid[int(np.argmin(curve))] - run["tau"]) < 1e-9
        )
        color, ls, label = styles[run["arm"]]
        ax.plot(
            grid,
            curve,
            color=color,
            ls=ls,
            lw=1.0,
            label=None if run["arm"] in seen else label,
        )
        ax.plot(run["tau"], min(curve), marker="o", color=color, ms=3)
        seen.add(run["arm"])
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yscale("log")
    ax.set_xlim(-1, 3)
    ax.set_xlabel("Retard commun τ (échantillons à 48 kHz)")
    ax.set_ylabel("ESR médiane, 35 prises (sans unité)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    savefig(fig, FIG / "delay_curves.pdf")
    plt.close(fig)


def train_length_readings() -> None:
    """d_i per seed: pre-registered raw reading P against the co-reading."""
    raw = load("demo/train_length/eval_train_length.json")["readings"]
    aligned = load("demo/train_length/eval_train_length_aligned.json")["readings"]
    groups = (
        ("L10\nbrut (P)", raw["L10_last"], BLUE, "o"),
        ("L10\nrecalé", aligned["L10"], BLUE, "o"),
        ("L20\nbrut (P)", raw["L20_last"], ORANGE, "s"),
        ("L20\nrecalé", aligned["L20"], ORANGE, "s"),
    )
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    thr, acc = v("trainlen_thr_verdict"), v("trainlen_thr_accept_seed")
    ax.axhspan(-thr, thr, color="#f0efec", zorder=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axhline(acc, color=MUTED, lw=0.8, ls="--")
    ax.text(
        -0.45,
        acc,
        "borne d'acceptation par graine",
        fontsize=6,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    offsets = (-0.12, 0.0, 0.12)
    for i, (_, block, color, marker) in enumerate(groups):
        di = block["s_pairs"]["d_i"]
        for off, seed in zip(offsets, ("42", "43", "44"), strict=True):
            filled = "recalé" not in groups[i][0]
            ax.plot(
                i + off,
                di[seed],
                marker=marker,
                color=color,
                mfc=color if filled else "white",
                ls="none",
            )
        ax.plot([i - 0.22, i + 0.22], [block["s_pairs"]["d"]] * 2, color=INK, lw=1.4)
        ax.text(
            i, 0.72, block["s_pairs"]["verdict"], ha="center", fontsize=6, color=INK
        )
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(-0.45, 0.8)
    ax.set_ylabel("d_i = s(bras) \u2212 s(C) (écart de ln ESR)")
    fig.tight_layout()
    savefig(fig, FIG / "train_length_readings.pdf")
    plt.close(fig)


def polarity_published() -> None:
    """Published Big Muff models and the inverted run: test L1 against test ESR."""
    pub = load("paper/icassp2027/data/published_bigmuff.json")["models"]
    inv = load("paper/icassp2027/data/polarity_ssmzoh_seed42.json")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    two = v("polar_two_mean_abs_y")
    ax.axvline(two, color=MUTED, lw=0.8, ls="--")
    ax.text(two * 0.97, 0.6, "2E|y|", fontsize=6, color=MUTED, ha="right")
    ax.axhline(4, color=MUTED, lw=0.8, ls=":")
    ax.text(
        0.0016,
        4 * 1.12,
        "ESR d'une sortie exactement inversée",
        fontsize=6,
        color=MUTED,
        va="bottom",
    )
    l1 = [m["test"]["l1"] for m in pub.values()]
    esr = [m["test"]["esr"] for m in pub.values()]
    ax.scatter(
        l1, esr, s=10, color=MUTED, label="modèles publiés (une run chacun)", zorder=2
    )
    a, n = inv["test_as_is"], inv["test_negated"]
    ax.scatter(
        [a["metric/test/l1"]],
        [a["metric/test/esr"]],
        s=22,
        color=ORANGE,
        marker="D",
        label="run inversée observée",
        zorder=3,
    )
    ax.scatter(
        [n["metric/test/l1"]],
        [n["metric/test/esr"]],
        s=22,
        color=BLUE,
        marker="D",
        label="même run, sortie niée",
        zorder=3,
    )
    ax.annotate(
        "",
        xy=(n["metric/test/l1"], n["metric/test/esr"]),
        xytext=(a["metric/test/l1"], a["metric/test/esr"]),
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.6},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.5e-3, 5e-2)
    ax.set_ylim(1e-2, 8)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(
        [2e-3, 5e-3, 1e-2, 2e-2, 4e-2], ["0,002", "0,005", "0,01", "0,02", "0,04"]
    )
    ax.set_xlabel("L1 de test (amplitude, pleine échelle = 1)")
    ax.set_ylabel("ESR de test (sans unité)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    savefig(fig, FIG / "polarity_published.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for make in (
        ffuzz_lag,
        vs100r_oracle,
        delay_curves,
        train_length_readings,
        polarity_published,
    ):
        make()
    print("\n".join(str(p.relative_to(HERE)) for p in sorted(FIG.glob("*.pdf"))))
