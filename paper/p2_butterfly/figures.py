#!/usr/bin/env python3
"""Figures of dossier P2 (effet papillon), single-column width, saved as PDF.

Reads the same tracked records as numbers.py (no model, no training):
- figures/fork_spread.pdf: children's log-ESR gap to the control, decide vs replay arm,
  at forks 5, 25 and 100 (diagnosis/butterfly/pilot_A_results.json);
- figures/dispersion.pdf: sd of log test ESR per condition, chi-square 95 % intervals,
  against the within-seed component of the nested design;
- figures/pilot_c_pairs.pdf: free against fixed schedule, per seed, seeds 42-57.

    uv run python paper/p2_butterfly/figures.py
"""

# ruff: noqa: RUF001  (French typography in labels and patterns)
from __future__ import annotations

import sys
from pathlib import Path

# numbers.py sits next to this file: keep it from shadowing the stdlib module `numbers`.
sys.path[0] = str(Path(__file__).resolve().parents[1])

import math  # noqa: E402
import statistics as st  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from numbers_lib import load, savefig  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTH = 3.4  # inches, one column
DECIDE, REPLAY, OTHER = "#b2182b", "#2166ac", "#636363"
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def log_ticks(axis, ticks: list[float]) -> None:
    """Fixed ticks on a log axis, labelled with a decimal comma."""
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_major_formatter(
        matplotlib.ticker.FixedFormatter([f"{t:g}".replace(".", ",") for t in ticks])
    )
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def esr(path: str) -> float:
    return load(path)["test_last"]["metric/test/esr"]


def fork_spread() -> None:
    """Decide vs replay at forks 5, 25, 100; at fork 5 the replay guard is free."""
    groups = load("diagnosis/butterfly/pilot_A_results.json")["groups"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    for i, fork in enumerate((5, 25, 100)):
        for arm, color, dx in (("decide", DECIDE, -0.13), ("replay", REPLAY, 0.13)):
            deltas = groups[f"fork{fork}_{arm}"]["delta"]
            hollow = fork == 5 and arm == "replay"
            ax.scatter(
                [i + dx] * len(deltas),
                deltas,
                s=14,
                marker="o" if arm == "decide" else "s",
                facecolors="none" if hollow else color,
                edgecolors=color,
                linewidths=0.8,
                zorder=3,
            )
    ax.axhline(0, color=OTHER, linewidth=0.6, linestyle=":")
    ax.set_xticks(range(3), ["5", "25", "100"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_xlabel("Époque de la fourche (perturbation d'un pas float32)")
    ax.set_ylabel("Δ ln ESR enfant − témoin (sans unité)")
    handles = [
        plt.Line2D([], [], marker="o", color=DECIDE, linestyle="", markersize=4),
        plt.Line2D([], [], marker="s", color=REPLAY, linestyle="", markersize=4),
        plt.Line2D(
            [],
            [],
            marker="s",
            markerfacecolor="none",
            color=REPLAY,
            linestyle="",
            markersize=4,
        ),
    ]
    labels = [
        "décide (décisions propres)",
        "rejoue (calendrier du témoin)",
        "rejoue, garde libre (fourche 5)",
    ]
    ax.legend(handles, labels, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    savefig(fig, HERE / "figures/fork_spread.pdf")
    plt.close(fig)


def dispersion() -> None:
    """sd(log ESR) per condition, all on the same Big Muff test set.

    Chi-square 95 % intervals only for independent seeds; the nested designs of 3 seeds
    x 2 runs (bench, pilot F) and the fork arms (four perturbations plus the unperturbed
    control, not i.i.d.) get the point estimate alone.
    """
    bench = "demo/nablafx_bench"
    c = load("diagnosis/butterfly/pilot_C_results.json")["esr"]
    x = load("diagnosis/butterfly/pilot_C_extension_results.json")["esr"]
    a = load("diagnosis/butterfly/pilot_A_results.json")["groups"]
    splits = (925, 736, 688, 744, 369)
    rows = [
        (
            "Banc, 3 × 2 (sans IC)",
            [
                esr(f"{bench}/ssmzoh_guard_seed{s}{r}.json")
                for s in (42, 43, 44)
                for r in ("", "_repeat")
            ],
            False,
        ),
        (
            "Pilote E, données fixées",
            [esr(f"{bench}/splitfix_seed{s}.json") for s in range(42, 48)],
            False,
        ),
        (
            "Pilote F, 3 × 2 (sans IC)",
            [
                esr(f"{bench}/plateau2e2_seed{s}{r}.json")
                for s in (42, 43, 44)
                for r in ("", "_repeat")
            ],
            False,
        ),
        ("Pilote C, décide (42–49)", list(c["decide"].values()), True),
        ("Pilote C, fixe (42–49)", list(c["fixe"].values()), True),
        ("Extension, décide (50–57)", [v["decide"] for v in x.values()], True),
        ("Extension, fixe (50–57)", [v["fixe"] for v in x.values()], True),
        (
            "Apparié, SSM-WaveNet",
            [esr(f"{bench}/paired_ssm_wavenet_s{s}.json") for s in splits],
            True,
        ),
        (
            "Apparié, S4-TF-L-16",
            [esr(f"{bench}/paired_s4_tf_l_16_s{s}.json") for s in splits],
            True,
        ),
        ("Fourche 100, décide (sans IC)", a["fork100_decide"]["test_esr"], True),
        ("Fourche 100, rejoue (sans IC)", a["fork100_replay"]["test_esr"], True),
    ]
    within = load("diagnosis/seeds/global_factor.json")["nested"]["ssm-changes"][
        "within_seed_sd"
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    for i, (label, values, deterministic) in enumerate(rows):
        s = st.stdev(math.log(v) for v in values)
        df = len(values) - 1
        lo = s * math.sqrt(df / stats.chi2.ppf(0.975, df))
        hi = s * math.sqrt(df / stats.chi2.ppf(0.025, df))
        color = REPLAY if deterministic else DECIDE
        if "sans IC" not in label:
            ax.plot([lo, hi], [i, i], color=color, linewidth=1.0)
        ax.plot([s], [i], marker="o", color=color, markersize=3.5)
        ax.text(
            1.02,
            i,
            f"n = {len(values)}",
            va="center",
            fontsize=5.5,
            transform=ax.get_yaxis_transform(),
        )
    ax.axvline(within, color=OTHER, linewidth=0.7, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(0.004, 2)
    log_ticks(ax.xaxis, [0.01, 0.03, 0.1, 0.3, 1])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Écart-type de ln ESR de test, IC χ² à 95 %\n(sans unité)")
    handles = [
        plt.Line2D([], [], marker="o", color=DECIDE, markersize=3.5),
        plt.Line2D([], [], marker="o", color=REPLAY, markersize=3.5),
        plt.Line2D([], [], color=OTHER, linestyle="--", linewidth=0.7),
    ]
    labels = ["non déterministe", "déterministe", "intra-graine (emboîté)"]
    fig.legend(
        handles, labels, frameon=False, loc="upper center", ncol=3, handlelength=1.5
    )
    fig.subplots_adjust(left=0.41, right=0.89, top=0.93, bottom=0.15)
    savefig(fig, HERE / "figures/dispersion.pdf")
    plt.close(fig)


def pilot_c_pairs() -> None:
    """Per seed, test ESR under the free (decide) and the common fixed schedule."""
    c = load("diagnosis/butterfly/pilot_C_results.json")["esr"]
    x = load("diagnosis/butterfly/pilot_C_extension_results.json")["esr"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    first = [(int(s), c["decide"][s], c["fixe"][s]) for s in c["decide"]]
    second = [(int(s), v["decide"], v["fixe"]) for s, v in x.items()]
    ax.scatter(
        [d for _, d, _ in first],
        [f for _, _, f in first],
        s=14,
        color=DECIDE,
        label="graines 42–49 (pilote C)",
        zorder=3,
    )
    ax.scatter(
        [d for _, d, _ in second],
        [f for _, _, f in second],
        s=14,
        marker="s",
        facecolors="none",
        edgecolors=REPLAY,
        linewidths=0.8,
        label="graines 50–57 (extension)",
        zorder=3,
    )
    for seed, d, f in first:
        if seed in (48, 49):
            ax.annotate(
                str(seed), (d, f), textcoords="offset points", xytext=(3, 3), fontsize=6
            )
    lo, hi = 0.03, 0.2
    ax.plot([lo, hi], [lo, hi], color=OTHER, linewidth=0.6, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        log_ticks(axis, [0.03, 0.05, 0.1, 0.2])
    ax.set_xlabel("ESR de test, calendrier libre (sans unité)")
    ax.set_ylabel("ESR de test, calendrier fixe commun\n(sans unité)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    savefig(fig, HERE / "figures/pilot_c_pairs.pdf")
    plt.close(fig)


def main() -> None:
    fork_spread()
    dispersion()
    pilot_c_pairs()


if __name__ == "__main__":
    main()
