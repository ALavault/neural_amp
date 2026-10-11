#!/usr/bin/env python3
"""Figures of dossier P3, from numbers.json (run numbers.py first).

- figures/fig_esr_vs_rate.pdf: ESR against the model at 48 kHz versus target rate,
  per correction;
- figures/fig_zoh_advance.pdf: T1 closed forms for a real mode (ZOH advance,
  kappa_mode) and the network kappa measured by kappa.py and kappa_foh.py;
- figures/fig_floor_levels.pdf: residual floor of the two-coefficient correction
  versus input level.

French labels with typographic minus and prime signs (RUF001 ignored for this file).

    uv run python paper/p3_sample_rate/figures.py
"""

# ruff: noqa: RUF001
from __future__ import annotations

import json
import sys
from pathlib import Path

# This directory holds numbers.py, which would shadow the stdlib module `numbers`
# (imported by numpy, statistics): replace it by paper/ on the import path, which
# also makes numbers_lib importable.
HERE = Path(__file__).resolve().parent
sys.path[0] = str(HERE.parent)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numbers_lib import savefig  # noqa: E402

NUM = json.loads((HERE / "numbers.json").read_text())
SEEDS = (925, 736, 688)
RATES = {"88": 88200 / 1000, "96": 96000 / 1000, "192": 192000 / 1000}
WIDTH = 3.4
plt.rcParams.update(
    {"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "pdf.fonttype": 42}
)


def v(key: str) -> float:
    return NUM[key]["value"]


def esr_vs_rate() -> None:
    arms = [
        ("up_delta_esr", "ZOH, Δ rééchelonné seul", "C3"),
        ("hold_esr", "ZOH + maintien 1 coef.", "C1"),
        ("twotap_esr", "ZOH + 2 coef. (= relecture FOH)", "C0"),
        ("fohtrain_plain_esr", "entraîné en FOH, Δ seul", "C2"),
    ]
    markers = dict(zip(SEEDS, ("o", "s", "^"), strict=True))
    x = list(RATES.values())
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    for prefix, label, color in arms:
        for i, s in enumerate(SEEDS):
            y = [v(f"{prefix}_s{s}_{r}") for r in RATES]
            ax.plot(
                x,
                y,
                color=color,
                marker=markers[s],
                ms=3,
                lw=0.8,
                label=label if i == 0 else None,
            )
    anchor = np.mean([v(f"up_delta_esr_s{s}_96") for s in SEEDS])
    guide = [anchor * (v(f"t4_jauge_{r}") / v("t4_jauge_96")) ** 2 for r in RATES]
    ax.plot(x, guide, "k--", lw=0.7, label="∝ jauge² de T4 (lecture après coup)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{r:g}".replace(".", ",") for r in x])
    ax.minorticks_off()
    ax.set_xlabel("fréquence cible fs′ (kHz)")
    ax.set_ylabel("ESR contre le modèle à 48 kHz")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.38), frameon=False)
    ax.set_title("Big Muff, 3 partitions (o 925, □ 736, △ 688)", fontsize=7)
    fig.tight_layout()
    savefig(fig, HERE / "figures" / "fig_esr_vs_rate.pdf")
    plt.close(fig)


def zoh_advance() -> None:
    s = np.logspace(-3, 2, 400)

    def g(x: np.ndarray) -> np.ndarray:
        return x / np.expm1(x)

    advance = 1 / s - 1 / np.expm1(s)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(WIDTH, 3.6), sharex=True)
    top.plot(s, advance, "C3", label="ZOH : 1/s − 1/(eˢ − 1)")
    top.axhline(0, color="C2", label="FOH : 0")
    top.axhline(v("t1_phi0"), color="k", ls=":", lw=0.7)
    top.set_ylabel("avance à fréquence nulle\n(échantillons)")
    top.legend(frameon=False)
    for r, fs, style in ((2, "96", "-"), (4, "192", "--")):
        kappa = 2 * (g(s / r) - g(s)) / (s * (1 - 1 / r))
        bottom.plot(s, kappa, "C0", ls=style, label=f"κ_mode, fs′ = {fs} kHz (r = {r})")
    bottom.axhline(1, color="k", ls=":", lw=0.7)
    bm = [v(f"kappa_bm_s{seed}_{r}") for seed in SEEDS for r in ("96", "192")]
    bottom.axhspan(
        min(bm),
        max(bm),
        color="C3",
        alpha=0.25,
        lw=0,
        label="réseau ZOH mesuré, Big Muff (3 checkpoints)",
    )
    bottom.axhline(
        v("kappa_vs100r"),
        color="C1",
        lw=1,
        label="réseau ZOH mesuré, VS100R (1 checkpoint)",
    )
    bottom.axhline(
        v("fohk_primaire"), color="C2", lw=1, label="réseau entraîné en FOH (T1-P1)"
    )
    bottom.set_xscale("log")
    bottom.set_ylim(-0.1, 2.9)
    bottom.set_xlabel("s = −Re(λΔ) d'un mode réel (sans unité)")
    bottom.set_ylabel("κ (sans unité)")
    bottom.legend(frameon=False, loc="upper right", fontsize=5.5)
    fig.tight_layout()
    savefig(fig, HERE / "figures" / "fig_zoh_advance.pdf")
    plt.close(fig)


def floor_levels() -> None:
    levels = (0, -20, -40, -60)
    markers = dict(zip(SEEDS, ("o", "s", "^"), strict=True))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    for s in SEEDS:
        y = [v(f"floor_esr_s{s}_m{abs(db)}") for db in levels]
        ax.plot(
            levels, y, "C0", marker=markers[s], ms=3, lw=0.8, label=f"partition {s}"
        )
    anchor = np.mean([v(f"floor_esr_s{s}_m40") for s in SEEDS])
    ax.plot(
        [-40, -60],
        [anchor, anchor],
        "k:",
        lw=0.8,
        label="plateau (indépendant du niveau)",
    )
    ax.plot(
        [-40, -60],
        [anchor, anchor / 100],
        "k--",
        lw=0.8,
        label="distorsion faible (÷100 par 20 dB)",
    )
    ax.set_yscale("log")
    ax.set_xticks(levels)
    ax.set_xlabel("niveau d'entrée relatif (dB)")
    ax.set_ylabel("ESR, 2 coef. à 96 kHz\ncontre le modèle à 48 kHz")
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    savefig(fig, HERE / "figures" / "fig_floor_levels.pdf")
    plt.close(fig)


if __name__ == "__main__":
    esr_vs_rate()
    zoh_advance()
    floor_levels()
    for f in sorted((HERE / "figures").glob("*.pdf")):
        print(f)
