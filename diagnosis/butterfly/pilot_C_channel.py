#!/usr/bin/env python3
"""By which channel does the data split act? (Read-only, records and logs, no GPU.)

POST HOC, not pre-registered. Only the variance ratio was declared in advance, and
pilot_C_verdict.py computes it. What follows was written after seeing that ratio land in
the intermediate band, and it is labelled as such wherever it is quoted.

Two things it does not do, on purpose.

It does not correlate anything with the gain fixe - decide. That difference contains the
decide outcome, so a seed extreme in one arm returns toward the mean in the other
whether or not any mechanism operates: regression to the mean would manufacture the
correlation. Each arm is correlated separately instead, against a property of the SPLIT,
which depends on no run.

And it does not treat the leave-one-out spread as a confidence interval. It is a
fragility check: it says how far the verdict label sits from flipping, nothing more.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(42, 50))
TAIL = (48, 49)  # the two seeds the fixed schedule rescues; a post-hoc partition
# First halving epoch of each decide run, read from its lr column by
# diagnosis/seeds/monitored_quantity.py. An outcome of the run, not a declared property.
FIRST_HALVING = {42: 285, 43: 252, 44: 161, 45: 234, 46: 255, 47: 164, 48: 167, 49: 141}
SUPPORTED, REFUTED = 0.50, 0.75


def log_esr(arm: str, seed: int) -> float:
    path = ROOT / f"demo/nablafx_bench/pilotC_{arm}_seed{seed}.json"
    return math.log(json.loads(path.read_text())["test_last"]["metric/test/esr"])


def ratio(seeds: list[int]) -> float:
    spread = {
        arm: statistics.stdev(log_esr(arm, s) for s in seeds)
        for arm in ("decide", "fixe")
    }
    return spread["fixe"] / spread["decide"]


def label(value: float) -> str:
    if value <= SUPPORTED:
        return "soutenue"
    return "refutee" if value >= REFUTED else "intermediaire"


def main() -> None:
    missing = [
        f"{a}_seed{s}"
        for a in ("decide", "fixe")
        for s in SEEDS
        if not (ROOT / f"demo/nablafx_bench/pilotC_{a}_seed{s}.json").exists()
    ]
    if missing:
        print(f"Il manque {', '.join(missing)} : rien n'est calcule.")
        return

    bulk = [s for s in SEEDS if s not in TAIL]
    print("Le rapport agrege est-il un melange ? (partition post hoc)")
    print(f"  les huit        rapport {ratio(SEEDS):.2f}")
    print(f"  sans {TAIL[0]} et {TAIL[1]}  rapport {ratio(bulk):.2f}")

    print("\nFragilite du libelle : une graine retiree a la fois.")
    values = [ratio([x for x in SEEDS if x != s]) for s in SEEDS]
    for seed, value in zip(SEEDS, values, strict=True):
        print(f"  sans {seed} : {value:.2f}  -> {label(value)}")
    print(
        f"  etendue {min(values):.2f} a {max(values):.2f} ;"
        f" les huit {ratio(SEEDS):.2f}"
    )

    props = json.loads(
        (ROOT / "diagnosis/seeds/split_property.json").read_text()
    )
    predictors = {
        "part de trames sous -40 dB": [
            props[str(s)]["val_quiet_share_mean"] for s in SEEDS
        ],
        "date de la 1re division libre": [FIRST_HALVING[s] for s in SEEDS],
        "RMS minimal de validation": [props[str(s)]["val_rms_min"] for s in SEEDS],
    }
    print("\nTest du canal : chaque bras correle separement, sans la difference.")
    print(f"{'predicteur':30s} {'-> ESR decide':>21} {'-> ESR fixe':>21}")
    for name, x in predictors.items():
        cells = []
        for arm in ("decide", "fixe"):
            r = stats.spearmanr(x, [log_esr(arm, s) for s in SEEDS])
            cells.append(f"rho {r.statistic:+.3f} p {r.pvalue:.3f}")
        print(f"{name:30s} {cells[0]:>21} {cells[1]:>21}")
    print(
        "\nSi la partition agit PAR le calendrier, la colonne fixe s'effondre ;"
        "\nsi elle agit directement, elle tient. A n = 8 l'ecart entre deux"
        "\ncorrelations appariees est lui-meme tres incertain."
    )


if __name__ == "__main__":
    sys.exit(main())
