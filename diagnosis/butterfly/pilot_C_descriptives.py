#!/usr/bin/env python3
"""The three descriptive measures pilot C's pre-registration asks for. (Read-only.)

pilot_C_verdict.py computes the verdict itself and is left untouched. This adds what the
pre-registration lists under "Descriptif" and nothing else: a one-sided F test of the
variance ratio on 7 and 7 degrees of freedom, the gap between the arms' mean log ESR -
a fixed schedule may change average quality, which is not the question - and sd(fixe)
against s(replay) = 0.365 measured at fork 5.

That last comparison is the one with teeth. The pre-registration says: if sd(fixe)
clearly exceeds s(replay), the seed effect surviving under a fixed schedule is born
before step 42 or in the data split.

Refuses to run before the sixteen runs exist, like the verdict script.
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
ARMS = ("decide", "fixe")
S_REPLAY_FORK5 = 0.365  # pilot_A_fork5_replay.md, the arm that imposed the schedule
DF = len(SEEDS) - 1


def log_esr(arm: str) -> list[float] | None:
    out = []
    for seed in SEEDS:
        path = ROOT / f"demo/nablafx_bench/pilotC_{arm}_seed{seed}.json"
        if not path.exists():
            return None
        out.append(math.log(json.loads(path.read_text())["test_last"]["metric/test/esr"]))
    return out


def main() -> None:
    arms = {a: log_esr(a) for a in ARMS}
    if any(v is None for v in arms.values()):
        print("Les seize runs ne sont pas la : rien n'est calcule.")
        return

    sd = {a: statistics.stdev(v) for a, v in arms.items()}
    mean = {a: statistics.fmean(v) for a, v in arms.items()}

    f_stat = sd["decide"] ** 2 / sd["fixe"] ** 2
    p_one_sided = stats.f.sf(f_stat, DF, DF)
    print("Test F unilateral du rapport des variances, sens pre-enregistre")
    print("(le bras fixe doit avoir la variance la PLUS PETITE si C est soutenue) :")
    print(
        f"  F({DF},{DF}) = var(decide)/var(fixe) = {f_stat:.2f}"
        f"   p = {p_one_sided:.4f}"
    )

    gap = mean["fixe"] - mean["decide"]
    print("\nEcart des moyennes entre bras - un calendrier fixe peut changer la")
    print("qualite moyenne, ce n'est pas la question posee :")
    print(
        f"  moyenne du log : decide {mean['decide']:+.3f}, fixe {mean['fixe']:+.3f},"
        f" ecart {gap:+.3f} ({math.exp(gap):.2f}x en ESR)"
    )

    print("\nsd(fixe) contre s(rejoue) de la fourche 5 :")
    print(f"  sd(fixe) = {sd['fixe']:.3f}   s(rejoue) = {S_REPLAY_FORK5:.3f}")
    print(
        "  lecture pre-enregistree : si sd(fixe) depasse nettement s(rejoue), l'effet"
        " de graine\n  qui subsiste sous calendrier fixe nait avant le pas 42 ou dans"
        " le decoupage des donnees."
    )


if __name__ == "__main__":
    sys.exit(main())
