#!/usr/bin/env python3
"""Pilot C extension, seeds 50 to 57. (Read-only on the runs, no GPU.)

Pre-registration: diagnosis/butterfly/pilot_C_extension.md, written 2026-09-24 before
any run of these seeds. Two uses, in this order:

  --classify  after the eight "decide" runs: each seed is early (first free halving
              before epoch 200) or late, written to pilot_C_extension_groups.json,
              which is committed before any "fixe" run starts;
  (default)   after the eight "fixe" runs: R = sd(log ESR fixe) / sd(log ESR decide)
              per group of at least three seeds, and the outcome table.

The halving epochs are read by pilot_C_verdict.halving_epochs, the instrument that
measured seeds 42 to 49, unchanged.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

from pilot_C_verdict import DECLARED, halving_epochs

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "demo/nablafx_bench"
HERE = Path(__file__).parent
GROUPS = HERE / "pilot_C_extension_groups.json"
SEEDS = list(range(50, 58))
CUTOFF = 200
LOW, HIGH = 0.6, 0.9
MIN_GROUP = 3


def run_name(arm: str, seed: int) -> str | None:
    """The run that stands for this cell: the first attempt, else its one relaunch."""
    for name in (f"pilotC_{arm}_seed{seed}", f"pilotC_{arm}_seed{seed}_r2"):
        if (BENCH / f"{name}.json").exists():
            return name
    return None


def esr(name: str) -> float:
    record = json.loads((BENCH / f"{name}.json").read_text())
    return record["test_last"]["metric/test/esr"]


def classify() -> int:
    names = {s: run_name("decide", s) for s in SEEDS}
    missing = [s for s, n in names.items() if n is None]
    if missing:
        print(f"{8 - len(missing)}/8 bras decide ; il manque les graines {missing}.")
        return 1
    groups = {}
    for seed, name in names.items():
        epochs = halving_epochs(name)
        first = epochs[0] if epochs else None
        group = "precoce" if first is not None and first < CUTOFF else "tardif"
        note = "" if first is not None else "  (aucune division : classee tardive)"
        print(
            f"  graine {seed} ({name}) : 1re division epoque {first} -> {group}{note}"
        )
        groups[seed] = {"run": name, "first_halving_epoch": first, "group": group}
    GROUPS.write_text(json.dumps(groups, indent=1) + "\n", encoding="utf-8")
    counts = {
        g: sum(v["group"] == g for v in groups.values()) for g in ("precoce", "tardif")
    }
    print(f"precoces {counts['precoce']}, tardives {counts['tardif']} -> {GROUPS.name}")
    return 0


def spread_ratio(pairs: list[tuple[float, float]]) -> float:
    decide = statistics.stdev(math.log(d) for d, _ in pairs)
    fixe = statistics.stdev(math.log(f) for _, f in pairs)
    return fixe / decide


def band(ratio: float) -> str:
    return "bas" if ratio <= LOW else "haut" if ratio >= HIGH else "moyen"


OUTCOMES = {
    ("bas", "bas"): "refutee : le calendrier resserre partout",
    ("bas", "moyen"): "partiellement soutenue",
    ("bas", "haut"): "soutenue",
    ("moyen", "bas"): "refutee : sens inverse",
    ("moyen", "moyen"): "indecidable",
    ("moyen", "haut"): "partiellement soutenue",
    ("haut", "bas"): "refutee : sens inverse",
    ("haut", "moyen"): "refutee : sens inverse",
    ("haut", "haut"): "refutee : aucun resserrement",
}


def verdict() -> int:
    if not GROUPS.exists():
        print(f"{GROUPS.name} absent : lancer --classify d'abord.")
        return 1
    groups = {int(k): v for k, v in json.loads(GROUPS.read_text()).items()}
    fixe = {s: run_name("fixe", s) for s in SEEDS}
    missing = [s for s, n in fixe.items() if n is None]
    if missing:
        print(f"{8 - len(missing)}/8 bras fixe ; il manque les graines {missing}.")
        return 1

    print("graine  groupe    decide    fixe     ecart en log")
    pairs = {}
    for s in SEEDS:
        d, f = esr(groups[s]["run"]), esr(fixe[s])
        pairs[s] = (d, f)
        print(
            f"  {s}  {groups[s]['group']:8s} {d:.4f}  {f:.4f}  {math.log(f / d):+.3f}"
        )

    print("\nporte : le bras fixe divise-t-il aux huit epoques declarees ?")
    gate = True
    for s in SEEDS:
        observed = halving_epochs(fixe[s])
        gate = gate and observed == DECLARED
        print(f"  {fixe[s]:24s} {'conforme' if observed == DECLARED else 'DIFFERE'}")

    ratios = {}
    for g in ("precoce", "tardif"):
        members = [s for s in SEEDS if groups[s]["group"] == g]
        if len(members) < MIN_GROUP:
            print(f"\n{g} : {len(members)} graine(s) < {MIN_GROUP} : aucun rapport")
            continue
        ratios[g] = spread_ratio([pairs[s] for s in members])
        print(f"\n{g} : graines {members}, R = {ratios[g]:.2f} ({band(ratios[g])})")

    if len(ratios) == 2:
        outcome = OUTCOMES[(band(ratios["precoce"]), band(ratios["tardif"]))]
    else:
        outcome = "non evaluable : un groupe a moins de trois graines"
    print(f"\nverdict : {outcome}")

    print("\nDescriptifs, hors verdict :")
    new_eight = spread_ratio(list(pairs.values()))
    print(f"  huit nouvelles graines : R = {new_eight:.2f} (graines 42-49 : 0,56)")
    gate_file = ROOT / "diagnosis/seeds/repro_gate.json"
    same_bench = (
        gate_file.exists()
        and json.loads(gate_file.read_text())["results"]["repro_gate_ssm_bigmuff"][
            "identical"
        ]
    )
    pooled = None
    if same_bench:
        old = {
            s: (esr(f"pilotC_decide_seed{s}"), esr(f"pilotC_fixe_seed{s}"))
            for s in range(42, 50)
        }
        pooled = spread_ratio([*old.values(), *pairs.values()])
        print(
            f"  seize graines (porte de reproductibilite franchie) : R = {pooled:.2f}"
        )
    else:
        print("  seize graines : non calcule, porte de reproductibilite non franchie")

    (HERE / "pilot_C_extension_results.json").write_text(
        json.dumps(
            {
                "esr": {s: {"decide": d, "fixe": f} for s, (d, f) in pairs.items()},
                "groups": {s: groups[s]["group"] for s in SEEDS},
                "ratios": ratios,
                "gate_schedule_exact": gate,
                "verdict": outcome,
                "ratio_new_eight": new_eight,
                "ratio_sixteen": pooled,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(classify() if "--classify" in sys.argv else verdict())
