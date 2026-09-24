#!/usr/bin/env python3
"""Verdict of the paired-protocol pre-check. (Read-only, records only, no GPU.)

Pre-registration: diagnosis/seeds/paired_protocol.md, written before the first run and
untouched afterwards. Refuses to conclude before every declared cell exists, like
diagnosis/butterfly/pilot_C_verdict.py.

The design is two architectures x three declared splits, one run per cell, no replicate:
at a fixed budget, replicating reduces only the error term and never the interaction,
which dominates here, and under --deterministic a same-seed replicate is bitwise
identical anyway.

The gate comes first because nothing below means anything without it. nablafx draws the
split AFTER building the processor, so two architectures at the same --seed normally
receive almost disjoint splits - one segment in twelve, measured. --split-seed fixes
that, and the gate verifies it held in the records rather than assuming it.

The pre-registered quantity is r = sd(difference in log between the two architectures)
over sqrt(2): the real paired residual between architectures. Supported at r <= 0.25,
refuted above 0.35, reported as it stands in between.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "demo/nablafx_bench"
MODELS = ("ssm-wavenet", "s4-tf-l-16")
SPLITS = (925, 736, 688)
SEED = 42
SUPPORTED, REFUTED = 0.25, 0.35
PILOT_C_RESIDUAL = 0.301  # the only paired measurement that existed before this one
Z = 2.80  # two-sided 5 %, power 80 %


def run_id(model: str, split: int) -> str:
    return f"paired_{model.replace('-', '_')}_s{split}"


def record(model: str, split: int) -> dict | None:
    path = BENCH / f"{run_id(model, split)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def gate(table: dict) -> bool:
    """Did the two architectures really receive the same split, split by split?"""
    print("Porte : la partition est-elle identique entre architectures ?")
    every = True
    for split in SPLITS:
        seen = {
            model: tuple(table[(model, split)].get("val_indices", ()))
            for model in MODELS
        }
        distinct = {v for v in seen.values() if v}
        if not distinct:
            print(f"  split-seed {split:3d} : aucun val_indices enregistre")
            every = False
            continue
        same = len(distinct) == 1
        every = every and same
        print(f"  split-seed {split:3d} : {'conforme' if same else 'DIFFERE'}")
    return every


def main() -> None:
    table = {}
    missing = []
    for model in MODELS:
        for split in SPLITS:
            row = record(model, split)
            if row is None:
                missing.append(run_id(model, split))
            table[(model, split)] = row
    if missing:
        print(f"{6 - len(missing)}/6 runs ; il manque {', '.join(missing)}.")
        return

    passed = gate(table)
    print()

    logs = {
        key: math.log(row["test_last"]["metric/test/esr"]) for key, row in table.items()
    }
    entete = " ".join(f"{m:>14}" for m in MODELS)
    print(f"{'partition':>10} " + entete + "   difference")
    differences = []
    for split in SPLITS:
        a, b = (logs[(m, split)] for m in MODELS)
        differences.append(a - b)
        print(
            f"{split:>10} "
            + " ".join(f"{logs[(m, split)]:14.3f}" for m in MODELS)
            + f"   {a - b:+.3f}"
        )

    residual = statistics.stdev(differences) / math.sqrt(2)
    verdict = (
        "appariement soutenu"
        if residual <= SUPPORTED
        else "appariement refute"
        if residual > REFUTED
        else "intermediaire, rapporte tel quel"
    )
    print(f"\nsd de la difference = {statistics.stdev(differences):.3f}")
    print(f"r = sd / racine(2)  = {residual:.3f}   (pilote C : {PILOT_C_RESIDUAL:.3f})")
    print(f"verdict : {verdict} (soutenu <= {SUPPORTED}, refute > {REFUTED})")
    print(f"porte franchie : {passed}")

    print("\nEffet minimal detectable qui en decoule, si r se confirme :")
    for n in (3, 5, 8, 14):
        mde = math.exp(Z * residual * math.sqrt(2 / n))
        print(f"  S = {n:2d} partitions : facteur {mde:.2f}")

    # Added 2026-09-24, after the first execution: the pre-registration says the
    # interval "sera rapporte" and this script did not compute it. Descriptive only;
    # the thresholds and the verdict above are untouched.
    df = len(differences) - 1
    low, high = (residual * math.sqrt(df / chi2.ppf(q, df)) for q in (0.975, 0.025))
    print(f"\nintervalle a 95 % de r (khi-deux, {df} ddl) : [{low:.3f}, {high:.3f}]")
    print("Effet minimal detectable a la borne haute de r :")
    for n in (3, 5, 8, 14):
        print(
            f"  S = {n:2d} partitions : facteur "
            f"{math.exp(Z * high * math.sqrt(2 / n)):.2f}"
        )
    mean = statistics.mean(differences)
    print(
        f"difference moyenne (descriptive, recettes et non architectures) : "
        f"{mean:+.3f} en log, facteur {math.exp(-mean):.2f}"
    )

    (Path(__file__).parent / "paired_protocol.json").write_text(
        json.dumps(
            {
                "models": list(MODELS),
                "splits": list(SPLITS),
                "log_esr": {f"{m}_{s}": v for (m, s), v in logs.items()},
                "differences": differences,
                "residual": residual,
                "residual_ci95": [low, high],
                "mean_difference": mean,
                "gate_same_split": passed,
                "verdict": verdict,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
