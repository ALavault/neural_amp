#!/usr/bin/env python3
"""Paired protocol over five splits, and the batch-order check. (Records only, no GPU.)

Pre-registration: the section "Élargissement aux partitions 744 et 369" of
diagnosis/seeds/paired_protocol.md, written 2026-09-24 before the four new runs. Same
quantity and thresholds as paired_protocol.py, which stays as it read the first three
splits: r = sd(log ESR difference) / sqrt(2), supported <= 0.25, refuted > 0.35.
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
SPLITS = (925, 736, 688, 744, 369)
NEW = (744, 369)
SUPPORTED, REFUTED = 0.25, 0.35


def record(model: str, split: int) -> dict | None:
    path = BENCH / f"paired_{model.replace('-', '_')}_s{split}.json"
    return json.loads(path.read_text()) if path.exists() else None


def order_check(table: dict) -> dict:
    """Per new split: at which logged epochs the two architectures share the order."""
    out = {}
    for split in NEW:
        orders = [table[(m, split)].get("sampler_order") or {} for m in MODELS]
        epochs = sorted(set(orders[0]) & set(orders[1]), key=int)
        same = {e: orders[0][e] == orders[1][e] for e in epochs}
        print(
            f"  partition {split} : "
            + ", ".join(
                f"epoque {int(e) + 1} {'identique' if v else 'DIFFERE'}"
                for e, v in same.items()
            )
        )
        out[split] = same
    return out


def main() -> int:
    table = {(m, s): record(m, s) for m in MODELS for s in SPLITS}
    missing = [f"{m}@{s}" for (m, s), r in table.items() if r is None]
    if missing:
        print(f"{10 - len(missing)}/10 runs ; il manque {', '.join(missing)}.")
        return 1
    print("Porte : meme partition entre architectures ?")
    for split in SPLITS:
        same = (
            table[(MODELS[0], split)]["val_indices"]
            == table[(MODELS[1], split)]["val_indices"]
        )
        print(f"  {split} : {'conforme' if same else 'DIFFERE'}")
    print("Ordre des lots aux epoques journalisees :")
    orders = order_check(table)
    d = [
        math.log(table[(MODELS[0], s)]["test_last"]["metric/test/esr"])
        - math.log(table[(MODELS[1], s)]["test_last"]["metric/test/esr"])
        for s in SPLITS
    ]
    r = statistics.stdev(d) / math.sqrt(2)
    df = len(d) - 1
    low, high = (r * math.sqrt(df / chi2.ppf(q, df)) for q in (0.975, 0.025))
    verdict = (
        "appariement soutenu"
        if r <= SUPPORTED
        else "appariement refute"
        if r > REFUTED
        else "intermediaire, rapporte tel quel"
    )
    if r <= SUPPORTED and high > REFUTED:
        verdict += " sur l'estimation ponctuelle"
    print(
        "\ndifferences en log : "
        + ", ".join(f"{s}: {x:+.3f}" for s, x in zip(SPLITS, d, strict=True))
    )
    print(f"r5 = {r:.3f}, IC 95 % [{low:.3f}, {high:.3f}] ({df} ddl) -> {verdict}")
    print(f"difference moyenne {statistics.mean(d):+.3f} (descriptif)")
    (Path(__file__).parent / "paired_extension.json").write_text(
        json.dumps(
            {
                "differences": dict(zip(map(str, SPLITS), d, strict=True)),
                "r5": r,
                "ci95": [low, high],
                "verdict": verdict,
                "order_same": {str(k): v for k, v in orders.items()},
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
