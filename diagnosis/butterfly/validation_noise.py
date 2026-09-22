#!/usr/bin/env python3
"""How noisy is the quantity the scheduler decides on? (Read-only, CPU.)

Everything in this pilot runs through decisions taken on the validation loss of twelve
three-second segments: halve the learning rate on a plateau, stop after fifty epochs
without improvement. E-0011 found that this quantity does not order the final test ESR.
This measures the obvious candidate cause - the sampling noise of a twelve-segment mean.

The split is the one the runs used: reproduced by repeating the pilot's construction
order
(seed 42, processor, then data module) and checked against the logged validation loss of
three finished runs, which it reproduces to 0.3 %, the difference being GPU against CPU.

Per child: the loss of each of the twelve segments; then a bootstrap over the segments,
which says how much of the gap between children a different draw of twelve segments
would have changed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnosis/butterfly"))

import lightning as pl  # noqa: E402
import product_nablafx_bench as bench  # noqa: E402
import torch  # noqa: E402
from mode_connectivity import processor, weights  # noqa: E402
from nablafx.evaluation.flexible_loss import FlexibleLoss  # noqa: E402

CHILDREN = [f"butterfly_ssm_seed42_f100_decide_k{k}" for k in range(5)] + [
    f"butterfly_ssm_seed42_f100_replay_k{k}" for k in range(1, 5)
]
DRAWS = 10_000


def logged(run: str) -> float:
    with (bench.RUNS_DIR / f"nablafx_{run}/logs/metrics.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["loss/val/tot"]]
    return float(rows[-1]["loss/val/tot"])


def main() -> None:
    torch.set_num_threads(6)
    pl.seed_everything(42, workers=True)
    bench.build_processor(
        SimpleNamespace(
            model="ssm-wavenet",
            num_blocks=8,
            channels=16,
            state_dim=4,
            output_act="tanh",
            discretization="zoh",
        )
    )
    data = bench.data_module("trainval")
    data.setup("fit")
    x = torch.stack([a for a, _ in data.val_dataset])
    y = torch.stack([b for _, b in data.val_dataset])
    loss = FlexibleLoss(
        losses=[
            {"name": "l1_loss", "weight": 1.0, "alias": "l1"},
            {"name": "mrstft_loss", "weight": 0.1, "alias": "mrstft"},
        ]
    )

    model = processor()
    per_segment = {}
    for run in CHILDREN:
        model.load_state_dict(weights(run))
        model.eval()
        values = []
        with torch.no_grad():
            for i in range(len(x)):
                model.reset_states()
                values.append(float(loss(model(x[i : i + 1]), y[i : i + 1])[-1]))
        per_segment[run] = values
        print(
            f"{run[-12:]}: moyenne {np.mean(values):.5f} (journal {logged(run):.5f}),"
            f" segments de {min(values):.4f} a {max(values):.4f}"
        )

    table = np.array([per_segment[r] for r in CHILDREN])
    means = table.mean(1)
    # Standard error of the twelve-segment mean, per child, against the spread between
    # children: the scheduler reads the first and is asked to resolve the second.
    standard_errors = table.std(1, ddof=1) / np.sqrt(table.shape[1])
    print(
        f"\necart entre enfants (ecart-type des moyennes) {means.std(ddof=1):.5f}"
        "\nerreur type d'une moyenne sur 12 segments, mediane "
        f"{np.median(standard_errors):.5f}"
        f"\nrapport bruit / signal {np.median(standard_errors) / means.std(ddof=1):.2f}"
    )

    rng = np.random.default_rng(0)
    order = np.argsort(means)
    agree = 0
    for _ in range(DRAWS):
        pick = rng.integers(0, table.shape[1], table.shape[1])
        if np.array_equal(np.argsort(table[:, pick].mean(1)), order):
            agree += 1
    best = np.array(
        [
            np.argmin(table[:, rng.integers(0, table.shape[1], table.shape[1])].mean(1))
            for _ in range(DRAWS)
        ]
    )
    winners = {CHILDREN[i][-12:]: int((best == i).sum()) for i in set(best.tolist())}
    print(
        f"bootstrap sur les 12 segments, {DRAWS} tirages :"
        f" classement complet inchange {agree / DRAWS:.1%} du temps ;"
        f" le meilleur enfant change de nom dans"
        f" {1 - winners.get(CHILDREN[order[0]][-12:], 0) / DRAWS:.1%} des tirages"
    )
    print("  gagnants du bootstrap :", winners)

    (Path(__file__).parent / "validation_noise.json").write_text(
        json.dumps(
            {
                "per_segment": per_segment,
                "between_children_sd": float(means.std(ddof=1)),
                "median_standard_error": float(np.median(standard_errors)),
                "ranking_unchanged": agree / DRAWS,
                "bootstrap_winners": winners,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
