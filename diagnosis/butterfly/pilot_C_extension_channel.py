#!/usr/bin/env python3
"""Pilot C extension: do the post hoc readings of seeds 42-49 hold on seeds 50-57?
(Read-only, CPU, no GPU.)

POST HOC, written 2026-09-26 after the extension verdict. Nothing here was declared in
pilot_C_extension.md. It replays, on the eight new seeds, the channel test of
pilot_C_channel.py (each arm correlated separately with a property of the split,
measured by diagnosis/seeds/split_property.py, splits verified against the logged
validation loss of the decide run), then describes the paired gain
ln(ESR fixe / ESR decide) against the first free halving and the decide run's length.
The gain contains the decide outcome: correlations with it are open to regression to
the mean, and are reported as descriptions only.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnosis/seeds"))

from mode_connectivity import processor, weights  # noqa: E402
from nablafx.evaluation.flexible_loss import FlexibleLoss  # noqa: E402
from pilot_C_verdict import RUNS, halving_epochs  # noqa: E402
from split_property import features, logged, split  # noqa: E402

SEEDS = range(50, 58)


def log_esr(arm: str, seed: int) -> float:
    path = ROOT / f"demo/nablafx_bench/pilotC_{arm}_seed{seed}.json"
    return math.log(json.loads(path.read_text())["test_last"]["metric/test/esr"])


def last_epoch(run: str) -> int:
    with (RUNS / f"nablafx_{run}/logs/metrics.csv").open() as stream:
        return max(int(r["epoch"]) for r in csv.DictReader(stream) if r.get("epoch"))


def main() -> None:
    torch.set_num_threads(4)
    loss = FlexibleLoss(
        losses=[
            {"name": "l1_loss", "weight": 1.0, "alias": "l1"},
            {"name": "mrstft_loss", "weight": 0.1, "alias": "mrstft"},
        ]
    )
    model = processor()
    rows = {}
    for seed in SEEDS:
        _, val_x, val_y, _ = split(seed)
        run = f"pilotC_decide_seed{seed}"
        model.load_state_dict(weights(run))
        model.eval()
        with torch.no_grad():
            model.reset_states()
            recomputed = float(loss(model(val_x), val_y)[-1])
        reference = logged(run)
        rows[seed] = {
            "quiet_share": features(val_y.numpy()[:, 0])["quiet_share_mean"],
            "first_halving": halving_epochs(run)[0],
            "decide_last_epoch": last_epoch(run),
            "log_decide": log_esr("decide", seed),
            "log_fixe": log_esr("fixe", seed),
            "split_check_gap": abs(recomputed - reference) / reference,
        }
        print(
            f"graine {seed} : trames calmes {rows[seed]['quiet_share']:.3f}, "
            f"controle de partition {rows[seed]['split_check_gap']:.1%}"
        )

    seeds = list(SEEDS)
    column = {k: [rows[s][k] for s in seeds] for k in rows[seeds[0]]}
    gain = [
        f - d for f, d in zip(column["log_fixe"], column["log_decide"], strict=True)
    ]
    out = {"rows": rows, "spearman": {}}
    print("\nTest du canal, bras par bras (comme pilot_C_channel.py) :")
    for name in ("quiet_share", "first_halving"):
        for arm in ("decide", "fixe"):
            r = stats.spearmanr(column[name], column[f"log_{arm}"])
            out["spearman"][f"{name}->{arm}"] = [r.statistic, r.pvalue]
            print(
                f"  {name:15s} -> ESR {arm:6s} rho {r.statistic:+.3f} p {r.pvalue:.3f}"
            )
    print("\nGain apparie ln(fixe / decide), descriptif (regression vers la moyenne) :")
    for name in ("first_halving", "decide_last_epoch"):
        r = stats.spearmanr(column[name], gain)
        out["spearman"][f"{name}->gain"] = [r.statistic, r.pvalue]
        print(f"  {name:17s} -> gain rho {r.statistic:+.3f} p {r.pvalue:.3f}")
    r = stats.spearmanr(column["first_halving"], column["decide_last_epoch"])
    out["spearman"]["first_halving->decide_last_epoch"] = [r.statistic, r.pvalue]
    print(f"  first_halving -> decide_last_epoch rho {r.statistic:+.3f}")
    (Path(__file__).parent / "pilot_C_extension_channel.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
