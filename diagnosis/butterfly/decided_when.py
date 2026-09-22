#!/usr/bin/env python3
"""When is the final ranking of runs decided? (Read-only, logs and records only.)

Rank correlation between the validation loss at epoch e and the final test ESR,
across runs that share a training/validation split:

- the five "decide" children of fork 100, which share the split of seed 42 and differ
  only by a one-float32-step perturbation and by the decisions they then take;
- the same for the "replay" children, as a contrast;
- the three same-seed pairs of the benchmark (original against repeat), where the split
  is shared within a pair but not between seeds, scored as binary comparisons.

Two signals per epoch: the validation loss of that epoch, and the running best, which is
what ReduceLROnPlateau and early stopping actually read. Reported with the epoch after
which the ranking no longer changes, next to the epoch of each child's first halving.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "demo/runs"
BUTTERFLY = ROOT / "demo/butterfly"
BENCH = ROOT / "demo/nablafx_bench"
GRID = (10, 25, 50, 100, 150, 200, 300, 400, 500, 600)


def validation(run_id: str) -> dict[int, float]:
    with (RUNS / f"nablafx_{run_id}/logs/metrics.csv").open() as stream:
        return {
            int(row["epoch"]): float(row["loss/val/tot"])
            for row in csv.DictReader(stream)
            if row["loss/val/tot"]
        }


def running_best(losses: dict[int, float]) -> dict[int, float]:
    best, out = math.inf, {}
    for epoch in sorted(losses):
        best = min(best, losses[epoch])
        out[epoch] = best
    return out


def halvings(lr_by_epoch: dict[str, float]) -> list[int]:
    epochs = sorted(lr_by_epoch, key=int)
    return [
        int(b)
        for a, b in itertools.pairwise(epochs)
        if lr_by_epoch[a] != lr_by_epoch[b]
    ]


def spread(values: list[float]) -> float:
    """Standard deviation in log, so validation loss and test ESR are comparable."""
    return float(stats.tstd([math.log(v) for v in values]))


def common(series: list[dict[int, float]]) -> list[int]:
    return sorted(set.intersection(*(set(s) for s in series)))


def settle(
    curves: list[dict[int, float]], final: list[float], epochs: list[int]
) -> int | None:
    """Smallest epoch after which the ranking of the curves matches the final one."""
    target = list(stats.rankdata(final))
    for start, epoch in enumerate(epochs):
        if all(
            list(stats.rankdata([curve[e] for curve in curves])) == target
            for e in epochs[start:]
        ):
            return epoch
    return None


def settle_pair(
    a: dict[int, float], b: dict[int, float], order: bool, epochs: list[int]
):
    """Same, for one pair: an exact ranking over five runs is decided by its near-ties,
    so each pair is read on its own against the size of its final gap."""
    for start, epoch in enumerate(epochs):
        if all((a[e] < b[e]) == order for e in epochs[start:]):
            return epoch
    return None


def group(name: str, run_ids: list[str], final: list[float], report: list[str]) -> dict:
    curves = [validation(r) for r in run_ids]
    bests = [running_best(c) for c in curves]
    epochs = common(curves)
    last = epochs[-1]
    out = {"runs": run_ids, "final_test_esr": final, "last_common_epoch": last}
    for label, series in (("val", curves), ("best", bests)):
        rhos = {
            e: stats.spearmanr([s[e] for s in series], final).statistic
            for e in [*GRID, last]
            if e in set(epochs)
        }
        out[f"rho_{label}"] = {str(e): round(float(v), 3) for e, v in rhos.items()}
        report.append(
            f"{name} [{label}] rho(epoch): "
            + " ".join(f"{e}:{v:+.2f}" for e, v in rhos.items())
        )
    out["settles_at_epoch"] = settle(bests, final, epochs)
    out["val_ranking_settles_at_epoch"] = settle(
        bests, [b[last] for b in bests], epochs
    )
    report.append(
        f"{name} running best: its ranking freezes at epoch"
        f" {out['val_ranking_settles_at_epoch']}, and matches the final ESR"
        " ranking from"
        f" epoch {out['settles_at_epoch']} (None = never, last common epoch {last},"
        f" {len(run_ids)} runs)"
    )
    pairs = [
        {
            "pair": [i, j],
            "gap_log_esr": abs(math.log(final[i] / final[j])),
            "settles_at_epoch": settle_pair(
                bests[i], bests[j], final[i] < final[j], epochs
            ),
        }
        for i, j in itertools.combinations(range(len(run_ids)), 2)
    ]
    out["pairs"] = sorted(pairs, key=lambda p: -p["gap_log_esr"])
    report.append(
        f"{name} per pair, |gap| in log -> epoch from which the running best keeps the"
        " right order: "
        + " ".join(
            f"k{p['pair'][0]}k{p['pair'][1]}:{p['gap_log_esr']:.3f}->{p['settles_at_epoch']}"
            for p in out["pairs"]
        )
    )
    return out


def main() -> None:
    report: list[str] = []
    out: dict = {}
    pooled: list[tuple[float, float]] = []

    for arm in ("decide", "replay"):
        run_ids = [f"butterfly_ssm_seed42_f100_{arm}_k{k}" for k in range(5)]
        records = [json.loads((BUTTERFLY / f"{r}.json").read_text()) for r in run_ids]
        final = [r["test_last"]["metric/test/esr"] for r in records]
        out[f"fork100_{arm}"] = group(f"fork 100 {arm}", run_ids, final, report)
        out[f"fork100_{arm}"]["halvings"] = [
            halvings(r["lr_by_epoch"]) for r in records
        ]
        report.append(
            f"fork 100 {arm} first halving per child: "
            + " ".join(
                str(h[0]) if h else "-" for h in out[f"fork100_{arm}"]["halvings"]
            )
        )
        # The tested checkpoint is last.ckpt, so pair each child's own terminal epoch.
        terminal = [validation(r) for r in run_ids]
        own = [(max(v), v[max(v)]) for v in terminal]
        out[f"fork100_{arm}"]["terminal"] = [
            {"run": r, "epoch": e, "val": round(loss, 5), "test_esr": round(f, 4)}
            for r, (e, loss), f in zip(run_ids, own, final, strict=True)
        ]
        rho = stats.spearmanr([loss for _, loss in own], final)
        out[f"fork100_{arm}"]["rho_terminal"] = round(float(rho.statistic), 3)
        report.append(
            f"fork 100 {arm} at each child's own last epoch "
            + " ".join(
                f"{e}:val={loss:.4f},esr={f:.4f}"
                for (e, loss), f in zip(own, final, strict=True)
            )
        )
        report.append(
            f"fork 100 {arm} rho(terminal val loss, final test ESR)"
            f" = {rho.statistic:+.2f}"
            f" (p = {rho.pvalue:.2f}, n = 5)"
        )
        # ESR is quadratic in the error amplitude where L1 + 0.1 MR-STFT is linear, so a
        # factor 2 between the two log spreads is imposed by the definitions alone.
        spread_val, spread_esr = spread([loss for _, loss in own]), spread(final)
        report.append(
            f"fork 100 {arm} log spread: val {spread_val:.3f},"
            f" test ESR {spread_esr:.3f},"
            f" ratio {spread_esr / spread_val:.1f},"
            f" beyond the definitional factor 2 {spread_esr / 2 / spread_val:.1f}"
        )
        # replay k0 is bitwise the control, that is decide k0: pool it once.
        pooled += [(loss, f) for (_, loss), f in zip(own, final, strict=True)][
            1 if arm == "replay" else 0 :
        ]

    rho = stats.spearmanr([p[0] for p in pooled], [p[1] for p in pooled])
    out["fork100_pooled"] = {
        "n": len(pooled),
        "rho_terminal": round(float(rho.statistic), 3),
        "p": round(float(rho.pvalue), 4),
    }
    report.append(
        "fork 100, the distinct children pooled (same split, same test set): "
        f"rho(terminal val loss, final test ESR) = {rho.statistic:+.2f}"
        f" (p = {rho.pvalue:.3f}, n = {len(pooled)})"
    )

    # Benchmark: the split is shared inside a same-seed pair only, so score pairs.
    pairs = []
    for seed in (42, 43, 44):
        names = [f"ssmzoh_guard_seed{seed}", f"ssmzoh_guard_seed{seed}_repeat"]
        esr = [
            json.loads((BENCH / f"{n}.json").read_text())["test_last"][
                "metric/test/esr"
            ]
            for n in names
        ]
        pairs.append((names, [running_best(validation(n)) for n in names], esr))
    epochs = common([best for _, bests, _ in pairs for best in bests])
    last = epochs[-1]
    agreement = {}
    for e in [g for g in GRID if g in set(epochs)] + [last]:
        agree = sum(
            (bests[0][e] < bests[1][e]) == (esr[0] < esr[1]) for _, bests, esr in pairs
        )
        agreement[str(e)] = f"{agree}/3"
    out["bench_same_seed_pairs"] = {
        "agreement_running_best_vs_final": agreement,
        "last_common_epoch": last,
    }
    report.append(
        "benchmark same-seed pairs, running best agrees with the final ESR: "
        + " ".join(f"{e}:{v}" for e, v in agreement.items())
    )

    text = "\n".join(report) + "\n"
    print(text, end="")
    (Path(__file__).parent / "decided_when.txt").write_text(text, encoding="utf-8")
    (Path(__file__).parent / "decided_when.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
