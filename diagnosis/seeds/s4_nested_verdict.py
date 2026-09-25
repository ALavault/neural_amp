#!/usr/bin/env python3
"""Nested seed / repeat analysis, S4-TF-L-16 released training, three comparable pairs.
(Records only, no GPU.)

diagnosis/seeds/s4_nested.md: pairs of runs at the same seed. Seeds 43 and 44 pair the
initial run with its repeat (verified comparable); seed 42 pairs the repeat of
2026-09-19 with the redo of 2026-09-25, the initial run predating the guard and the
weight-decay exemption. Log test ESR, last checkpoint. Within-seed variance from the
pairs, seed component from the seed means, F(2, 3) as in hypotheses_nested.md.
Pre-registered criterion for S4 (hypotheses_nested.md, N): within-seed sd >= 0.12.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy.stats import f as fisher

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "demo/nablafx_bench"
PAIRS = {
    42: ("s4tfl16_seed42_repeat", "s4tfl16_seed42_redo"),
    43: ("s4tfl16_seed43", "s4tfl16_seed43_repeat"),
    44: ("s4tfl16_seed44", "s4tfl16_seed44_repeat"),
}


def log_esr(run: str) -> float:
    record = json.loads((BENCH / f"{run}.json").read_text())
    return math.log(record["test_last"]["metric/test/esr"])


def nested(pairs: dict) -> dict:
    values = {s: [log_esr(r) for r in runs] for s, runs in pairs.items()}
    ms_within = statistics.mean((a - b) ** 2 / 2 for a, b in values.values())
    means = [statistics.mean(v) for v in values.values()]
    ms_between = 2 * statistics.variance(means)
    f_value = ms_between / ms_within
    p = fisher.sf(f_value, len(pairs) - 1, len(pairs))
    seed_var = max(0.0, (ms_between - ms_within) / 2)
    return {
        "delta_log": {s: v[1] - v[0] for s, v in values.items()},
        "within_sd": math.sqrt(ms_within),
        "seed_sd": math.sqrt(seed_var),
        "sd_of_means": statistics.stdev(means),
        "F": f_value,
        "p": p,
    }


def main() -> None:
    check = nested({s: PAIRS[s] for s in (43, 44)})
    print(f"controle, deux paires verifiees : intra-graine {check['within_sd']:.3f}")
    result = nested(PAIRS)
    for s, d in result["delta_log"].items():
        print(f"  graine {s} : {PAIRS[s][0]} -> {PAIRS[s][1]}  delta log {d:+.3f}")
    print(
        f"trois paires : intra-graine {result['within_sd']:.3f}, composante graine "
        f"{result['seed_sd']:.3f}, F(2,3) = {result['F']:.2f}, p = {result['p']:.2f}"
    )
    print(f"critere N pour S4 (intra-graine >= 0,12) : {result['within_sd'] >= 0.12}")
    Path(__file__).with_name("s4_nested_verdict.json").write_text(
        json.dumps({"two_verified_pairs": check, "three_pairs": result}, indent=1)
        + "\n"
    )


if __name__ == "__main__":
    main()
