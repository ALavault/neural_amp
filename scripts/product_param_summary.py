#!/usr/bin/env python3
"""Tabulate the four parametric runs of demo/param/predictions.md from their records.

Reads demo/nablafx_bench/param_*.json (test_last_by_setting, val_last_by_setting) and
prints per-setting ESR and MR-STFT, the seen and unseen means, and the ratios each
prediction Q1-Q6 is judged on. Missing runs are reported as missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    ("ssm", "marshall"): "param_ssm_wavenet_marshall",
    ("s4", "marshall"): "param_s4_marshall",
    ("ssm", "ffuzz"): "param_ssm_wavenet_ffuzz",
    ("s4", "ffuzz"): "param_s4_ffuzz",
}
UNSEEN_FFUZZ = ("F030_V100", "F070_V100")
INTERP, EXTREME, SUSPECT = (
    "B010_M030_T070_G050",
    "B100_M000_T000_G050",
    "B030_M070_T100_G050",
)


def load(run_id: str) -> dict | None:
    path = ROOT / "demo/nablafx_bench" / f"{run_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def means(settings: dict, keys) -> dict:
    return {m: mean(settings[k][m] for k in keys) for m in ("esr", "mrstft")}


def main() -> None:
    summary = {}
    for (model, device), run_id in RUNS.items():
        record = load(run_id)
        if record is None:
            print(f"{run_id}: missing")
            continue
        test, val = record["test_last_by_setting"], record["val_last_by_setting"]
        print(
            f"\n## {run_id}  step {record['global_step']}"
            f"  stopped {record['stopped_epoch']}"
            f"  flips {record['polarity_flips']}  overall test ESR"
            f" {record['test_last']['metric/test/esr']:.4f}"
        )
        for name, table in (("test", test), ("val", val)):
            for setting, m in table.items():
                print(
                    f"  {name:4s} {setting:22s} esr {m['esr']:.4f}"
                    f"  mrstft {m['mrstft']:.4f}  n {m['segments']}"
                )
        if device == "ffuzz":
            seen = [k for k in test if k not in UNSEEN_FFUZZ]
            row = {"seen": means(test, seen), "unseen": means(test, UNSEEN_FFUZZ)}
            e = {k: v["esr"] for k, v in test.items()}
            row["q1_f030"] = e["F030_V100"] / mean([e["F020_V100"], e["F040_V100"]])
            row["q1_f070"] = e["F070_V100"] / mean([e["F060_V100"], e["F080_V100"]])
        else:
            row = {"seen": means(val, list(val)), "unseen": means(test, list(test))}
            e = {k: v["esr"] for k, v in test.items()}
            row["q2_extreme_over_interp"] = e[EXTREME] / e[INTERP]
            ranked = sorted(e, key=e.get, reverse=True)
            row["q4_suspect_rank"] = ranked.index(SUSPECT) + 1
        row["unseen_over_seen_esr"] = row["unseen"]["esr"] / row["seen"]["esr"]
        summary[(model, device)] = row
        print("  " + json.dumps(row))
    if ("ssm", "ffuzz") in summary and ("s4", "ffuzz") in summary:
        ssm, s4 = summary[("ssm", "ffuzz")], summary[("s4", "ffuzz")]
        print(
            "\nQ5 s4/ssm seen ESR (ffuzz):"
            f" {s4['seen']['esr'] / ssm['seen']['esr']:.3f}"
        )
        print(
            f"Q6 ssm gap / s4 gap: "
            f"{ssm['unseen_over_seen_esr'] / s4['unseen_over_seen_esr']:.3f}"
        )
    for model in ("ssm", "s4"):
        if (model, "marshall") in summary:
            print(
                f"Q3 {model} marshall unseen/seen ESR:"
                f" {summary[(model, 'marshall')]['unseen_over_seen_esr']:.3f}"
            )


if __name__ == "__main__":
    main()
