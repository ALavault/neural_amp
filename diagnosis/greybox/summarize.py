#!/usr/bin/env python3
"""Ratios of DIAGNOSIS_greybox.md section 4 from the oracle passes. Read-only.

Per take, r = ESR_B(reference) / ESR_B(variant), median over the 8 takes, with the
references of the review deviation: V0c for V1, V2 and V4 (same total steps), V2c for
V3; the ratios against V0 and V2 beside them. V0 of the control pass is compared with
V0 of the main pass (same seed 0: GPU rounding only). Test: oracle.json +
oracle_controls.json; validation: oracle_di_validation.json +
oracle_di_validation_controls.json. Writes diagnosis/greybox/summary.json.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASSES = {
    "test": ("oracle.json", "oracle_controls.json"),
    "validation": ("oracle_di_validation.json", "oracle_di_validation_controls.json"),
}
READINGS = {
    "V1": "V0c",
    "V2": "V0c",
    "V4": "V0c",
    "V3": "V2c",
}
RAW = {"V1": "V0", "V2": "V0", "V4": "V0", "V3": "V2"}


def b(entry: dict) -> float:
    return entry["after"]["B"] if "after" in entry else entry["B"]


def main() -> None:
    out = {}
    for section, (main_file, control_file) in PASSES.items():
        main = json.loads((HERE / main_file).read_text())["by_take"]
        controls = json.loads((HERE / control_file).read_text())["by_take"]
        merged = {
            take: {k: b(v) for k, v in main[take].items() if isinstance(v, dict)}
            | {
                k: b(v)
                for k, v in controls[take].items()
                if k in ("V0c", "V2c") and isinstance(v, dict)
            }
            | {"V0_controls": b(controls[take]["V0"])}
            for take in main
        }
        summary = {"takes": list(merged)}
        summary["v0_relative_gap"] = max(
            abs(m["V0_controls"] / m["V0"] - 1) for m in merged.values()
        )
        for variant, reference in READINGS.items():
            if variant not in next(iter(merged.values())):
                continue
            if reference not in next(iter(merged.values())):
                continue
            per_take = {t: m[reference] / m[variant] for t, m in merged.items()}
            raw = {t: m[RAW[variant]] / m[variant] for t, m in merged.items()}
            summary[f"{variant}_vs_{reference}"] = {
                "median": statistics.median(per_take.values()),
                "min": min(per_take.values()),
                "max": max(per_take.values()),
                "by_take": per_take,
            }
            summary[f"{variant}_vs_{RAW[variant]}"] = statistics.median(raw.values())
        summary["V0c_vs_V0"] = statistics.median(
            m["V0"] / m["V0c"] for m in merged.values()
        )
        if "V2c" in next(iter(merged.values())):
            summary["V2c_vs_V2"] = statistics.median(
                m["V2"] / m["V2c"] for m in merged.values()
            )
        seeds = [
            k.split("_")[1]
            for k in next(iter(merged.values()))
            if k.startswith("GREY_")
        ]
        summary["GREY_vs_V0"] = {
            s: statistics.median(m[f"GREY_{s}"] / m[f"V0_{s}"] for m in merged.values())
            for s in seeds
        }
        summary["C_s42_median_B"] = statistics.median(
            m["C_s42"] for m in merged.values()
        )
        summary["V1_median_B"] = statistics.median(m["V1"] for m in merged.values())
        summary["V1_below_C"] = sum(m["V1"] < m["C_s42"] for m in merged.values())
        if "V0_default" in next(iter(merged.values())):
            summary["V0_s42_vs_V0_default"] = statistics.median(
                m["V0_default"] / m["V0_s42"] for m in merged.values()
            )
        out[section] = summary
    (HERE / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for section, summary in out.items():
        print(
            section,
            json.dumps(
                {
                    k: v
                    for k, v in summary.items()
                    if not isinstance(v, dict) or "median" not in v
                },
                indent=0,
            )[:1200],
        )
        for k, v in summary.items():
            if isinstance(v, dict) and "median" in v:
                print(
                    f"  {k}: median {v['median']:.3f} [{v['min']:.3f}, {v['max']:.3f}]"
                )


if __name__ == "__main__":
    main()
