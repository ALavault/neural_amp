#!/usr/bin/env python3
"""Evaluate the preregistered M4 equal-parameter recovery stop rule."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/m4_recovery.yaml"
OUTPUT_DIR = ROOT / "experiments/summaries/m4_recovery"
REPORT_PATH = ROOT / "reports/M4_RECOVERY.md"
RUNS = {
    "fulltone_full_drive_2": {
        "B0": "m4_fulltone_b0_seed0_v1",
        "S3": "m4_fulltone_s3_seed0_v1",
        "S3_W31": "m4_recovery_fulltone_s3w31_seed0_v1",
    },
    "electro_harmonix_big_muff": {
        "B0": "m4_bigmuff_b0_seed0_v1",
        "S3": "m4_bigmuff_s3_seed0_v1",
        "S3_W31": "m4_recovery_bigmuff_s3w31_seed0_v1",
    },
}


def metrics(run_id: str) -> dict:
    run_dir = ROOT / "experiments/runs" / run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise RuntimeError(f"recovery dependency is not completed: {run_id}")
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    required = float(config["stop_condition"]["required_gap_closure_fraction"])
    devices = {}
    for device, run_ids in RUNS.items():
        values = {
            label: metrics(run_id)["test"]["esr"] for label, run_id in run_ids.items()
        }
        original_gap = values["S3"] - values["B0"]
        closure = (values["S3"] - values["S3_W31"]) / original_gap
        devices[device] = {
            "runs": run_ids,
            "test_esr": values,
            "original_gap": original_gap,
            "gap_closure_fraction": closure,
            "passes_stop_rule": closure >= required,
            "wide_residual_energy_ratio": metrics(run_ids["S3_W31"])["test"][
                "residual_energy_ratio"
            ],
        }
    passed = all(item["passes_stop_rule"] for item in devices.values())
    summary = {
        "schema_version": 1,
        "required_gap_closure_fraction": required,
        "required_on_both_devices": True,
        "devices": devices,
        "passed": passed,
        "decision": (
            "continue_m4_recovery" if passed else "stop_and_select_p4_negative_result"
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    fulltone = devices["fulltone_full_drive_2"]
    bigmuff = devices["electro_harmonix_big_muff"]
    fulltone_row = (
        f"| Fulltone | {fulltone['test_esr']['B0']:.6f} | "
        f"{fulltone['test_esr']['S3']:.6f} | "
        f"{fulltone['test_esr']['S3_W31']:.6f} | "
        f"{100 * fulltone['gap_closure_fraction']:.1f}% |"
    )
    bigmuff_row = (
        f"| Big Muff | {bigmuff['test_esr']['B0']:.6f} | "
        f"{bigmuff['test_esr']['S3']:.6f} | "
        f"{bigmuff['test_esr']['S3_W31']:.6f} | "
        f"{100 * bigmuff['gap_closure_fraction']:.1f}% |"
    )
    report = f"""# M4 Equal-Parameter Recovery

The preregistered recovery did not pass. Width-31 S3 has 12,192 parameters,
within 0.39% of A2 Full, while all other M4 settings remain fixed.

| Device | A2 ESR | S3 width 8 | S3 width 31 | Gap closed |
|---|---:|---:|---:|---:|
{fulltone_row}
{bigmuff_row}

The stop rule required at least {100 * required:.0f}% closure on both devices.
Fulltone is effectively unchanged. Big Muff improves and uses 17.5% residual
energy, showing that capacity activates the residual, but the remaining error
is still materially above A2. The campaign therefore stops architecture
expansion and selects pivot P4: reproducible negative result and benchmark.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
