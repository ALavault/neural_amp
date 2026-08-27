#!/usr/bin/env python3
"""Run the two preregistered equal-parameter M4 recovery diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/m4_recovery.yaml"
LABELS = {
    "fulltone_full_drive_2": "fulltone",
    "electro_harmonix_big_muff": "bigmuff",
}


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for device in config["devices"]:
        run_id = f"m4_recovery_{LABELS[device]}_s3w31_seed0_v1"
        run_dir = ROOT / "experiments/runs" / run_id
        if run_dir.exists():
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))[
                "status"
            ]
            if status != "completed":
                raise RuntimeError(f"existing recovery run failed: {run_id}")
            print(f"Already completed: {run_id}")
            continue
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_m4_smoke.py"),
                "--model",
                config["model"],
                "--device",
                device,
                "--seed",
                str(config["seed"]),
                "--run-id",
                run_id,
                "--recovery-wide",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
