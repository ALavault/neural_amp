#!/usr/bin/env python3
"""Resume the fixed 24-cell M4 smoke matrix without replacing runs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/m4_smoke.yaml"
DEVICE_LABELS = {
    "fulltone_full_drive_2": "fulltone",
    "electro_harmonix_big_muff": "bigmuff",
}


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for device in config["devices"]:
        for seed in config["seeds"]:
            for model in config["models"]:
                run_id = f"m4_{DEVICE_LABELS[device]}_{model.lower()}_seed{seed}_v1"
                run_dir = ROOT / "experiments/runs" / run_id
                if run_dir.exists():
                    status = json.loads(
                        (run_dir / "status.json").read_text(encoding="utf-8")
                    )["status"]
                    if status != "completed":
                        raise RuntimeError(f"existing matrix run failed: {run_id}")
                    print(f"Already completed: {run_id}")
                    continue
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/campaigns/run_m4_smoke.py"),
                        "--model",
                        model,
                        "--device",
                        device,
                        "--seed",
                        str(seed),
                        "--run-id",
                        run_id,
                    ],
                    cwd=ROOT,
                    check=True,
                )


if __name__ == "__main__":
    main()
