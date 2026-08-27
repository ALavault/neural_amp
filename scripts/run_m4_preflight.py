#!/usr/bin/env python3
"""Run one two-step physical preflight for every M4 model path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "fulltone_full_drive_2"


def main() -> None:
    for model in ("B0", "B2", "S3", "S4"):
        run_id = f"m4_preflight_fulltone_{model.lower()}_seed0_v1"
        run_dir = ROOT / "experiments/runs" / run_id
        if run_dir.exists():
            status = json.loads((run_dir / "status.json").read_text())["status"]
            if status != "completed":
                raise RuntimeError(f"existing preflight is not completed: {run_id}")
            print(f"Already completed: {run_id}")
            continue
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_m4_smoke.py"),
                "--model",
                model,
                "--device",
                DEVICE,
                "--seed",
                "0",
                "--run-id",
                run_id,
                "--preflight",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
