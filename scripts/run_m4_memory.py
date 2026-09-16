#!/usr/bin/env python3
"""Run the preregistered M4 linear-memory diagnostic (S3 with 33-tap FIRs)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/m4_memory.yaml"
LABELS = {
    "fulltone_full_drive_2": "fulltone",
    "electro_harmonix_big_muff": "bigmuff",
}


def run_id(config: dict, device: str, model: str, seed: int, preflight: bool) -> str:
    stage = "m4_memory_preflight" if preflight else "m4_memory"
    return f"{stage}_{LABELS[device]}_{model.lower()}t{config['taps']}_seed{seed}_v1"


def launch(model: str, device: str, seed: int, run: str, preflight: bool) -> None:
    run_dir = ROOT / "experiments/runs" / run
    if run_dir.exists():
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "completed":
            raise RuntimeError(f"existing memory run failed: {run}")
        print(f"Already completed: {run}")
        return
    command = [
        sys.executable,
        str(ROOT / "scripts/run_m4_smoke.py"),
        "--model",
        model,
        "--device",
        device,
        "--seed",
        str(seed),
        "--run-id",
        run,
        "--memory-taps",
    ]
    if preflight:
        command.append("--preflight")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["S3"])
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for model in args.models:
        if model not in config["models"]:
            raise ValueError(f"model outside the preregistered diagnostic: {model}")
    primary = config["decision"]["primary_device"]
    devices = [primary] + [d for d in config["devices"] if d != primary]
    for model in args.models:
        launch(model, primary, 0, run_id(config, primary, model, 0, True), True)
        for seed in config["seeds"]:
            for device in devices:
                launch(
                    model,
                    device,
                    seed,
                    run_id(config, device, model, seed, False),
                    False,
                )


if __name__ == "__main__":
    main()
