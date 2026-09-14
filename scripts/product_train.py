#!/usr/bin/env python3
"""Train one A2 model for the plugin demo with the official NAM trainer."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from nam.models import init_from_nam
from nam.train import full

from fssr_nam.metrics.time import time_metrics
from fssr_nam.product.data import device_pairs

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/product_a2.yaml"
RUNS_LOG = ROOT / "demo/RUNS.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the last Lightning checkpoint if one exists",
    )
    return parser.parse_args()


def build_configs(
    config: dict, pairs: dict, max_epochs: int
) -> tuple[dict, dict, dict]:
    data = {
        "common": {"delay": 0, "require_input_pre_silence": None},
        "train": {
            "x_path": str(pairs["train"][0]),
            "y_path": str(pairs["train"][1]),
            "ny": int(config["segment_output_samples"]),
        },
        "validation": {
            "x_path": str(pairs["validation"][0]),
            "y_path": str(pairs["validation"][1]),
            "ny": None,
        },
        "joint": [],
    }
    model = json.loads((ROOT / config["model_config"]).read_text(encoding="utf-8"))
    learning = {
        "train_dataloader": dict(config["train_dataloader"]),
        "val_dataloader": dict(config["validation_dataloader"]),
        "trainer": {
            "accelerator": config["accelerator"],
            "devices": int(config["devices_count"]),
            "max_epochs": max_epochs,
            "precision": config["precision"],
            "deterministic": "warn",
            "num_sanity_val_steps": 0,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "logger": True,
        },
        "trainer_fit_kwargs": {},
    }
    return data, model, learning


def evaluate_test(run_dir: Path, test_pair: tuple[Path, Path]) -> dict[str, dict]:
    """Score both exported submodels on the held-out test pair."""
    container = json.loads((run_dir / "model.nam").read_text(encoding="utf-8"))
    x, rate = sf.read(test_pair[0], dtype="float32")
    target, target_rate = sf.read(test_pair[1], dtype="float32")
    if rate != 48_000 or target_rate != 48_000:
        raise RuntimeError("test pair is not 48 kHz")
    metrics = {}
    labels = ("lite", "full")
    for label, submodel in zip(labels, container["config"]["submodels"], strict=True):
        model_dict = submodel["model"]
        (run_dir / f"model_{label}.nam").write_text(
            json.dumps(model_dict) + "\n", encoding="utf-8"
        )
        model = init_from_nam(model_dict).eval()
        with torch.inference_mode():
            prediction = model(torch.from_numpy(x), pad_start=True).cpu().numpy()
        if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"invalid {label} test prediction")
        metrics[label] = time_metrics(prediction, target)
    return metrics


def _find_last_checkpoint(run_dir: Path) -> Path | None:
    """Find the most recent Lightning checkpoint to resume from.

    The NAM trainer saves checkpoint_last_NNNN_SSSS.ckpt and per-epoch
    checkpoint_epoch_NNNN.ckpt files. We prefer checkpoint_last if it
    exists (it includes the optimizer state), otherwise the highest-epoch
    checkpoint_epoch.
    """
    ckpt_dir = run_dir / "lightning_logs"
    if not ckpt_dir.exists():
        return None
    # Find the latest version directory
    versions = sorted(ckpt_dir.glob("version_*"), key=lambda p: p.name)
    if not versions:
        return None
    checkpoints = list((versions[-1] / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        return None
    # Prefer checkpoint_last, then highest epoch
    last = [c for c in checkpoints if "checkpoint_last" in c.name]
    if last:
        return max(last, key=lambda p: p.stat().st_mtime)
    epoch = [c for c in checkpoints if "checkpoint_epoch" in c.name]
    if epoch:
        return max(epoch, key=lambda p: p.stat().st_mtime)
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def run_training(
    pairs: dict,
    *,
    run_id: str,
    device: str,
    seed: int,
    max_epochs: int,
    config: dict | None = None,
    progress_bar: bool = True,
    resume: bool = False,
) -> dict:
    """Train one A2 model on prepared pairs and append the run to demo/RUNS.jsonl.

    With resume=True, continues from the last Lightning checkpoint if one
    exists in the run directory. This makes long runs survive OOM kills on
    shared machines.
    """
    config = config or yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    run_dir = ROOT / "demo/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    data, model, learning = build_configs(config, pairs, max_epochs)
    learning["trainer"]["enable_progress_bar"] = progress_bar

    if resume:
        ckpt = _find_last_checkpoint(run_dir)
        if ckpt is not None:
            learning.setdefault("trainer_fit_kwargs", {})["ckpt_path"] = str(ckpt)
            print(f"  resuming from {ckpt.name}", flush=True)

    started = time.perf_counter()
    full.main(data, model, learning, run_dir, no_show=True, make_plots=False)
    metrics = evaluate_test(run_dir, pairs["test"])
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    record = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "device": device,
        "seed": seed,
        "max_epochs": max_epochs,
        "minutes": round((time.perf_counter() - started) / 60.0, 2),
        "test_esr": {label: value["esr"] for label, value in metrics.items()},
        "scope": "product measurement, dev set",
    }
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    record = run_training(
        device_pairs(ROOT / config["manifest"], args.device, root=ROOT),
        run_id=args.run_id or f"product_a2_{args.device}_seed{args.seed}",
        device=args.device,
        seed=args.seed,
        max_epochs=args.max_epochs or int(config["max_epochs"]),
        config=config,
        resume=args.resume,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
