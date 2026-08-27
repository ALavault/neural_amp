#!/usr/bin/env python3
"""Train one immutable native-rate Wright LSTM-64 competence trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

from fssr_nam.losses import WrightLoss
from fssr_nam.metrics.time import time_metrics
from fssr_nam.models import WrightLSTM
from fssr_nam.reporting.ledger import append_run
from fssr_nam.training.wright import (
    evaluate_prediction,
    frame_audio,
    predict_streaming,
    train_epoch,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/r1_competence.yaml"
DATA_CONFIG_PATH = ROOT / "configs/data/r1_wright_bigmuff_native.yaml"
MANIFEST_PATH = ROOT / "datasets/manifests/r1_wright_bigmuff_native.json"
SPLIT_PATH = ROOT / "datasets/splits/r1_wright_bigmuff_native.json"
LEDGER_PATH = ROOT / ".codex_campaign/RUN_LEDGER.jsonl"
GLOBAL_INDEX_PATH = ROOT / ".codex_campaign/RESULT_INDEX.csv"
R1_INDEX_PATH = ROOT / ".codex_campaign/r1/RESULT_INDEX.csv"
PREFLIGHT_PATH = ROOT / "experiments/summaries/r1_wright_preflight.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_hash(*payloads: bytes) -> str:
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_worktree() -> None:
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            ".",
            ":(exclude)experiments/runs/**",
            ":(exclude).codex_campaign/RUN_LEDGER.jsonl",
            ":(exclude).codex_campaign/RESULT_INDEX.csv",
            ":(exclude).codex_campaign/r1/RESULT_INDEX.csv",
            ":(exclude).codex_campaign/r1/STATE.md",
            ":(exclude).codex_campaign/r1/MATURITY.json",
            ":(exclude).codex_campaign/r1/CLAIMS.md",
            ":(exclude).codex_campaign/r1/FAILURES.md",
            ":(exclude).codex_campaign/r1/HANDOFF.md",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("counted R1 runs require a clean committed worktree")


def seed_everything(seed: int) -> torch.Generator:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed)


def load_audio(
    data_config: dict[str, Any], split: str
) -> tuple[np.ndarray, np.ndarray]:
    entry = data_config["splits"][split]
    input_path = ROOT / entry["input_path"]
    target_path = ROOT / entry["target_path"]
    if sha256(input_path) != entry["input_sha256"]:
        raise RuntimeError(f"input checksum mismatch for {split}")
    if sha256(target_path) != entry["target_sha256"]:
        raise RuntimeError(f"target checksum mismatch for {split}")
    signal, input_rate = sf.read(input_path, dtype="float32")
    target, target_rate = sf.read(target_path, dtype="float32")
    if input_rate != 44100 or target_rate != 44100 or signal.shape != target.shape:
        raise RuntimeError(f"invalid Wright pair for {split}")
    if (
        signal.ndim != 1
        or not np.isfinite(signal).all()
        or not np.isfinite(target).all()
    ):
        raise RuntimeError(f"invalid audio values for {split}")
    return signal, target


def environment(device: torch.device) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_device": str(device),
        "precision": "float32",
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def run_preflight(config: dict[str, Any], data_config: dict[str, Any]) -> None:
    generator = seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal, target = load_audio(data_config, "train")
    frames = frame_audio(signal, int(config["segment_samples"]))[:4]
    target_frames = frame_audio(target, int(config["segment_samples"]))[:4]
    model = WrightLSTM(hidden_size=64).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    loss, updates = train_epoch(
        model,
        frames,
        target_frames,
        WrightLoss(),
        optimizer,
        batch_size=2,
        warmup_samples=int(config["warmup_samples"]),
        tbptt_samples=int(config["tbptt_samples"]),
        device=device,
        generator=generator,
    )
    prediction = predict_streaming(
        model,
        signal[:32768],
        device=device,
        chunk_samples=4093,
    )
    result = {
        "schema_version": 1,
        "status": "passed",
        "device": str(device),
        "loss": loss,
        "optimizer_updates": updates,
        "finite_prediction": bool(np.isfinite(prediction).all()),
        "samples": int(prediction.size),
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if (
        PREFLIGHT_PATH.exists()
        and PREFLIGHT_PATH.read_text(encoding="utf-8") != serialized
    ):
        raise RuntimeError("refusing to overwrite divergent Wright preflight")
    PREFLIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


def append_indexes(
    run_id: str, seed: int, status: str, primary: str, relative_path: str
) -> None:
    with GLOBAL_INDEX_PATH.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerow(
            [
                run_id,
                "R1_COMPETENCE",
                "W0",
                "electro_harmonix_big_muff",
                seed,
                status,
                primary,
                relative_path,
            ]
        )
    with R1_INDEX_PATH.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerow(
            [
                run_id,
                "competence",
                "bigmuff",
                "wright_lstm64",
                seed,
                status,
                f"{relative_path}/metrics.json",
            ]
        )


def main() -> None:
    args = parse_args()
    config_bytes = CONFIG_PATH.read_bytes()
    data_config_bytes = DATA_CONFIG_PATH.read_bytes()
    manifest_bytes = MANIFEST_PATH.read_bytes()
    split_bytes = SPLIT_PATH.read_bytes()
    config = yaml.safe_load(config_bytes)
    data_config = yaml.safe_load(data_config_bytes)
    if args.seed not in config["seeds"]:
        raise ValueError("seed is outside the locked competence protocol")
    if args.preflight:
        if args.run_id is not None or args.resume:
            raise ValueError("preflight does not accept run-id or resume")
        run_preflight(config, data_config)
        return
    if not args.run_id:
        raise ValueError("a counted competence run requires --run-id")
    expected_id = f"r1_competence_bigmuff_wright_lstm64_wright_seed{args.seed}_v1"
    if args.run_id != expected_id:
        raise ValueError(f"run-id must be {expected_id}")

    commit = git_commit()
    run_dir = ROOT / "experiments/runs" / args.run_id
    if args.resume:
        if not run_dir.is_dir():
            raise RuntimeError("resume run directory is absent")
        recorded_commit = (
            (run_dir / "git-commit.txt").read_text(encoding="utf-8").strip()
        )
        if commit != recorded_commit:
            raise RuntimeError("resume commit differs from the recorded run commit")
    else:
        require_clean_worktree()
        run_dir.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "predictions", "figures"):
            (run_dir / name).mkdir()

    resolved = {
        "campaign": config,
        "data": data_config,
        "run_id": args.run_id,
        "seed": args.seed,
    }
    resolved_bytes = yaml.safe_dump(resolved, sort_keys=True).encode()
    config_hash = hashlib.sha256(resolved_bytes).hexdigest()
    data_hash = combined_hash(manifest_bytes, split_bytes)
    started_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.perf_counter()
    status = "failed"
    failure_reason = ""
    primary = ""

    if not args.resume:
        (run_dir / "config-resolved.yaml").write_bytes(resolved_bytes)
        (run_dir / "command.txt").write_text(
            " ".join(sys.argv) + "\n", encoding="utf-8"
        )
        (run_dir / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
        (run_dir / "seed.txt").write_text(f"{args.seed}\n", encoding="utf-8")
        (run_dir / "environment.json").write_text(
            json.dumps(environment(torch.device("cuda")), indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(MANIFEST_PATH, run_dir / "dataset-manifest.json")
        shutil.copyfile(SPLIT_PATH, run_dir / "split-manifest.json")
        (run_dir / "status.json").write_text(
            json.dumps({"status": "running", "started_at": started_iso}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    def log(message: str) -> None:
        print(message, flush=True)
        with (run_dir / "stdout.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("the locked competence training requires CUDA")
        generator = seed_everything(args.seed)
        device = torch.device("cuda")
        train_input, train_target = load_audio(data_config, "train")
        validation_input, validation_target = load_audio(data_config, "validation")
        test_input, test_target = load_audio(data_config, "test")
        frame_samples = int(config["segment_samples"])
        input_frames = frame_audio(train_input, frame_samples)
        target_frames = frame_audio(train_target, frame_samples)
        model = WrightLSTM(hidden_size=64).to(device)
        loss_function = WrightLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["optimizer"]["learning_rate"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config["scheduler"]["factor"]),
            patience=int(config["scheduler"]["patience_validations"]),
        )
        history: list[dict[str, Any]] = []
        best_validation_loss = float("inf")
        best_epoch = 0
        patience = 0
        optimizer_updates = 0
        first_epoch = 1
        if args.resume:
            resume = torch.load(
                run_dir / "checkpoints/last-state.pt",
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(resume["model"])
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            generator.set_state(resume["generator_state"])
            history = resume["history"]
            best_validation_loss = float(resume["best_validation_loss"])
            best_epoch = int(resume["best_epoch"])
            patience = int(resume["patience"])
            optimizer_updates = int(resume["optimizer_updates"])
            first_epoch = int(resume["epoch"]) + 1
            log(f"resuming at epoch={first_epoch}")

        maximum_epochs = int(config["maximum_epochs"])
        validation_frequency = int(config["validation_frequency_epochs"])
        early_patience = int(config["early_stopping"]["patience_validations"])
        for epoch in range(first_epoch, maximum_epochs + 1):
            epoch_started = time.perf_counter()
            training_loss, updates = train_epoch(
                model,
                input_frames,
                target_frames,
                loss_function,
                optimizer,
                batch_size=int(config["batch_size"]),
                warmup_samples=int(config["warmup_samples"]),
                tbptt_samples=int(config["tbptt_samples"]),
                device=device,
                generator=generator,
            )
            optimizer_updates += updates
            record: dict[str, Any] = {
                "epoch": epoch,
                "training_loss": training_loss,
                "optimizer_updates": optimizer_updates,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "wall_seconds": time.perf_counter() - epoch_started,
            }
            if epoch % validation_frequency == 0:
                validation_prediction = predict_streaming(
                    model,
                    validation_input,
                    device=device,
                    chunk_samples=32768,
                )
                validation_loss, validation_esr = evaluate_prediction(
                    validation_prediction, validation_target, loss_function
                )
                scheduler.step(validation_loss)
                record["validation_wright_loss"] = validation_loss
                record["validation_esr"] = validation_esr
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_epoch = epoch
                    patience = 0
                    torch.save(
                        model.state_dict(), run_dir / "checkpoints/best-model.pt"
                    )
                else:
                    patience += 1
                history.append(record)
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "generator_state": generator.get_state(),
                        "history": history,
                        "best_validation_loss": best_validation_loss,
                        "best_epoch": best_epoch,
                        "patience": patience,
                        "optimizer_updates": optimizer_updates,
                    },
                    run_dir / "checkpoints/last-state.pt",
                )
                (run_dir / "training-history.json").write_text(
                    json.dumps(history, indent=2) + "\n", encoding="utf-8"
                )
                log(
                    f"epoch={epoch} train={training_loss:.9g} "
                    f"val={validation_loss:.9g} esr={validation_esr:.9g} "
                    f"patience={patience}"
                )
                if patience > early_patience:
                    log(f"early stopping at epoch={epoch}")
                    break
            else:
                history.append(record)

        if best_epoch < 1:
            raise RuntimeError("competence training produced no validation checkpoint")
        model.load_state_dict(
            torch.load(
                run_dir / "checkpoints/best-model.pt",
                map_location=device,
                weights_only=True,
            )
        )
        test_prediction = predict_streaming(
            model, test_input, device=device, chunk_samples=32768
        )
        alternate_samples = min(test_input.size, 262144)
        alternate = predict_streaming(
            model,
            test_input[:alternate_samples],
            device=device,
            chunk_samples=4093,
        )
        block_difference = float(
            np.max(np.abs(alternate - test_prediction[:alternate_samples]), initial=0.0)
        )
        if block_difference > 2.0e-5:
            raise RuntimeError(f"post-training block parity failed: {block_difference}")
        test_loss, test_esr = evaluate_prediction(
            test_prediction, test_target, loss_function
        )
        metrics = {
            "best_epoch": best_epoch,
            "best_validation_wright_loss": best_validation_loss,
            "optimizer_updates": optimizer_updates,
            "epochs_completed": history[-1]["epoch"],
            "test_wright_loss": test_loss,
            "test_esr": test_esr,
            "test_time_metrics": time_metrics(test_prediction, test_target),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "latency_samples": 0,
            "sample_rate": 44100,
            "checks": {
                "finite_prediction": bool(np.isfinite(test_prediction).all()),
                "block_parity_max_abs": block_difference,
                "block_parity_prefix_samples": alternate_samples,
            },
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        sf.write(
            run_dir / "predictions/test-output.wav",
            test_prediction,
            44100,
            subtype="FLOAT",
        )
        checkpoint_path = run_dir / "checkpoints/best-model.pt"
        (run_dir / "checkpoints/index.json").write_text(
            json.dumps(
                {
                    "path": "best-model.pt",
                    "sha256": sha256(checkpoint_path),
                    "selection": "lowest_validation_wright_loss",
                    "epoch": best_epoch,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        status = "completed"
        primary = f"test_esr={test_esr:.9g}"
        log(f"completed {args.run_id}: {primary}")
    except Exception as error:
        failure_reason = f"{type(error).__name__}: {error}"
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        raise
    finally:
        elapsed = time.perf_counter() - started
        finished_iso = datetime.now().astimezone().isoformat(timespec="seconds")
        if not (run_dir / "metrics.json").exists():
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {"status": "unavailable", "reason": failure_reason}, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
        if not (run_dir / "checkpoints/index.json").exists():
            (run_dir / "checkpoints/index.json").write_text(
                json.dumps({"status": "unavailable"}, indent=2) + "\n",
                encoding="utf-8",
            )
        (run_dir / "timings.json").write_text(
            json.dumps(
                {
                    "wall_seconds": elapsed,
                    "completed": status == "completed",
                    "gpu_peak_memory_bytes": (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else None
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "started_at": started_iso,
                    "finished_at": finished_iso,
                    "failure_reason": failure_reason,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        relative_path = str(run_dir.relative_to(ROOT))
        append_run(
            LEDGER_PATH,
            {
                "date": finished_iso,
                "run_id": args.run_id,
                "phase": "R1_COMPETENCE",
                "model": "W0_Wright_LSTM64",
                "device": "electro_harmonix_big_muff_native_44100",
                "seed": args.seed,
                "commit": commit,
                "config_sha256": config_hash,
                "data_sha256": data_hash,
                "status": status,
                "failure_reason": failure_reason,
                "results_path": relative_path,
            },
        )
        append_indexes(args.run_id, args.seed, status, primary, relative_path)


if __name__ == "__main__":
    main()
