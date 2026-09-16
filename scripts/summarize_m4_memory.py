#!/usr/bin/env python3
"""Evaluate the preregistered M4 linear-memory diagnostic against its thresholds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from fssr_nam.training.m4 import (
    delay_target,
    latency_samples,
    model_factory,
    training_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/training/m4_memory.yaml"
MODEL_CONFIG_PATH = ROOT / "configs/training/m3_synthetic.yaml"
MANIFEST_PATH = ROOT / "datasets/manifests/m4_internal.json"
OUTPUT_DIR = ROOT / "experiments/summaries/m4_memory"
REPORT_PATH = ROOT / "reports/M4_MEMORY.md"
LABELS = {
    "fulltone_full_drive_2": "fulltone",
    "electro_harmonix_big_muff": "bigmuff",
}
SAMPLE_RATE = 48_000
WINDOW_BATCH = 20


def esr(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sum(np.square(prediction - target), dtype=np.float64)
    denominator = np.sum(np.square(target), dtype=np.float64)
    return float(numerator / max(denominator, np.finfo(float).eps))


def band_error(prediction, target, low: float, high: float) -> float:
    """Share of the ESR carried by the error spectrum between low and high Hz."""
    spectrum = np.abs(np.fft.rfft((prediction - target).astype(np.float64))) ** 2
    frequency = np.fft.rfftfreq(target.size, d=1.0 / SAMPLE_RATE)
    mask = (frequency >= low) & (frequency < high)
    return float(np.sum(spectrum[mask]) / np.sum(spectrum) * esr(prediction, target))


def device_audio(manifest: dict, device: str, split: str):
    entry = next(
        item
        for item in manifest["files"]
        if item["device"] == device and item["split"] == split
    )
    x, _ = sf.read(ROOT / entry["input_path"], dtype="float32")
    y, _ = sf.read(ROOT / entry["target_path"], dtype="float32")
    return x, y


def load_run(run_id: str) -> dict:
    run_dir = ROOT / "experiments/runs" / run_id
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise RuntimeError(f"run is not completed: {run_id}")
    return {
        "run_id": run_id,
        "dir": run_dir,
        "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
        "resolved": yaml.safe_load(
            (run_dir / "config-resolved.yaml").read_text(encoding="utf-8")
        ),
    }


def train_window_esr(run: dict, manifest: dict, device: torch.device) -> float:
    """ESR of the selected checkpoint on the exact training windows of steps a..b."""
    resolved = run["resolved"]
    campaign = resolved["campaign"]
    code = resolved["model"]
    model_config = (
        resolved.get("fssr_model")
        or yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))["model"]
    )
    model = model_factory(code, root=ROOT, model_config=model_config)
    model.load_state_dict(
        torch.load(
            run["dir"] / "checkpoints/model-state.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model = model.to(device).eval()
    train_x, train_y = device_audio(manifest, resolved["device"], "train")
    train_target = delay_target(train_y, latency_samples(model))
    context = int(campaign["context_samples"])
    output = int(campaign["output_samples"])
    batch = int(campaign["batch_size"])
    steps = int(resolved["execution"]["optimizer_steps"])
    first, last = (int(v) for v in CONFIG["decision"]["train_window_steps"])
    rng = np.random.default_rng(int(resolved["seed"]))
    windows, targets = [], []
    for step in range(1, steps + 1):
        starts = rng.integers(context, train_x.size - output, size=batch)
        if first <= step <= last:
            windows.extend(train_x[s - context : s + output] for s in starts)
            targets.extend(train_target[s : s + output] for s in starts)
    error = 0.0
    energy = 0.0
    with torch.inference_mode():
        for index in range(0, len(windows), WINDOW_BATCH):
            chunk = torch.from_numpy(np.stack(windows[index : index + WINDOW_BATCH]))
            target = np.stack(targets[index : index + WINDOW_BATCH])
            prediction = (
                training_prediction(model, code, chunk.to(device), output).cpu().numpy()
            )
            error += float(np.sum(np.square(prediction - target), dtype=np.float64))
            energy += float(np.sum(np.square(target), dtype=np.float64))
    return error / energy


def evaluate_run(run_id: str, manifest: dict, device: torch.device) -> dict:
    run = load_run(run_id)
    resolved = run["resolved"]
    _, test_y = device_audio(manifest, resolved["device"], "test")
    prediction = np.fromfile(run["dir"] / "predictions/test_prediction.f32", "<f4")
    model = model_factory(
        resolved["model"],
        root=ROOT,
        model_config=resolved.get("fssr_model")
        or yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))["model"],
    )
    target = delay_target(test_y, latency_samples(model))[: prediction.size]
    low, high = CONFIG["decision"]["band_hz"]
    test = run["metrics"]["test"]
    train_esr = train_window_esr(run, manifest, device)
    fssr = resolved.get("fssr_model") or {}
    return {
        "run_id": run_id,
        "phase": (
            "M4_MEMORY"
            if resolved.get("memory")
            else "M4_RECOVERY"
            if resolved.get("recovery")
            else "M4"
        ),
        "model": resolved["model"],
        "device": resolved["device"],
        "seed": int(resolved["seed"]),
        "taps": fssr.get("taps"),
        "parameters": run["metrics"]["parameters"],
        "best_step": run["metrics"]["best_step"],
        "test_esr": float(test["esr"]),
        "test_esr_recomputed": esr(prediction, target),
        "gain_error": float(test["gain_error"]),
        "residual_energy_ratio": float(test["residual_energy_ratio"]),
        "band_error": band_error(prediction, target, low, high),
        "train_window_esr": train_esr,
        "transfer_loss": float(test["esr"]) - train_esr,
    }


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def decide(rows: list[dict]) -> dict:
    decision = CONFIG["decision"]
    model = decision["primary_model"]
    primary = decision["primary_device"]
    regression = decision["regression"]
    memory = [r for r in rows if r["phase"] == "M4_MEMORY" and r["model"] == model]
    frozen = [r for r in rows if r["phase"] == "M4" and r["model"] == model]
    a2 = [r for r in rows if r["phase"] == "M4" and r["model"] == "B0"]
    result: dict = {"primary_model": model, "primary_device": primary}
    fulltone = [r for r in memory if r["device"] == primary]
    if not fulltone:
        result["verdict"] = "no_memory_runs"
        return result
    med = {
        key: median([r[key] for r in fulltone])
        for key in ("test_esr", "band_error", "train_window_esr", "transfer_loss")
    }
    confirmed = decision["confirmed"]
    if (
        med["test_esr"] <= confirmed["test_esr_max"]
        and med["band_error"] <= confirmed["band_error_max"]
        and med["transfer_loss"] <= confirmed["transfer_loss_max"]
    ):
        verdict = "confirmed"
    elif med["test_esr"] > decision["refuted"]["test_esr_min"]:
        verdict = "refuted"
    else:
        verdict = "partial"
    discriminant = "not_needed"
    if verdict != "confirmed":
        limits = decision["discriminant"]
        if med["train_window_esr"] <= limits["train_window_esr_transfer_limited_max"]:
            discriminant = "transfer_limited"
        elif (
            med["train_window_esr"]
            >= limits["train_window_esr_optimisation_limited_min"]
        ):
            discriminant = "optimisation_limited"
        else:
            discriminant = "mixed"
    result.update(
        {
            "seeds_present": sorted(r["seed"] for r in fulltone),
            "medians": med,
            "seed0": next((r for r in fulltone if r["seed"] == 0), None),
            "verdict": verdict,
            "discriminant": discriminant,
        }
    )
    frozen_primary = [r["test_esr"] for r in frozen if r["device"] == primary]
    a2_primary = [r["test_esr"] for r in a2 if r["device"] == primary]
    if frozen_primary and a2_primary:
        gap = median(frozen_primary) - median(a2_primary)
        result["gap_closure_fraction_vs_a2_median"] = (
            median(frozen_primary) - med["test_esr"]
        ) / gap
    control = regression["device"]
    control_new = [r["test_esr"] for r in memory if r["device"] == control]
    control_ref = [r["test_esr"] for r in frozen if r["device"] == control]
    if control_new and control_ref:
        increase = median(control_new) - median(control_ref)
        result["regression_control"] = {
            "device": control,
            "median_test_esr_new": median(control_new),
            "median_test_esr_reference": median(control_ref),
            "median_increase": increase,
            "passes": increase <= regression["median_test_esr_max_increase"],
        }
    return result


def default_run_ids() -> list[str]:
    ids = []
    for device in CONFIG["devices"]:
        for model in CONFIG["models"]:
            for seed in CONFIG["seeds"]:
                run_id = (
                    f"m4_memory_{LABELS[device]}_{model.lower()}"
                    f"t{CONFIG['taps']}_seed{seed}_v1"
                )
                if (ROOT / "experiments/runs" / run_id).exists():
                    ids.append(run_id)
        for model in ("S3", "S4", "B0"):
            ids.extend(CONFIG["references"][device][model])
    return ids


def report(rows: list[dict], decision: dict) -> str:
    lines = [
        "# M4 linear-memory diagnostic (S3, 33-tap FIRs)",
        "",
        "| Run | Model | Device | Seed | Taps | Params | Test ESR | "
        "100-300 Hz | Train-window ESR | Transfer | Residual ratio |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_id']} | {r['model']} | {LABELS[r['device']]} | {r['seed']} | "
            f"{r['taps'] if r['taps'] is not None else '-'} | {r['parameters']} | "
            f"{r['test_esr']:.4f} | {r['band_error']:.4f} | "
            f"{r['train_window_esr']:.4f} | {r['transfer_loss']:+.4f} | "
            f"{r['residual_energy_ratio']:.2e} |"
        )
    lines += ["", "```json", json.dumps(decision, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [
        evaluate_run(run_id, manifest, device)
        for run_id in (args.run_id or default_run_ids())
    ]
    decision = decide(rows)
    summary = {
        "schema_version": 1,
        "config": CONFIG,
        "rows": rows,
        "decision": decision,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(rows, decision), encoding="utf-8")
    print(report(rows, decision))


CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
