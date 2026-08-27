#!/usr/bin/env python3
"""Generate local, license-restricted listening examples from M4 seed 0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from fssr_nam.training.m4 import delay_target, model_factory

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "experiments/audio_examples/m4_fulltone_seed0"
INPUT_PATH = ROOT / "datasets/raw/internal_m4/fulltone_full_drive_2/test_input.wav"
TARGET_PATH = ROOT / "datasets/raw/internal_m4/fulltone_full_drive_2/test_target.wav"
RUNS = {
    "A2": "m4_fulltone_b0_seed0_v1",
    "B2": "m4_fulltone_b2_seed0_v1",
    "S3": "m4_fulltone_s3_seed0_v1",
    "S4": "m4_fulltone_s4_seed0_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prediction(run_id: str, sample_count: int) -> np.ndarray:
    path = ROOT / "experiments/runs" / run_id / "predictions/test_prediction.f32"
    prediction = np.fromfile(path, dtype=np.float32)
    if prediction.size != sample_count:
        raise RuntimeError(f"unexpected prediction size for {run_id}")
    return prediction


def streamed_residual(code: str, run_id: str, signal: np.ndarray) -> np.ndarray:
    config = yaml.safe_load(
        (ROOT / "configs/training/m3_synthetic.yaml").read_text(encoding="utf-8")
    )["model"]
    model = model_factory(code, root=ROOT, model_config=config).eval()
    checkpoint = ROOT / "experiments/runs" / run_id / "checkpoints/model-state.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.reset_state()
    residuals = []
    with torch.inference_mode():
        for start in range(0, signal.size, 8192):
            chunk = torch.from_numpy(signal[start : start + 8192])
            _, _, residual = model.stream_components(chunk)
            residuals.append(residual.numpy())
    return np.concatenate(residuals)


def write_audio(name: str, signal: np.ndarray, sample_rate: int) -> dict[str, object]:
    path = OUTPUT_DIR / f"{name}.wav"
    sf.write(path, signal, sample_rate, subtype="PCM_24")
    return {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "peak": float(np.max(np.abs(signal), initial=0.0)),
    }


def main() -> None:
    input_signal, sample_rate = sf.read(INPUT_PATH, dtype="float32")
    target, target_rate = sf.read(TARGET_PATH, dtype="float32")
    if sample_rate != 48_000 or target_rate != sample_rate:
        raise RuntimeError("unexpected listening-example sample rate")
    predictions = {
        label: read_prediction(run_id, target.size) for label, run_id in RUNS.items()
    }
    residuals = {
        code: streamed_residual(code, RUNS[code], input_signal) for code in ("S3", "S4")
    }
    start = 5 * sample_rate
    stop = 15 * sample_rate
    playback = {
        "input": input_signal[start:stop],
        "reference": target[start:stop],
        **{label.lower(): signal[start:stop] for label, signal in predictions.items()},
        "s3_residual": residuals["S3"][start:stop],
        "s4_residual": residuals["S4"][start:stop],
    }
    common_peak = max(float(np.max(np.abs(signal))) for signal in playback.values())
    common_gain = 10 ** (-1.0 / 20.0) / common_peak
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        write_audio(name, signal * common_gain, sample_rate)
        for name, signal in playback.items()
    ]
    for label, prediction in predictions.items():
        latency = 16 if label == "S4" else 0
        reference = delay_target(target, latency)
        error = 4.0 * (prediction[start:stop] - reference[start:stop])
        error_gain = 10 ** (-1.0 / 20.0) / max(
            float(np.max(np.abs(error))), np.finfo(float).eps
        )
        entry = write_audio(
            f"{label.lower()}_error_x4", error * error_gain, sample_rate
        )
        entry["independent_error_playback_gain"] = error_gain
        files.append(entry)
    manifest = {
        "schema_version": 1,
        "tier": "INTERNAL_DEV",
        "source_device": "Fulltone Full Drive 2",
        "source_setting": "V100_T050_O050_B000",
        "source_license": "CC-BY-NC-4.0",
        "redistribution_in_final_archive": False,
        "sample_rate": sample_rate,
        "start_sample": start,
        "stop_sample": stop,
        "duration_seconds": (stop - start) / sample_rate,
        "common_playback_gain": common_gain,
        "scientific_metrics_use_playback_normalization": False,
        "error_amplification": 4.0,
        "model_runs": RUNS,
        "files": files,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
