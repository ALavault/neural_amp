#!/usr/bin/env python3
"""Validate the compatibility path against the released Wright LSTM-64."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from fssr_nam.losses import WrightLoss
from fssr_nam.metrics.time import time_metrics
from fssr_nam.models import WrightLSTM
from fssr_nam.training.wright import evaluate_prediction, predict_streaming

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/r1_wright_bigmuff_native.yaml"
OUTPUT_PATH = ROOT / "experiments/summaries/r1_wright_reference/validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    reference_path = ROOT / config["reference_model"]["path"]
    if sha256(reference_path) != config["reference_model"]["sha256"]:
        raise RuntimeError("released Wright model checksum mismatch")
    test = config["splits"]["test"]
    input_path = ROOT / test["input_path"]
    target_path = ROOT / test["target_path"]
    if sha256(input_path) != test["input_sha256"]:
        raise RuntimeError("Wright test input checksum mismatch")
    if sha256(target_path) != test["target_sha256"]:
        raise RuntimeError("Wright test target checksum mismatch")
    signal, input_rate = sf.read(input_path, dtype="float32")
    target, target_rate = sf.read(target_path, dtype="float32")
    if input_rate != 44100 or target_rate != 44100 or signal.shape != target.shape:
        raise RuntimeError("invalid native-rate Wright test pair")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WrightLSTM.from_wright_json(reference_path).to(device)
    prediction = predict_streaming(model, signal, device=device, chunk_samples=32768)
    prefix_samples = min(signal.size, 262144)
    alternate = predict_streaming(
        model,
        signal[:prefix_samples],
        device=device,
        chunk_samples=4093,
    )
    difference = float(
        np.max(np.abs(alternate - prediction[:prefix_samples]), initial=0.0)
    )
    if difference > 2.0e-5:
        raise RuntimeError(f"Wright block parity failed: {difference:.9g}")
    reference_loss, reference_esr = evaluate_prediction(
        prediction, target, WrightLoss()
    )
    if not 0.06 <= reference_esr <= 0.09:
        raise RuntimeError(f"unexpected released-model ESR: {reference_esr:.9g}")
    result = {
        "schema_version": 1,
        "source_commit": config["source_commit"],
        "reference_model_sha256": config["reference_model"]["sha256"],
        "sample_rate": input_rate,
        "samples": int(signal.size),
        "device": str(device),
        "reference_wright_loss": reference_loss,
        "reference_test_esr": reference_esr,
        "time_metrics": time_metrics(prediction, target),
        "block_parity_prefix_samples": prefix_samples,
        "block_parity_max_abs": difference,
        "finite_prediction": bool(np.isfinite(prediction).all()),
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if OUTPUT_PATH.exists() and OUTPUT_PATH.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("refusing to overwrite divergent Wright validation")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
