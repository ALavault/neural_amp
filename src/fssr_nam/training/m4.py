"""Common model construction and causal inference for the M4 smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from nam.models.factory import init as init_nam_model

from fssr_nam.models.recurrent import CausalGRUBaseline
from fssr_nam.training.m3 import model_factory as fssr_model_factory


def model_factory(code: str, *, root: Path, model_config: dict):
    if code == "B0":
        path = (
            root / "third_party/neural-amp-modeler/nam/train/_resources/"
            "config_model_packed.json"
        )
        packed_config = json.loads(path.read_text(encoding="utf-8"))["net"]
        packed = init_nam_model(
            packed_config["name"], kwargs={"config": packed_config["config"]}
        )
        return packed.extract_submodel(1)
    if code == "B2":
        return CausalGRUBaseline(hidden_size=62)
    if code in {"S3", "S4"}:
        return fssr_model_factory(code, model_config)
    raise ValueError(f"unknown M4 model code: {code}")


def latency_samples(model) -> int:
    return int(getattr(model, "latency_samples", 0))


def training_prediction(
    model, code: str, windows: torch.Tensor, output_samples: int
) -> torch.Tensor:
    """Predict the output tail after a common causal warm-up context."""
    if code == "B0":
        prediction = model(windows, pad_start=False)
    else:
        prediction = model(windows)[..., -output_samples:]
    if prediction.shape != (*windows.shape[:-1], output_samples):
        raise RuntimeError(f"unexpected {code} training output shape")
    return prediction


def delay_target(signal: np.ndarray, delay: int) -> np.ndarray:
    if delay < 0:
        raise ValueError("declared latency must be non-negative")
    target = np.asarray(signal, dtype=np.float32)
    if delay == 0:
        return target.copy()
    delayed = np.zeros_like(target)
    delayed[delay:] = target[:-delay]
    return delayed


def causal_predict(
    model,
    code: str,
    signal: np.ndarray,
    *,
    device: torch.device,
    context_samples: int,
    block_samples: int,
) -> np.ndarray:
    """Run causal block inference with model state reset at file boundaries."""
    source = np.asarray(signal, dtype=np.float32)
    outputs: list[np.ndarray] = []
    model.eval()
    reset = getattr(model, "reset_state", None)
    if reset is not None:
        reset()
    with torch.inference_mode():
        for start in range(0, source.size, block_samples):
            stop = min(source.size, start + block_samples)
            if code == "B0":
                left = max(0, start - context_samples)
                chunk = source[left:stop]
                missing = context_samples - (start - left)
                if missing:
                    chunk = np.pad(chunk, (missing, 0))
                tensor = torch.from_numpy(chunk).to(device)
                output = model(tensor, pad_start=False)
            else:
                tensor = torch.from_numpy(source[start:stop]).to(device)
                output = model.stream(tensor)
            outputs.append(output.detach().cpu().numpy())
    prediction = np.concatenate(outputs).astype(np.float32, copy=False)
    if prediction.shape != source.shape or not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"invalid {code} causal prediction")
    return prediction
