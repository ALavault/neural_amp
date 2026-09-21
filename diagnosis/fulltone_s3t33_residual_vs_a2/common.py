"""Shared read-only helpers: audio, checkpoints, replayed training windows. CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from fssr_nam.training.m4 import causal_predict, model_factory, training_prediction

ROOT = Path("/fastdata/lavaulta/neural_amp")
DATA = ROOT / "datasets/raw/internal_m4/fulltone_full_drive_2"
RUNS = ROOT / "experiments/runs"
OUT = Path(__file__).resolve().parent
FS = 48_000
CONTEXT = 6346
OUTPUT = 8192
BATCH = 2
STEPS = 200
S3_RUNS = {s: f"m4_memory_fulltone_s3t33_seed{s}_v1" for s in range(3)}
A2_RUNS = {s: f"m4_fulltone_b0_seed{s}_v1" for s in range(3)}
S3_17_RUNS = {s: f"m4_fulltone_s3_seed{s}_v1" for s in range(3)}
BANDS = ((0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 24000))

torch.set_num_threads(8)


def esr(p, t):
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return float(np.sum((p - t) ** 2) / np.sum(t**2))


def audio(split: str):
    x, _ = sf.read(DATA / f"{split}_input.wav", dtype="float32")
    y, _ = sf.read(DATA / f"{split}_target.wav", dtype="float32")
    return x, y


def load_model(run_id: str):
    run_dir = RUNS / run_id
    resolved = yaml.safe_load((run_dir / "config-resolved.yaml").read_text())
    code = resolved["model"]
    model_config = resolved.get("fssr_model") or yaml.safe_load(
        (ROOT / "configs/training/m3_synthetic.yaml").read_text()
    )["model"]
    model = model_factory(code, root=ROOT, model_config=model_config)
    model.load_state_dict(
        torch.load(run_dir / "checkpoints/model-state.pt", map_location="cpu", weights_only=True)
    )
    return model.eval(), code, resolved


def saved_test_prediction(run_id: str) -> np.ndarray:
    return np.fromfile(RUNS / run_id / "predictions/test_prediction.f32", "<f4")


def replay_starts(seed: int, train_size: int) -> list[np.ndarray]:
    """Exact per-step window starts of scripts/campaigns/run_m4_smoke.py for this seed."""
    rng = np.random.default_rng(seed)
    return [rng.integers(CONTEXT, train_size - OUTPUT, size=BATCH) for _ in range(STEPS)]


def windows_for_steps(train_x, train_y, starts, first: int, last: int):
    """Input windows (context + output) and output targets for steps first..last (1-based)."""
    xs, ys, ss = [], [], []
    for step, batch in enumerate(starts, start=1):
        if first <= step <= last:
            for s in batch:
                xs.append(train_x[s - CONTEXT : s + OUTPUT])
                ys.append(train_y[s : s + OUTPUT])
                ss.append(int(s))
    return np.stack(xs), np.stack(ys), np.array(ss)


def predict_windows(model, code: str, windows: np.ndarray, chunk: int = 20) -> np.ndarray:
    out = []
    with torch.inference_mode():
        for i in range(0, len(windows), chunk):
            tensor = torch.from_numpy(windows[i : i + chunk])
            out.append(training_prediction(model, code, tensor, OUTPUT).numpy())
    return np.concatenate(out)


def predict_file(model, code: str, signal: np.ndarray) -> np.ndarray:
    return causal_predict(
        model, code, signal, device=torch.device("cpu"), context_samples=CONTEXT, block_samples=OUTPUT
    )


def band_table(error: np.ndarray, target: np.ndarray) -> dict:
    """Share of ESR carried by each band (error spectrum), plus target energy share."""
    e = np.fft.rfft(np.asarray(error, dtype=np.float64))
    t = np.fft.rfft(np.asarray(target, dtype=np.float64))
    f = np.fft.rfftfreq(target.size, d=1.0 / FS)
    pe, pt = np.abs(e) ** 2, np.abs(t) ** 2
    total = esr(target + error, target)
    rows = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        rows[f"{lo}-{hi}"] = {
            "esr_share": float(pe[m].sum() / pe.sum() * total),
            "target_energy_share": float(pt[m].sum() / pt.sum()),
        }
    rows["total"] = total
    return rows


def causal_envelope(x: np.ndarray, tau_ms: float = 10.0) -> np.ndarray:
    """One-pole causal RMS envelope of the input."""
    from scipy.signal import lfilter

    alpha = 1.0 - np.exp(-1.0 / (FS * tau_ms * 1e-3))
    power = lfilter([alpha], [1.0, -(1.0 - alpha)], np.asarray(x, dtype=np.float64) ** 2)
    return np.sqrt(power)


def dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n")
