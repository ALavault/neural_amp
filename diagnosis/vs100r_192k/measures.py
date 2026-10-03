#!/usr/bin/env python3
"""Measures of DIAGNOSIS_vs100r_192k.md §3 (H1, H3 to H6). Read-only, CPU.

H1a scalar oracle gain per take; H1b take 16 with the controls of the neighbouring seen
pairs; H3 relative fall of the validation loss over the last 2000 steps; H4 share of
B192's MR-STFT (x4 sizes) coming from bins above 24 kHz, on the test takes at 192 kHz;
H5 chunked inference (144000 samples, state reset per chunk) against continuous; H6
oracle fractional delay per take. Writes measures.json.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import product_vs100r_eval as ev  # noqa: E402

# Optional arguments: A48 run, B192 run, output name (default: the seed-42 pair).
RUNS = {"A48": "vs100r_gc_film_joint_s925_r2", "B192": "vs100r192_film_s42"}
OUTPUT = "measures.json"
if len(sys.argv) == 4:
    RUNS, OUTPUT = {"A48": sys.argv[1], "B192": sys.argv[2]}, sys.argv[3]
UP, CHUNK = 4, 144_000
NEIGHBOURS = {
    "true (2,5 ; 7,5)": (0.25, 0.75),
    "(2,5 ; 5)": (0.25, 0.5),
    "(5 ; 7,5)": (0.5, 0.75),
}
FFT = [4096, 8192, 2048]
HOP = [480, 960, 200]
WIN = [2400, 4800, 960]


def esr(y: np.ndarray, r: np.ndarray) -> float:
    return float(((r - y) ** 2).sum() / (r**2).sum())


def run(model, arm: str, loop: np.ndarray, controls, chunk: int | None = None):
    x = loop if arm == "B192" else resample_poly(loop, 1, UP)
    c = torch.tensor(controls)
    if chunk is None:
        y = ev.run(model, x, c)
    else:
        y = np.concatenate(
            [ev.run(model, x[i : i + chunk], c) for i in range(0, len(x), chunk)]
        )
    return y if arm == "A48" else resample_poly(y, 1, UP)


def shift(x: np.ndarray, delay: float) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    return np.fft.irfft(
        spectrum * np.exp(-2j * np.pi * np.fft.rfftfreq(len(x)) * delay), len(x)
    )


def oracle_delay(y: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    best = min((esr(ev.scored(shift(y, d)), r), d) for d in np.arange(-2, 2.0001, 0.05))
    return best


def val_fall(run_id: str) -> float:
    rows = [
        r
        for r in csv.DictReader(
            open(ROOT / f"demo/runs/nablafx_{run_id}/logs/metrics.csv")
        )
        if r.get("loss/val/tot")
    ]
    pts = [(int(r["step"]), float(r["loss/val/tot"])) for r in rows]
    late = [v for s, v in pts if s >= 14_000]
    early = [v for s, v in pts if 12_000 <= s < 13_000]
    return (np.mean(early) - np.mean(late)) / np.mean(late)


def ultrasonic_share(y: np.ndarray, r: np.ndarray, rate: int = 192_000) -> dict:
    """auraloss STFTLoss terms (spectral convergence + mean |log| magnitude) per resolution,
    with the share of each term carried by bins above 24 kHz."""
    yt, rt = torch.tensor(y), torch.tensor(r)
    out = {"log_mag_share": [], "sc_numerator_share": []}
    for n, hop, win in zip(FFT, HOP, WIN, strict=True):
        window = torch.hann_window(win, dtype=torch.float64)
        Y = (
            torch.stft(yt, n, hop, win, window, return_complex=True)
            .abs()
            .clamp(min=1e-8)
        )
        R = (
            torch.stft(rt, n, hop, win, window, return_complex=True)
            .abs()
            .clamp(min=1e-8)
        )
        freqs = torch.fft.rfftfreq(n, 1 / rate)
        high = freqs > 24_000
        log_err = (torch.log(R) - torch.log(Y)).abs()
        sc_num = (R - Y) ** 2
        out["log_mag_share"].append(float(log_err[high].sum() / log_err.sum()))
        out["sc_numerator_share"].append(float(sc_num[high].sum() / sc_num.sum()))
    return out


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    takes = [t for t in ev.takes() if t["split"] == "test" and not t["low_signal"]]
    models = {arm: ev.load(run_id) for arm, run_id in RUNS.items()}
    result = {
        "val_fall_12k_to_14k": {arm: val_fall(r) for arm, r in RUNS.items()},
        "takes": {},
    }
    for take in takes:
        loop, fx = ev.audio(take)
        real = ev.scored(resample_poly(fx, 1, UP))
        controls = [v / 10 for v in take["gain_contour"]]
        row = {}
        for arm, model in models.items():
            y = run(model, arm, loop, controls)
            ys = ev.scored(y)
            gain = float((ys * real).sum() / (ys * ys).sum())
            best_esr, best_delay = oracle_delay(y, real)
            chunked = ev.scored(run(model, arm, loop, controls, CHUNK))
            row[arm] = {
                "esr": esr(ys, real),
                "oracle_gain_db": 20 * math.log10(abs(gain)),
                "esr_after_gain": esr(gain * ys, real),
                "oracle_delay": best_delay,
                "esr_after_delay": best_esr,
                "esr_chunked": esr(chunked, real),
            }
        if take["order"] == 16:
            row["neighbours"] = {
                arm: {
                    name: esr(ev.scored(run(model, arm, loop, c)), real)
                    for name, c in NEIGHBOURS.items()
                }
                for arm, model in models.items()
            }
            y192 = ev.run(models["B192"], loop, torch.tensor(controls))
            row["ultrasonic_share_B192"] = ultrasonic_share(
                ev.scored(y192, 192_000), ev.scored(fx, 192_000)
            )
        result["takes"][take["order"]] = row
        print(take["order"], json.dumps(row)[:300], flush=True)
    # H4 on one more take, a seen pair with B192 better (take 3).
    loop, fx = next(ev.audio(t) for t in takes if t["order"] == 3)
    t3 = next(t for t in takes if t["order"] == 3)
    y192 = ev.run(
        models["B192"], loop, torch.tensor([v / 10 for v in t3["gain_contour"]])
    )
    result["ultrasonic_share_B192_take3"] = ultrasonic_share(
        ev.scored(y192, 192_000), ev.scored(fx, 192_000)
    )
    (HERE / OUTPUT).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(result["val_fall_12k_to_14k"]))


if __name__ == "__main__":
    main()
