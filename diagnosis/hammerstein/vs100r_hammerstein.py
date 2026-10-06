#!/usr/bin/env python3
"""Parallel Hammerstein analysis of the VS100R and of its models (exploration).

Each take holds three exponential sweeps (20 Hz - 20 kHz, 5 s, -20/-10/-3 dBFS) at
192 kHz. Reference input: the take's own loopback; analysis at 48 kHz after
resample_poly(., 1, 4) of every output, the decimation of the training data.
Device: the FX send. Models: A48 runs fed the decimated loopback, B192 runs fed the
192 kHz loopback (output decimated), as in eval_192k.py.
The OD2 saturates hard: the polynomial filters G_n depend strongly on the level and
high orders compensate each other, so the main comparison is on the measured harmonic
responses D_k (frequency-dependent harmonic distortion, no inversion), at equal level;
G_n are kept as a complement.
Writes vs100r_hammerstein.json: per take, level and system, |D_k| and |G_n| per
third-octave (dB) and the errors to the device per order and band (level in dB,
complex relative);
the device's repeatability over the takes of the (5, 5) pair; and how Hammerstein the
device is (prediction ESR of the pink-noise and di_test sections from the identified
filters). Read-only, CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
import product_vs100r_eval as ev  # noqa: E402

from fssr_nam.product.hammerstein import Sweep, calibration, identify  # noqa: E402

RATE, UP, ORDER, NFFT = 48_000, 4, 5, 8192
LEVELS = (-20, -10, -3)
SWEEP = Sweep()
SECTIONS = {s["section"]: s for s in ev.LAYOUT["sections"]}
RUNS = {
    "A48": ["vs100r_gc_film_joint_s925_r2", "vs100r48_film_s43", "vs100r48_film_s44"],
    "B192": ["vs100r192_film_s42", "vs100r192_film_s43", "vs100r192_film_s44"],
    "C": ["vs100r48fast_film_s42", "vs100r48fast_film_s43", "vs100r48fast_film_s44"],
    # demo/odd_gate and demo/greybox (absent records are skipped).
    "ODD": ["vs100r48odd_film_s42", "vs100r48odd_film_s43", "vs100r48odd_film_s44"],
    "GREY": ["vs100r48grey_s42", "vs100r48grey_s43", "vs100r48grey_s44"],
}
CENTRES = [1000 * 2 ** (k / 3) for k in range(-10, 13)]  # 100 Hz to 16 kHz


def decimate(x: np.ndarray) -> np.ndarray:
    return resample_poly(x, 1, UP)


def section(take: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(loopback, FX send) of one section at 192 kHz, isolated spikes repaired."""
    p, s = take["pass"], SECTIONS[name]
    a, b = p["start"] + s["start"], p["start"] + s["start"] + s["samples"]
    stamp = p["file"].removeprefix("02-Looopbck-")
    loop = sf.read(ev.MEDIA / p["file"], dtype="float64", start=a, stop=b)[0]
    fx = sf.read(ev.MEDIA / f"03-FX Out-{stamp}", dtype="float64", start=a, stop=b)[0]
    for i in take["spikes"]:
        if a < i < b - 1:
            fx[i - a] = 0.5 * (fx[i - a - 1] + fx[i - a + 1])
    return loop, fx


def bands(freqs: np.ndarray, values: np.ndarray) -> list[float]:
    out = []
    for c in CENTRES:
        sel = (freqs >= c * 2 ** (-1 / 6)) & (freqs < c * 2 ** (1 / 6))
        out.append(float(np.median(values[sel])) if sel.any() else float("nan"))
    return out


def summary(g: np.ndarray, freqs: np.ndarray) -> list[list[float]]:
    return [bands(freqs, 20 * np.log10(np.abs(g[n]) + 1e-30)) for n in range(ORDER)]


def errors(g: np.ndarray, ref: np.ndarray, freqs: np.ndarray) -> dict:
    level = 20 * np.log10((np.abs(g) + 1e-30) / (np.abs(ref) + 1e-30))
    complex_rel = np.abs(g - ref) / (np.abs(ref) + 1e-30)
    return {
        "level_db": [bands(freqs, level[n]) for n in range(ORDER)],
        "complex_rel": [bands(freqs, complex_rel[n]) for n in range(ORDER)],
    }


def predict(g: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Output of the identified model for input x (both at 48 kHz). The filters are
    the spectra of windows starting PRE_S before each response: shift them back."""
    from fssr_nam.product.hammerstein import PRE_S

    pre = int(PRE_S * RATE)
    y = np.zeros(len(x))
    for n in range(ORDER):
        h = np.roll(np.fft.irfft(g[n], NFFT), -pre)
        h[NFFT - pre :] = 0  # the pre-window part is non-causal noise
        spectrum = np.fft.rfft(x ** (n + 1), len(x) + NFFT) * np.fft.rfft(
            h, len(x) + NFFT
        )
        y += np.fft.irfft(spectrum, len(x) + NFFT)[: len(x)]
    return y


def esr(pred: np.ndarray, ref: np.ndarray) -> float:
    cut = slice(RATE // 2, -RATE // 10)
    return float(((ref[cut] - pred[cut]) ** 2).sum() / (ref[cut] ** 2).sum())


@torch.no_grad()
def model_output(model, arm: str, loop: np.ndarray, controls) -> np.ndarray:
    c = torch.tensor(controls)
    if arm == "B192":
        return decimate(ev.run(model, loop, c))
    return ev.run(model, decimate(loop), c)


def main() -> None:
    torch.set_num_threads(16)
    takes = [t for t in ev.takes() if not t["low_signal"]]
    test = [t for t in takes if t["split"] == "test"]
    models = {
        arm: {r: ev.load(r) for r in runs if (bench.RESULTS_DIR / f"{r}.json").exists()}
        for arm, runs in RUNS.items()
    }
    out = {"centres": CENTRES, "takes": {}, "repeatability": {}, "validity": {}}
    freqs = None
    for take in test:
        controls = [v / 10 for v in take["gain_contour"]]
        row = {}
        for level in LEVELS:
            loop, fx = section(take, f"sweep_{level}dBFS")
            m = calibration(loop, decimate, RATE, SWEEP, ORDER, NFFT)
            dev = identify(decimate(fx), loop, decimate, RATE, SWEEP, ORDER, NFFT, m)
            freqs = dev.freqs
            entry = {
                "device_g": summary(dev.g, freqs),
                "device_d": summary(dev.d, freqs),
                "models": {},
            }
            for arm, runs in models.items():
                for run_id, model in runs.items():
                    y = model_output(model, arm, loop, controls)
                    fit = identify(y, loop, decimate, RATE, SWEEP, ORDER, NFFT, m)
                    entry["models"][run_id] = {
                        "arm": arm,
                        "d": summary(fit.d, freqs),
                        "g": summary(fit.g, freqs),
                        "d_errors": errors(fit.d, dev.d, freqs),
                        "g_errors": errors(fit.g, dev.g, freqs),
                    }
            # How Hammerstein the device is at this level: predict other sections.
            validity = {}
            for name in ("pink_noise_-30dBFS_rms", "di_test"):
                x_loop, x_fx = section(take, name)
                pred = predict(dev.g, decimate(x_loop))
                validity[name] = esr(pred, decimate(x_fx))
            entry["validity_esr"] = validity
            row[str(level)] = entry
        out["takes"][str(take["order"])] = row
        print(
            take["order"],
            {lv: row[str(lv)]["validity_esr"] for lv in LEVELS},
            flush=True,
        )
    # Device repeatability: takes of the (5, 5) pair, |G_n| spread per band.
    same = [t for t in takes if tuple(t["gain_contour"]) == (5.0, 5.0)]
    for level in LEVELS:
        mags = []
        for take in same:
            loop, fx = section(take, f"sweep_{level}dBFS")
            dev = identify(decimate(fx), loop, decimate, RATE, SWEEP, ORDER, NFFT)
            mags.append(summary(dev.d, dev.freqs))
        mags = np.array(mags)  # takes, order, band
        out["repeatability"][str(level)] = {
            "takes": [t["order"] for t in same],
            "d_sd_db": np.nanstd(mags, axis=0).tolist(),
        }
        print("répétabilité", level, len(same), "prises", flush=True)
    (HERE / "vs100r_hammerstein.json").write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
