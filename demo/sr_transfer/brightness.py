#!/usr/bin/env python3
"""Brightness at 96 and 192 kHz (predictions_brightness.md). Read-only, CPU.

Pre-registered 2026-10-02 (commit 205a9c4, sine window declared in the next commit)
before any measurement. For the ZOH and FOH SSM-WaveNets on Big Muff, Δ rescaled to
96 / 192 kHz, Kaiser-64 resampling (floor.hq) both ways:
R1 third-octave levels of the model at fs minus the resampler-only control, 4-16 kHz;
R2 distance of the per-band levels to the real target, 1-16 kHz, fs against 48 kHz;
R3 sines at prime f0: inharmonic energy share and harmonic level changes.
Writes demo/sr_transfer/brightness.json.
"""

from __future__ import annotations

import importlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
floor = importlib.import_module("floor")
bench, transfer = floor.bench, floor.transfer

RATE = 48_000
RATES = (96_000, 192_000)
RUNS = {
    "zoh": [f"paired_ssm_wavenet_s{s}" for s in (925, 736, 688)],
    "foh": [f"paired_ssm_wavenet_foh_s{s}" for s in (925, 736, 688)],
}
CENTRES = [1000 * 2 ** (k / 3) for k in range(-12, 13)]  # 63 Hz to 16 kHz
BRIGHT = [c for c in CENTRES if 3900 < c < 16500]  # 4 to 16 kHz
CLOSE = [c for c in CENTRES if 990 < c < 16500]  # 1 to 16 kHz
PRIMES = (211, 997, 3001)
LISTENED = (10, 8)


def flat(x: torch.Tensor) -> np.ndarray:
    return transfer.interior(x, RATE).reshape(-1).double().numpy()


def band_levels(x: np.ndarray) -> dict[float, float]:
    freqs, psd = welch(x, RATE, nperseg=8192, window="hann")
    df = freqs[1] - freqs[0]
    out = {}
    for c in CENTRES:
        band = (freqs >= c * 2 ** (-1 / 6)) & (freqs < c * 2 ** (1 / 6))
        out[c] = 10 * math.log10(psd[band].sum() * df + 1e-30)
    return out


def mean_over(levels: dict, ref: dict, centres: list[float], absolute=False) -> float:
    diffs = [levels[c] - ref[c] for c in centres]
    return (
        statistics.mean(abs(d) for d in diffs) if absolute else statistics.mean(diffs)
    )


@torch.no_grad()
def at_rate(model, base, x48: torch.Tensor, rate: int) -> torch.Tensor:
    transfer.set_rate(model, rate, base)
    if rate == RATE:
        return transfer.run(model, x48)
    return floor.hq(transfer.run(model, floor.hq(x48, RATE, rate)), rate, RATE)


@torch.no_grad()
def sine_spectrum(model, base, f0: int, amplitude: float, rate: int) -> np.ndarray:
    """|FFT| of the 0.5-1.5 s window of a 1.6 s sine run at `rate`, at 48 kHz."""
    n = np.arange(int(1.6 * rate))
    x = torch.tensor(amplitude * np.sin(2 * np.pi * f0 * n / rate), dtype=torch.float32)
    transfer.set_rate(model, rate, base)
    y = transfer.run(model, x.reshape(1, 1, -1))
    if rate != RATE:
        y = floor.hq(y, rate, RATE)
    window = y.reshape(-1).double().numpy()[RATE // 2 : RATE // 2 + RATE]
    return np.abs(np.fft.rfft(window))


def sine_readings(spectra: dict[int, np.ndarray], f0: int) -> dict:
    harmonic = np.arange(f0, RATE // 2, f0)
    mask = np.ones(RATE // 2 + 1, dtype=bool)
    mask[0] = False
    mask[harmonic] = False
    out = {}
    for rate, mag in spectra.items():
        energy = mag**2
        total = energy[1:].sum()
        out[f"I_{rate // 1000}"] = 10 * math.log10(energy[mask].sum() / total + 1e-30)
    band = harmonic[(harmonic >= 4000) & (harmonic <= 16000)]
    ref = 20 * np.log10(spectra[RATE][band] + 1e-30)
    for rate in RATES:
        level = 20 * np.log10(spectra[rate][band] + 1e-30)
        out[f"H_{rate // 1000}"] = float(np.median(np.abs(level - ref)))
        out[f"H_signed_{rate // 1000}"] = float(np.median(level - ref))
    return out


def fractional_shift(x: np.ndarray, delay: float) -> np.ndarray:
    """Advance x by `delay` samples (FFT phase), per whole signal."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x))
    return np.fft.irfft(spectrum * np.exp(2j * np.pi * freqs * delay), len(x))


@torch.no_grad()
def checkpoint(kind: str, run_id: str, x48, y48, rms: float, delays: dict) -> dict:
    model, _ = transfer.load_model(run_id, "last")
    base = {
        id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
        for m in transfer.ssm_modules(model)
    }
    out48 = at_rate(model, base, x48, RATE)
    real = band_levels(flat(y48))
    l48 = band_levels(flat(out48))
    result = {
        "precondition_48_minus_real_4_16k": mean_over(l48, real, BRIGHT),
        "D_48": mean_over(l48, real, CLOSE, absolute=True),
        "levels_real": real,
        "levels_48": l48,
    }
    listened = {}
    for rate in RATES:
        out = at_rate(model, base, x48, rate)
        control = floor.hq(floor.hq(out48, RATE, rate), rate, RATE)
        lfs, lctl = band_levels(flat(out)), band_levels(flat(control))
        key = str(rate // 1000)
        result[f"B_{key}"] = mean_over(lfs, lctl, BRIGHT)
        result[f"control_minus_48_4_16k_{key}"] = mean_over(lctl, l48, BRIGHT)
        result[f"D_{key}"] = mean_over(lfs, real, CLOSE, absolute=True)
        result[f"D_control_{key}"] = mean_over(lctl, real, CLOSE, absolute=True)
        result[f"levels_{key}"] = lfs
        for seg in LISTENED:
            a = transfer.interior(out48[seg], RATE).reshape(-1).double().numpy()
            b = transfer.interior(out[seg], RATE).reshape(-1).double().numpy()
            row = {
                "esr_vs_48": float(((a - b) ** 2).sum() / (a**2).sum()),
                "bright_vs_48_4_16k": mean_over(band_levels(b), band_levels(a), BRIGHT),
            }
            if kind == "zoh":
                shifted = fractional_shift(b, delays[run_id][str(rate)])
                row["esr_vs_48_delay_compensated"] = float(
                    ((a - shifted) ** 2).sum() / (a**2).sum()
                )
            listened[f"{seg}_{key}"] = row
    result["listened"] = listened
    sines = {}
    for f0 in PRIMES:
        for label, gain in (("test_rms", 1.0), ("minus12dB", 10 ** (-12 / 20))):
            amplitude = rms * math.sqrt(2) * gain
            spectra = {
                r: sine_spectrum(model, base, f0, amplitude, r) for r in (RATE, *RATES)
            }
            sines[f"{f0}_{label}"] = sine_readings(spectra, f0)
    result["sines"] = sines
    print(
        f"{run_id}: B192 {result['B_192']:+.2f} dB, D48 {result['D_48']:.2f} "
        f"D192 {result['D_192']:.2f}, I48 {sines['997_test_rms']['I_48']:.1f} "
        f"I192 {sines['997_test_rms']['I_192']:.1f} dB, "
        f"H192 {sines['997_test_rms']['H_192']:.2f} dB",
        flush=True,
    )
    return result


def count(values: list[bool]) -> int:
    return sum(bool(v) for v in values)


def main() -> None:
    torch.set_num_threads(16)
    data = bench.data_module("test")
    data.setup("test")
    x48 = torch.stack([s[0] for s in data.test_dataset])
    y48 = torch.stack([s[1] for s in data.test_dataset])
    rms = float(np.sqrt(np.mean(flat(x48) ** 2)))
    kappa = json.loads((HERE / "kappa.json").read_text())["bigmuff"]
    delays = {
        r: {str(rate): kappa[r][str(rate)]["delay_median"] for rate in RATES}
        for r in kappa
    }
    results = {
        f"{kind}/{run_id}": checkpoint(kind, run_id, x48, y48, rms, delays)
        for kind, runs in RUNS.items()
        for run_id in runs
    }
    cks = list(results.values())
    readings = {"input_rms": rms, "segments": len(x48)}
    for rate in ("96", "192"):
        b = [c[f"B_{rate}"] for c in cks]
        readings[f"R1_{rate}"] = (
            "moins brillant"
            if count(v <= -0.5 for v in b) >= 5
            else "plus brillant"
            if count(v >= 0.5 for v in b) >= 5
            else "non tranché"
        )
        closer = count(c[f"D_{rate}"] <= c["D_48"] - 0.25 for c in cks)
        farther = count(c[f"D_{rate}"] >= c["D_48"] + 0.25 for c in cks)
        readings[f"R2_{rate}"] = (
            "plus proche"
            if closer >= 5
            else "plus loin"
            if farther >= 5
            else "non tranché"
        )
    main_sine = [c["sines"]["997_test_rms"] for c in cks]
    aliasing = count(
        s["I_192"] <= s["I_48"] - 10 and s["H_192"] <= 0.5 for s in main_sine
    )
    changed = count(s["H_192"] > 1 for s in main_sine)
    readings["R3"] = (
        "repliement"
        if aliasing >= 5
        else "réponse changée"
        if changed >= 5
        else "non tranché"
    )
    results["readings"] = readings
    (HERE / "brightness.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False) + "\n"
    )
    print(json.dumps(readings, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
