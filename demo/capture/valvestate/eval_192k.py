#!/usr/bin/env python3
"""Evaluation of predictions_192k.md, at 48 kHz. Read-only, CPU.

A48 runs get the loopback decimated by resample_poly(., 1, 4); B192 runs get the
192 kHz loopback and their output is decimated the same way. Reference: the FX send
decimated the same way. 8 unseen test takes, scored region of product_vs100r_eval.
M1: per run s = mean over takes of ln ESR; d = mean s(B192) - mean s(A48), against
ln 1.5 and 2 sigma sqrt(2/3), sigma the pooled within-arm sd of s.
M2: third-octave levels (Welch 8192, Hann) of the scored regions put end to end,
e_b = L_run - L_real; |e| at the 16 kHz third-octave read explicitly, mean |e| over
1-16 kHz reported. Descriptive: 997 Hz sine, off-grid energy share in two windows
(settling check), spectra saved.
Readings are given only when the six runs exist. Writes eval_192k.json (and .npz).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly, welch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
import product_vs100r_eval as ev  # noqa: E402

ARMS = {
    "A48": ["vs100r_gc_film_joint_s925_r2", "vs100r48_film_s43", "vs100r48_film_s44"],
    "B192": ["vs100r192_film_s42", "vs100r192_film_s43", "vs100r192_film_s44"],
}
RATE, UP = 48_000, 4
CENTRES = [1000 * 2 ** (k / 3) for k in range(-12, 13)]
CLOSE = [c for c in CENTRES if 990 < c < 16500]
TOP = CENTRES[-1]  # 16 kHz third-octave
F0 = 997


def levels(x: np.ndarray) -> dict[float, float]:
    freqs, psd = welch(x, RATE, nperseg=8192, window="hann")
    df = freqs[1] - freqs[0]
    return {
        c: 10
        * math.log10(
            psd[(freqs >= c * 2 ** (-1 / 6)) & (freqs < c * 2 ** (1 / 6))].sum() * df
            + 1e-30
        )
        for c in CENTRES
    }


@torch.no_grad()
def output48(
    model, arm: str, loop192: np.ndarray, controls: torch.Tensor
) -> np.ndarray:
    if arm == "A48":
        return ev.run(model, resample_poly(loop192, 1, UP), controls)
    return resample_poly(ev.run(model, loop192, controls), 1, UP)


@torch.no_grad()
def sine(model, arm: str, amplitude: float) -> dict:
    rate = RATE if arm == "A48" else RATE * UP
    n = np.arange(2 * rate)
    x = amplitude * np.sin(2 * np.pi * F0 * n / rate)
    y = ev.run(model, x, torch.tensor([0.5, 0.5]))
    if arm == "B192":
        y = resample_poly(y, 1, UP)
    out = {}
    harmonic = np.arange(F0, RATE // 2, F0)
    for name, start in (("w05", RATE // 2), ("w09", int(0.9 * RATE))):
        mag = np.abs(np.fft.rfft(y[start : start + RATE])) ** 2
        mask = np.ones(len(mag), bool)
        mask[0] = False
        mask[harmonic] = False
        out[name] = 10 * math.log10(mag[mask].sum() / mag[1:].sum() + 1e-30)
        out[f"spectrum_{name}"] = mag
    return out


def main() -> None:
    torch.set_num_threads(16)
    takes = [t for t in ev.takes() if t["split"] == "test" and not t["low_signal"]]
    data = {t["order"]: ev.audio(t) for t in takes}
    real = np.concatenate(
        [ev.scored(resample_poly(fx, 1, UP)) for _, fx in data.values()]
    )
    rms = float(
        np.sqrt(
            np.mean(
                np.concatenate([resample_poly(lp, 1, UP) for lp, _ in data.values()])
                ** 2
            )
        )
    )
    real_levels = levels(real)
    result, spectra = {"input_rms": rms, "takes": list(data), "runs": {}}, {}
    for arm, runs in ARMS.items():
        for run_id in runs:
            if not (bench.RESULTS_DIR / f"{run_id}.json").exists():
                print(f"{run_id}: absent", flush=True)
                continue
            model = ev.load(run_id)
            outs, esrs = [], {}
            for take in takes:
                loop, fx = data[take["order"]]
                controls = torch.tensor([v / 10 for v in take["gain_contour"]])
                y = output48(model, arm, loop, controls)
                ref = resample_poly(fx, 1, UP)
                esrs[take["order"]] = ev.esr(y, ref)
                outs.append(ev.scored(y))
            lv = levels(np.concatenate(outs))
            e = {c: lv[c] - real_levels[c] for c in CENTRES}
            s = sine(model, arm, rms * math.sqrt(2))
            spectra[run_id] = np.stack([s.pop("spectrum_w05"), s.pop("spectrum_w09")])
            result["runs"][run_id] = {
                "arm": arm,
                "esr_by_take": esrs,
                "s": statistics.mean(math.log(v) for v in esrs.values()),
                "e_by_band": e,
                "abs_e_16k": abs(e[TOP]),
                "mean_abs_e_1_16k": statistics.mean(abs(e[c]) for c in CLOSE),
                "sine_offgrid_db": s,
            }
            print(
                f"{run_id}: s {result['runs'][run_id]['s']:.3f}, "
                f"|e16k| {abs(e[TOP]):.2f} dB",
                flush=True,
            )
    runs = result["runs"]
    if all(r in runs for rs in ARMS.values() for r in rs):
        s = {arm: [runs[r]["s"] for r in rs] for arm, rs in ARMS.items()}
        sigma = math.sqrt(
            (statistics.variance(s["A48"]) + statistics.variance(s["B192"])) / 2
        )
        d = statistics.mean(s["B192"]) - statistics.mean(s["A48"])
        bound = 2 * sigma * math.sqrt(2 / 3)
        m1 = (
            "B192 plus proche"
            if d <= -math.log(1.5) and d < -bound
            else "A48 plus proche"
            if d >= math.log(1.5) and d > bound
            else "non tranché"
        )
        top = {arm: [runs[r]["abs_e_16k"] for r in rs] for arm, rs in ARMS.items()}
        diff = statistics.mean(top["B192"]) - statistics.mean(top["A48"])
        pairs = list(zip(top["A48"], top["B192"], strict=True))
        b_wins = sum(b < a for a, b in pairs)
        a_wins = sum(a < b for a, b in pairs)
        m2 = (
            "B192 plus proche dans l'aigu"
            if diff <= -0.5 and b_wins >= 2
            else "A48 plus proche dans l'aigu"
            if diff >= 0.5 and a_wins >= 2
            else "non tranché"
        )
        result["readings"] = {
            "d": d,
            "sigma": sigma,
            "bound": bound,
            "M1": m1,
            "mean_abs_e16k_diff_B_minus_A": diff,
            "M2": m2,
        }
        print(json.dumps(result["readings"], indent=1, ensure_ascii=False))
    (HERE / "eval_192k.json").write_text(json.dumps(result, indent=1) + "\n")
    np.savez_compressed(HERE / "eval_192k_sines.npz", **spectra)


if __name__ == "__main__":
    main()
