"""Class bound of a pre-FIR(T) -> post-FIR(T) cascade on Fulltone: LS FIR of 2T-1 taps
fitted on train, factorised into two T-tap FIRs, evaluated on test. Read-only."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.linalg import solve_toeplitz
from scipy.signal import fftconvolve

ROOT = Path("/fastdata/lavaulta/neural_amp")
DATA = ROOT / "datasets/raw/internal_m4/fulltone_full_drive_2"
OUT = Path(__file__).resolve().parent / "cascade_bound.json"


def esr(p, t):
    return float(np.sum((p - t) ** 2) / np.sum(t**2))


def factorise(h, taps, iters=300):
    a = np.zeros(taps); a[0] = 1.0
    b = h[:taps].copy()
    n = len(h)
    for _ in range(iters):
        A = np.zeros((n, taps))
        for k in range(taps):
            A[k:k + taps, k] = a
        b = np.linalg.lstsq(A, h, rcond=None)[0]
        B = np.zeros((n, taps))
        for k in range(taps):
            B[k:k + taps, k] = b
        a = np.linalg.lstsq(B, h, rcond=None)[0]
    return a, b


def main():
    tx, _ = sf.read(DATA / "train_input.wav", dtype="float64")
    ty, _ = sf.read(DATA / "train_target.wav", dtype="float64")
    vx, _ = sf.read(DATA / "validation_input.wav", dtype="float64")
    vy, _ = sf.read(DATA / "validation_target.wav", dtype="float64")
    sx, _ = sf.read(DATA / "test_input.wav", dtype="float64")
    sy, _ = sf.read(DATA / "test_target.wav", dtype="float64")
    n = tx.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X, Y = np.fft.rfft(tx, nfft), np.fft.rfft(ty, nfft)
    rxx = np.fft.irfft(X * np.conj(X), nfft)[:512].copy()
    rxy = np.fft.irfft(Y * np.conj(X), nfft)[:512]
    rxx[0] *= 1 + 1e-8
    res = {"identity": {"train": esr(tx, ty), "validation": esr(vx, vy), "test": esr(sx, sy)}}
    for taps in (17, 25, 33, 49, 65):
        mem = 2 * taps - 1
        h = solve_toeplitz(rxx[:mem], rxy[:mem])
        a, b = factorise(h, taps)
        casc = np.convolve(a, b)
        res[f"taps_{taps}"] = {
            "cascade_memory": mem,
            "ls_fir_esr": {s: esr(fftconvolve(x, h)[: x.size], y) for s, x, y in (("train", tx, ty), ("validation", vx, vy), ("test", sx, sy))},
            "factorised_relative_error": float(np.linalg.norm(casc - h) / np.linalg.norm(h)),
            "factorised_test_esr": esr(fftconvolve(sx, casc)[: sx.size], sy),
            "ls_fir_max_abs_tap": float(np.max(np.abs(h))),
            "ls_fir_l2_delta_from_identity": float(np.linalg.norm(h - np.eye(1, mem, 0)[0])),
            "pre_max_abs_delta_from_delta": float(np.max(np.abs(a - np.eye(1, taps, 0)[0]))),
            "post_max_abs_delta_from_delta": float(np.max(np.abs(b - np.eye(1, taps, 0)[0]))),
        }
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


main()
