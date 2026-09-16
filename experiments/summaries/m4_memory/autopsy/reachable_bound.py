"""Reachable linear bounds on Fulltone (fit on train, eval on test), read-only.

1. Ridge LS FIR toward the identity delta, memory M, penalty lambda: reports test ESR
   versus the coefficient displacement max|h - delta| (Adam reach proxy).
2. One-pole IIR family y = a*x + b*onepole(x, tau) (+ optional 2nd pole): grid over tau.
"""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.linalg import solve_toeplitz
from scipy.signal import fftconvolve, lfilter

ROOT = Path("/fastdata/lavaulta/neural_amp")
DATA = ROOT / "datasets/raw/internal_m4/fulltone_full_drive_2"
OUT = Path(__file__).resolve().parent / "reachable_bound.json"
FS = 48000.0


def esr(p, t):
    return float(np.sum((p - t) ** 2) / np.sum(t**2))


def main():
    tx, _ = sf.read(DATA / "train_input.wav", dtype="float64")
    ty, _ = sf.read(DATA / "train_target.wav", dtype="float64")
    sx, _ = sf.read(DATA / "test_input.wav", dtype="float64")
    sy, _ = sf.read(DATA / "test_target.wav", dtype="float64")
    n = tx.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X, Y = np.fft.rfft(tx, nfft), np.fft.rfft(ty, nfft)
    rxx = np.fft.irfft(X * np.conj(X), nfft)[:1024].copy()
    rxy = np.fft.irfft(Y * np.conj(X), nfft)[:1024]
    energy = rxx[0]
    res = {"train_input_energy": float(energy)}
    ridge = {}
    for mem in (17, 33, 65, 97, 129, 257, 513):
        rows = []
        delta = np.zeros(mem); delta[0] = 1.0
        for lam_rel in (0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
            lam = lam_rel * energy
            r = rxx[:mem].copy(); r[0] += lam
            # minimise |Xh-y|^2 + lam |h-delta|^2  ->  (R + lam I) h = rxy + lam delta
            h = solve_toeplitz(r, rxy[:mem] + lam * delta)
            rows.append({
                "lambda_rel": lam_rel,
                "max_abs_delta": float(np.max(np.abs(h - delta))),
                "l2_delta": float(np.linalg.norm(h - delta)),
                "train_esr": esr(fftconvolve(tx, h)[: tx.size], ty),
                "test_esr": esr(fftconvolve(sx, h)[: sx.size], sy),
            })
        ridge[f"memory_{mem}"] = rows
    res["ridge_fir_toward_identity"] = ridge
    # one-pole family: features x and onepole(x, tau); LS on the 2 gains (a, b) for each tau
    onepole = {}
    def lp(x, tau_ms):
        alpha = 1.0 - np.exp(-1.0 / (FS * tau_ms * 1e-3))
        return lfilter([alpha], [1.0, -(1.0 - alpha)], x)
    for tau_ms in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0):
        F = np.stack((tx, lp(tx, tau_ms)), axis=1)
        g = np.linalg.lstsq(F, ty, rcond=None)[0]
        Ft = np.stack((sx, lp(sx, tau_ms)), axis=1)
        onepole[f"tau_{tau_ms}ms"] = {
            "gains_a_b": [float(v) for v in g],
            "train_esr": esr(F @ g, ty),
            "test_esr": esr(Ft @ g, sy),
        }
    res["onepole_plus_direct"] = onepole
    # two poles (tau1, tau2) + direct
    two = {}
    for t1 in (0.5, 1.0, 2.0):
        for t2 in (3.0, 5.0, 10.0, 20.0):
            F = np.stack((tx, lp(tx, t1), lp(tx, t2)), axis=1)
            g = np.linalg.lstsq(F, ty, rcond=None)[0]
            Ft = np.stack((sx, lp(sx, t1), lp(sx, t2)), axis=1)
            two[f"tau_{t1}_{t2}ms"] = {"gains": [float(v) for v in g], "train_esr": esr(F @ g, ty), "test_esr": esr(Ft @ g, sy)}
    res["two_poles_plus_direct"] = two
    # one pole + short FIR (17 taps) on the pole output and on x: the S0 pre-FIR could absorb this
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    for mem, rows in ridge.items():
        print(mem)
        for r in rows:
            print(f"  lam={r['lambda_rel']:<7} maxd={r['max_abs_delta']:8.3f} l2={r['l2_delta']:8.3f} train={r['train_esr']:.4f} test={r['test_esr']:.4f}")
    print("onepole", json.dumps(onepole, indent=1))
    print("twopoles", json.dumps(two, indent=1))


main()
