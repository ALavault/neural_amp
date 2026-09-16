"""H2/H6: level-dependent linear response, gated FIR bank, Hammerstein bound, linear
recoverability of the S3 and A2 residuals. Fits on the full train file, evaluation on
test and on the seed-0 windows 101-200. Read-only; CPU.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from common import (
    A2_RUNS,
    FS,
    OUTPUT,
    S3_RUNS,
    audio,
    causal_envelope,
    dump,
    esr,
    load_model,
    predict_file,
    replay_starts,
    saved_test_prediction,
    windows_for_steps,
)

TAPS = 65
TRANCHES = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 10.0))
CHUNK = 250_000
PROBE_HZ = (50, 100, 150, 200, 300, 500, 800, 1200, 2000)


def lag_matrix(x: np.ndarray, start: int, stop: int, taps: int) -> np.ndarray:
    """Rows n in [start, stop): [x[n], x[n-1], ..., x[n-taps+1]] (zero before 0)."""
    padded = np.concatenate((np.zeros(taps - 1), x))
    idx = np.arange(start, stop)[:, None] + (taps - 1 - np.arange(taps))[None, :]
    return padded[idx]


def ls_fir(x, y, taps, mask=None, ridge=1e-9):
    xd, yd = x.astype(np.float64), y.astype(np.float64)
    ata = np.zeros((taps, taps)); atb = np.zeros(taps)
    for s in range(0, xd.size, CHUNK):
        e = min(xd.size, s + CHUNK)
        A = lag_matrix(xd, s, e, taps)
        b = yd[s:e]
        if mask is not None:
            m = mask[s:e]
            A, b = A[m], b[m]
        ata += A.T @ A; atb += A.T @ b
    ata += ridge * np.trace(ata) / taps * np.eye(taps)
    return np.linalg.solve(ata, atb)


def apply_fir(x, h):
    return fftconvolve(x.astype(np.float64), h)[: x.size]


def response_db(h, freqs):
    w = 2 * np.pi * np.asarray(freqs) / FS
    z = np.exp(-1j * np.outer(w, np.arange(len(h))))
    return 20 * np.log10(np.abs(z @ h) + 1e-12)


def gated_bank(x_train, y_train, x_test, y_test, env_train, env_test, label):
    single = ls_fir(x_train, y_train, TAPS)
    out = {"single_fir": {"train_esr": esr(apply_fir(x_train, single), y_train), "test_esr": esr(apply_fir(x_test, single), y_test)}}
    ref = response_db(single, PROBE_HZ)
    bank_train = np.zeros_like(y_train, dtype=np.float64); bank_test = np.zeros_like(y_test, dtype=np.float64)
    tranches = {}
    for lo, hi in TRANCHES:
        m_train = (env_train >= lo) & (env_train < hi)
        m_test = (env_test >= lo) & (env_test < hi)
        h = ls_fir(x_train, y_train, TAPS, mask=m_train)
        p_train, p_test = apply_fir(x_train, h), apply_fir(x_test, h)
        bank_train[m_train] = p_train[m_train]; bank_test[m_test] = p_test[m_test]
        resp = response_db(h, PROBE_HZ)
        tranches[f"{lo}-{hi}"] = {
            "train_samples": int(m_train.sum()),
            "gain_db_at_200hz": float(resp[PROBE_HZ.index(200)]),
            "shape_db_rel_200hz": {str(f): round(float(r - resp[PROBE_HZ.index(200)]), 2) for f, r in zip(PROBE_HZ, resp)},
            "shape_db_rel_single_fir_shape": {str(f): round(float((r - resp[PROBE_HZ.index(200)]) - (rs - ref[PROBE_HZ.index(200)])), 2) for f, r, rs in zip(PROBE_HZ, resp, ref)},
            "local_train_esr_single": float(np.sum((apply_fir(x_train, single)[m_train] - y_train[m_train]) ** 2) / np.sum(y_train[m_train].astype(np.float64) ** 2)),
            "local_train_esr_gated": float(np.sum((p_train[m_train] - y_train[m_train]) ** 2) / np.sum(y_train[m_train].astype(np.float64) ** 2)),
        }
    out["single_fir_shape_db_rel_200hz"] = {str(f): round(float(r - ref[PROBE_HZ.index(200)]), 2) for f, r in zip(PROBE_HZ, ref)}
    out["tranches"] = tranches
    out["gated_bank"] = {"train_esr": esr(bank_train, y_train), "test_esr": esr(bank_test, y_test)}
    print(label, "single", out["single_fir"], "gated", out["gated_bank"], flush=True)
    for k, v in tranches.items():
        print("  ", k, v["train_samples"], "gain200", round(v["gain_db_at_200hz"], 2), "shape rel single", v["shape_db_rel_single_fir_shape"], flush=True)
    return out


def hammerstein(x_train, y_train, x_test, y_test, taps=63):
    feats = lambda x: (x.astype(np.float64), np.tanh(3 * x), np.tanh(6 * x))
    ft, fs = feats(x_train), feats(x_test)
    k = len(ft)
    ata = np.zeros((k * taps, k * taps)); atb = np.zeros(k * taps)
    for s in range(0, x_train.size, CHUNK):
        e = min(x_train.size, s + CHUNK)
        A = np.concatenate([lag_matrix(f, s, e, taps) for f in ft], axis=1)
        ata += A.T @ A; atb += A.T @ y_train[s:e].astype(np.float64)
    ata += 1e-9 * np.trace(ata) / (k * taps) * np.eye(k * taps)
    h = np.linalg.solve(ata, atb).reshape(k, taps)
    pt = sum(apply_fir(f, hh) for f, hh in zip(ft, h))
    ps = sum(apply_fir(f, hh) for f, hh in zip(fs, h))
    return h, pt, ps


def main():
    train_x, train_y = audio("train")
    test_x, test_y = audio("test")
    env_train, env_test = causal_envelope(train_x), causal_envelope(test_x)
    result = {"envelope": "causal one-pole RMS, tau 10 ms", "taps": TAPS}
    result["envelope_tranche_share_train"] = {f"{lo}-{hi}": float(((env_train >= lo) & (env_train < hi)).mean()) for lo, hi in TRANCHES}

    # predictions of the seed-0 models on the full train file (for residual analysis and
    # for estimating their own level-dependent responses)
    preds_train, preds_test = {}, {}
    for label, run_id in (("S3_t33_seed0", S3_RUNS[0]), ("A2_seed0", A2_RUNS[0])):
        model, code, _ = load_model(run_id)
        preds_train[label] = predict_file(model, code, train_x)
        preds_test[label] = saved_test_prediction(run_id)
        result[f"{label}_full_train_esr"] = esr(preds_train[label], train_y)
        print(label, "full-train ESR", result[f"{label}_full_train_esr"], flush=True)

    # H2: level-gated FIR bank on target, on A2 output, on S3 output
    result["target"] = gated_bank(train_x, train_y, test_x, test_y, env_train, env_test, "target")
    for label in preds_train:
        result[label] = gated_bank(train_x, preds_train[label], test_x, preds_test[label], env_train, env_test, label)

    # Hammerstein bound (reproduces the previous diagnosis) evaluated on test and on windows 101-200
    h, pt, ps = hammerstein(train_x, train_y, test_x, test_y)
    starts = replay_starts(0, train_x.size)
    _, _, ss = windows_for_steps(train_x, train_y, starts, 101, 200)
    idx = np.concatenate([np.arange(s, s + OUTPUT) for s in ss])
    result["hammerstein_63"] = {"train_esr": esr(pt, train_y), "test_esr": esr(ps, test_y), "windows_101_200_seed0_esr": esr(pt[idx], train_y[idx])}
    single = ls_fir(train_x, train_y, TAPS)
    p_single = apply_fir(train_x, single)
    result["single_fir_65_windows_101_200_seed0_esr"] = esr(p_single[idx], train_y[idx])
    for label in preds_train:
        result[f"{label}_windows_101_200_from_full_train_prediction"] = esr(preds_train[label][idx], train_y[idx])
    print("hammerstein", result["hammerstein_63"], flush=True)

    # gated Hammerstein: static NL + FIR per tranche
    bank_train = np.zeros_like(train_y, dtype=np.float64); bank_test = np.zeros_like(test_y, dtype=np.float64)
    for lo, hi in TRANCHES:
        m_train = (env_train >= lo) & (env_train < hi); m_test = (env_test >= lo) & (env_test < hi)
        feats = lambda x: (x.astype(np.float64), np.tanh(3 * x), np.tanh(6 * x))
        ft, fs = feats(train_x), feats(test_x)
        k, taps = 3, 63
        ata = np.zeros((k * taps, k * taps)); atb = np.zeros(k * taps)
        for s in range(0, train_x.size, CHUNK):
            e = min(train_x.size, s + CHUNK)
            A = np.concatenate([lag_matrix(f, s, e, taps) for f in ft], axis=1)[m_train[s:e]]
            ata += A.T @ A; atb += A.T @ train_y[s:e].astype(np.float64)[m_train[s:e]]
        ata += 1e-9 * np.trace(ata) / (k * taps) * np.eye(k * taps)
        hh = np.linalg.solve(ata, atb).reshape(k, taps)
        bank_train[m_train] = sum(apply_fir(f, h1) for f, h1 in zip(ft, hh))[m_train]
        bank_test[m_test] = sum(apply_fir(f, h1) for f, h1 in zip(fs, hh))[m_test]
    result["gated_hammerstein_63"] = {"train_esr": esr(bank_train, train_y), "test_esr": esr(bank_test, test_y), "windows_101_200_seed0_esr": esr(bank_train[idx], train_y[idx])}
    print("gated hammerstein", result["gated_hammerstein_63"], flush=True)

    # H6: linear recoverability of residuals
    for label in preds_train:
        r_train = train_y.astype(np.float64) - preds_train[label]
        r_test = test_y.astype(np.float64) - preds_test[label]
        rows = {}
        for taps in (65, 257):
            h = ls_fir(train_x, r_train, taps)
            rows[f"fir_{taps}_from_input"] = {
                "train_residual_energy_removed": float(1 - np.sum((r_train - apply_fir(train_x, h)) ** 2) / np.sum(r_train**2)),
                "test_residual_energy_removed": float(1 - np.sum((r_test - apply_fir(test_x, h)) ** 2) / np.sum(r_test**2)),
                "test_esr_after_correction": esr(preds_test[label] + apply_fir(test_x, h), test_y),
            }
        # from the model's own output (post-linear correction)
        h = ls_fir(preds_train[label], r_train, 65)
        rows["fir_65_from_own_output"] = {
            "test_residual_energy_removed": float(1 - np.sum((r_test - apply_fir(preds_test[label], h)) ** 2) / np.sum(r_test**2)),
            "test_esr_after_correction": esr(preds_test[label] + apply_fir(preds_test[label], h), test_y),
        }
        result[f"{label}_residual_recovery"] = rows
        print(label, "residual recovery", rows, flush=True)
    dump("level_dependence.json", result)


if __name__ == "__main__":
    main()
