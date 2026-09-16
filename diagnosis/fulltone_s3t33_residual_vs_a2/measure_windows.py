"""Harness check, train-window vs test gap, band and level-tranche tables, S3 internals.

Read-only on runs; CPU. Outputs windows_gap.json and per-window ESR arrays (npz).
"""

from __future__ import annotations

import numpy as np
import torch

from common import (
    A2_RUNS,
    OUT,
    OUTPUT,
    S3_17_RUNS,
    S3_RUNS,
    audio,
    band_table,
    causal_envelope,
    dump,
    esr,
    load_model,
    predict_windows,
    replay_starts,
    saved_test_prediction,
    windows_for_steps,
)

TRANCHES = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 10.0))


def tranche_table(pred, target, env):
    rows = {}
    err = (pred.astype(np.float64) - target) ** 2
    tgt = target.astype(np.float64) ** 2
    for lo, hi in TRANCHES:
        m = (env >= lo) & (env < hi)
        rows[f"{lo}-{hi}"] = {
            "samples": int(m.sum()),
            "local_esr": float(err[m].sum() / tgt[m].sum()),
            "error_share": float(err[m].sum() / err.sum()),
            "target_energy_share": float(tgt[m].sum() / tgt.sum()),
        }
    return rows


def segment_gain_esr(pred, target, seg=2400):
    p = pred.astype(np.float64)
    t = target.astype(np.float64)
    n = p.size // seg * seg
    P, T = p[:n].reshape(-1, seg), t[:n].reshape(-1, seg)
    g = (P * T).sum(1) / np.maximum((P * P).sum(1), 1e-12)
    corrected = (P * g[:, None]).ravel()
    return esr(corrected, t[:n]), float(np.std(g)), float(np.median(g))


def s3_internals(model, x: np.ndarray) -> dict:
    with torch.inference_mode():
        tensor = torch.from_numpy(x)
        modulation = model.slow(tensor[None, :])[0]
        pre = model.core.pre(tensor)
        u = model.core.drive * modulation[0] * pre + model.core.offset + modulation[1]
        out, core, res = model.forward_components(tensor)
    knots = model.core.shaper.knots.numpy()
    hist, _ = np.histogram(u.numpy(), bins=np.concatenate(([-10], knots, [10])))
    return {
        "residual_energy_ratio": float((res**2).sum() / (out**2).sum()),
        "slow_drive_factor_range": [float(modulation[0].min()), float(modulation[0].max())],
        "slow_offset_delta_range": [float(modulation[1].min()), float(modulation[1].max())],
        "slow_gain_factor_range": [float(modulation[2].min()), float(modulation[2].max())],
        "spline_input_abs_quantiles_50_90_99": [float(v) for v in np.quantile(np.abs(u.numpy()), [0.5, 0.9, 0.99])],
        "spline_input_histogram_by_knot_interval": {
            f"{lo:+.2f}..{hi:+.2f}": int(c)
            for lo, hi, c in zip(np.concatenate(([-10], knots)), np.concatenate((knots, [10])), hist)
        },
        "core_ablation_no_residual_esr_delta": None,
    }


def parameter_deltas(model) -> dict:
    pre = model.core.pre.coefficients.detach().numpy()
    post = model.core.post.coefficients.detach().numpy()
    d_pre = pre.copy(); d_pre[-1] -= 1.0
    d_post = post.copy(); d_post[-1] -= 1.0
    knots = model.core.shaper.knots.numpy()
    values = model.core.shaper.values.detach().numpy() - knots
    slopes = model.core.shaper.slopes.detach().numpy() - 1.0
    return {
        "pre_fir_l2_delta_from_delta": float(np.linalg.norm(d_pre)),
        "pre_fir_max_abs_delta": float(np.abs(d_pre).max()),
        "post_fir_l2_delta_from_delta": float(np.linalg.norm(d_post)),
        "post_fir_max_abs_delta": float(np.abs(d_post).max()),
        "spline_value_delta_by_knot": {f"{k:+.2f}": round(float(v), 4) for k, v in zip(knots, values)},
        "spline_slope_delta_by_knot": {f"{k:+.2f}": round(float(v), 4) for k, v in zip(knots, slopes)},
        "drive": float(model.core.drive), "offset": float(model.core.offset), "output_gain": float(model.core.output_gain),
        "residual_scale": float(model.residual.residual_scale),
        "residual_output_projection_norm": float(model.residual.output_projection.weight.norm()),
    }


def main():
    train_x, train_y = audio("train")
    test_x, test_y = audio("test")
    env_test = causal_envelope(test_x)
    result = {"identity_test_esr": esr(test_x, test_y), "identity_test_bands": band_table(test_x.astype(np.float64) - test_y, test_y)}
    per_window = {}
    for label, runs in (("S3_t33", S3_RUNS), ("A2", A2_RUNS), ("S3_t17", S3_17_RUNS)):
        for seed, run_id in runs.items():
            model, code, _ = load_model(run_id)
            starts = replay_starts(seed, train_x.size)
            row = {"run_id": run_id}
            for name, (first, last) in (("steps_1_100", (1, 100)), ("steps_101_200", (101, 200))):
                wx, wy, _ = windows_for_steps(train_x, train_y, starts, first, last)
                pred = predict_windows(model, code, wx)
                row[f"train_window_esr_{name}"] = esr(pred.ravel(), wy.ravel())
                per_window[f"{label}_seed{seed}_{name}"] = np.array([esr(p, t) for p, t in zip(pred, wy)])
                if name == "steps_101_200":
                    row["train_window_bands_101_200"] = band_table(pred.ravel().astype(np.float64) - wy.ravel(), wy.ravel())
                    env_w = np.concatenate([causal_envelope(w)[-OUTPUT:] for w in wx])
                    row["train_window_tranches_101_200"] = tranche_table(pred.ravel(), wy.ravel(), env_w)
            pred_test = saved_test_prediction(run_id)
            row["test_esr_saved"] = esr(pred_test, test_y)
            row["test_bands"] = band_table(pred_test.astype(np.float64) - test_y, test_y)
            row["test_tranches"] = tranche_table(pred_test, test_y, env_test)
            g_esr, g_std, g_med = segment_gain_esr(pred_test, test_y)
            row["test_esr_after_50ms_segment_gain"] = g_esr
            row["segment_gain_std"] = g_std
            row["segment_gain_median"] = g_med
            row["test_esr_after_global_gain"] = float(1 - (np.dot(pred_test.astype(np.float64), test_y) ** 2) / (np.dot(pred_test, pred_test) * np.dot(test_y, test_y)))
            if code == "S3":
                row["internals_test"] = s3_internals(model, test_x)
                row["parameter_deltas"] = parameter_deltas(model)
            result[f"{label}_seed{seed}"] = row
            print(label, seed, {k: (round(v, 4) if isinstance(v, float) else "") for k, v in row.items() if isinstance(v, float)})
    # gap decomposition per seed (same seed pairs) and medians
    gaps = {}
    for seed in range(3):
        s3, a2 = result[f"S3_t33_seed{seed}"], result[f"A2_seed{seed}"]
        gaps[f"seed{seed}"] = {
            "gap_train_windows_101_200": s3["train_window_esr_steps_101_200"] - a2["train_window_esr_steps_101_200"],
            "gap_test": s3["test_esr_saved"] - a2["test_esr_saved"],
            "s3_transfer": s3["test_esr_saved"] - s3["train_window_esr_steps_101_200"],
            "a2_transfer": a2["test_esr_saved"] - a2["train_window_esr_steps_101_200"],
        }
    med = lambda key, label: float(np.median([result[f"{label}_seed{s}"][key] for s in range(3)]))
    gaps["medians"] = {
        "s3_t33_test": med("test_esr_saved", "S3_t33"), "a2_test": med("test_esr_saved", "A2"),
        "s3_t33_train_101_200": med("train_window_esr_steps_101_200", "S3_t33"),
        "a2_train_101_200": med("train_window_esr_steps_101_200", "A2"),
        "s3_t33_train_1_100": med("train_window_esr_steps_1_100", "S3_t33"),
        "a2_train_1_100": med("train_window_esr_steps_1_100", "A2"),
    }
    result["gaps"] = gaps
    dump("windows_gap.json", result)
    np.savez(OUT / "per_window_esr.npz", **per_window)
    print(gaps)


if __name__ == "__main__":
    main()
