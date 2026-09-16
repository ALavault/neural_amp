"""Reachable bound of the S3 t33 class from each checkpoint (H1), and linear LS bound
on the same 400 training windows. Read-only; CPU.

Block coordinate refit with pre-FIR, slow controller and residual TCN frozen at the
checkpoint: post-FIR by LS (closed form), spline (values, slopes) by LS (the Hermite
output is linear in them), pre-FIR by a damped Gauss-Newton step with backtracking.
Objective: MSE over the output regions of the 400 windows actually seen in training.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.linalg import solve_toeplitz

from common import (
    A2_RUNS,
    CONTEXT,
    OUTPUT,
    S3_RUNS,
    audio,
    band_table,
    dump,
    esr,
    load_model,
    predict_file,
    predict_windows,
    replay_starts,
    windows_for_steps,
)

CHUNK = 25
ITERATIONS = 8


def spline_basis(shaper, u: torch.Tensor) -> torch.Tensor:
    """Basis (34, N): output of the spline for one-hot values / slopes."""
    n = len(shaper.knots)
    values, slopes = shaper.values.data.clone(), shaper.slopes.data.clone()
    columns = []
    with torch.inference_mode():
        for k in range(n):
            shaper.values.data.zero_(); shaper.slopes.data.zero_(); shaper.values.data[k] = 1.0
            columns.append(shaper(u).clone())
        for k in range(n):
            shaper.values.data.zero_(); shaper.slopes.data.zero_(); shaper.slopes.data[k] = 1.0
            columns.append(shaper(u).clone())
    shaper.values.data.copy_(values); shaper.slopes.data.copy_(slopes)
    return torch.stack(columns)


def causal_conv(signal: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """signal (..., N), h (T,) with h[-1] the current-sample tap (CausalFIR convention)."""
    flat = signal.reshape(-1, 1, signal.shape[-1])
    padded = torch.nn.functional.pad(flat, (len(h) - 1, 0))
    return torch.nn.functional.conv1d(padded, h[None, None, :]).reshape(signal.shape)


class Refit:
    def __init__(self, model, wx: np.ndarray, wy: np.ndarray):
        self.model = model
        self.wx, self.wy = wx, wy
        self.core = model.core
        self.shaper = model.core.shaper
        self.drive = float(model.core.drive)
        self.offset = float(model.core.offset)
        self.gain = float(model.core.output_gain)
        # frozen per-window signals: input, slow modulation, residual (inert, kept fixed)
        self.chunks = []
        with torch.inference_mode():
            for i in range(0, len(wx), CHUNK):
                x = torch.from_numpy(wx[i : i + CHUNK])
                y = torch.from_numpy(wy[i : i + CHUNK]).double()
                modulation = model.slow(x)
                _, _, residual = model.forward_components(x)
                self.chunks.append((x, modulation, residual[..., -OUTPUT:].double(), y))
        self.target_energy = sum(float((c[3] ** 2).sum()) for c in self.chunks)

    # ----- forward pieces -------------------------------------------------------
    def pre_out(self, x, modulation, a):
        return self.drive * modulation[:, 0] * causal_conv(x, a) + self.offset + modulation[:, 1]

    def core_out(self, u, modulation, theta, h):
        s = self.spline_eval(u, theta)
        return self.gain * modulation[:, 2] * causal_conv(s, h)

    def spline_eval(self, u, theta):
        n = len(self.shaper.knots)
        self.shaper.values.data.copy_(theta[:n]); self.shaper.slopes.data.copy_(theta[n:])
        with torch.inference_mode():
            return self.shaper(u)

    def objective(self, a, theta, h) -> float:
        err = 0.0
        with torch.inference_mode():
            for x, modulation, residual, y in self.chunks:
                u = self.pre_out(x, modulation, a)
                out = self.core_out(u, modulation, theta, h)[..., -OUTPUT:].double() + residual
                err += float(((out - y) ** 2).sum())
        return err / self.target_energy

    # ----- block updates ----------------------------------------------------------
    def solve_post(self, a, theta, ridge=1e-9):
        T = len(self.core.post.coefficients)
        ata = torch.zeros(T, T, dtype=torch.float64); atb = torch.zeros(T, dtype=torch.float64)
        with torch.inference_mode():
            for x, modulation, residual, y in self.chunks:
                u = self.pre_out(x, modulation, a)
                s = self.spline_eval(u, theta)
                scale = (self.gain * modulation[:, 2])[..., -OUTPUT:]
                lags = torch.stack([torch.roll(s, T - 1 - k, dims=-1)[..., -OUTPUT:] for k in range(T)], -1)
                design = (lags * scale[..., None]).reshape(-1, T).double()
                b = (y - residual).reshape(-1)
                ata += design.T @ design; atb += design.T @ b
        ata += ridge * ata.diagonal().mean() * torch.eye(T, dtype=torch.float64)
        return torch.linalg.solve(ata, atb).float()

    def solve_spline(self, a, theta, h, ridge=1e-9):
        K = 2 * len(self.shaper.knots)
        ata = torch.zeros(K, K, dtype=torch.float64); atb = torch.zeros(K, dtype=torch.float64)
        with torch.inference_mode():
            for x, modulation, residual, y in self.chunks:
                u = self.pre_out(x, modulation, a)
                basis = spline_basis(self.shaper, u)  # (K, B, N)
                filtered = causal_conv(basis.reshape(-1, u.shape[-1]), h).reshape(basis.shape)[..., -OUTPUT:]
                scale = (self.gain * modulation[:, 2])[..., -OUTPUT:]
                design = (filtered * scale[None]).reshape(K, -1).double()
                b = (y - residual).reshape(-1)
                ata += design @ design.T; atb += design @ b
        ata += ridge * ata.diagonal().mean() * torch.eye(K, dtype=torch.float64)
        return torch.linalg.solve(ata, atb).float()

    def gauss_newton_pre(self, a, theta, h, current: float):
        """One damped Gauss-Newton step on the pre-FIR with backtracking on the true objective."""
        T = len(a)
        n = len(self.shaper.knots)
        jtj = torch.zeros(T, T, dtype=torch.float64); jtr = torch.zeros(T, dtype=torch.float64)
        with torch.inference_mode():
            for x, modulation, residual, y in self.chunks:
                u = self.pre_out(x, modulation, a)
                self.shaper.values.data.copy_(theta[:n]); self.shaper.slopes.data.copy_(theta[n:])
                ds = self.shaper.derivative(u) * self.drive * modulation[:, 0]
                xl = torch.stack([torch.roll(x, T - 1 - k, dims=-1) for k in range(T)], 0)  # (T,B,N)
                jac = causal_conv((ds[None] * xl).reshape(-1, x.shape[-1]), h).reshape(T, *x.shape)[..., -OUTPUT:]
                jac = (jac * (self.gain * modulation[:, 2])[None, :, -OUTPUT:]).reshape(T, -1).double()
                out = self.core_out(u, modulation, theta, h)[..., -OUTPUT:].double() + residual
                r = (y - out).reshape(-1)
                jtj += jac @ jac.T; jtr += jac @ r
        for damping in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0):
            step = torch.linalg.solve(jtj + damping * jtj.diagonal().mean() * torch.eye(T, dtype=torch.float64), jtr).float()
            for alpha in (1.0, 0.5, 0.25):
                candidate = a + alpha * step
                value = self.objective(candidate, theta, h)
                if value < current:
                    return candidate, value
        return a, current

    def run(self):
        a = self.core.pre.coefficients.data.clone()
        theta = torch.cat((self.shaper.values.data, self.shaper.slopes.data)).clone()
        h = self.core.post.coefficients.data.clone()
        trace = {"checkpoint": self.objective(a, theta, h)}
        h1 = self.solve_post(a, theta)
        trace["post_only"] = self.objective(a, theta, h1)
        theta1 = self.solve_spline(a, theta, h)
        trace["spline_only"] = self.objective(a, theta1, h)
        current = trace["post_only"]; h = h1
        history = []
        for it in range(ITERATIONS):
            theta = self.solve_spline(a, theta, h); v1 = self.objective(a, theta, h)
            h = self.solve_post(a, theta); v2 = self.objective(a, theta, h)
            a, v3 = self.gauss_newton_pre(a, theta, h, v2)
            history.append({"iteration": it + 1, "after_spline": v1, "after_post": v2, "after_pre_gn": v3})
            print(f"  iter {it+1}: spline {v1:.5f} post {v2:.5f} pre-GN {v3:.5f}", flush=True)
            if current - v3 < 1e-5:
                current = v3
                break
            current = v3
        trace["alternated"] = current
        trace["history"] = history
        # write back into the model
        n = len(self.shaper.knots)
        self.core.pre.coefficients.data.copy_(a)
        self.core.post.coefficients.data.copy_(h)
        self.shaper.values.data.copy_(theta[:n]); self.shaper.slopes.data.copy_(theta[n:])
        trace["fitted_parameters"] = {
            "pre_max_abs_delta_from_delta": float((a - torch.eye(1, len(a))[0].flip(0)).abs().max()),
            "post_max_abs_delta_from_delta": float((h - torch.eye(1, len(h))[0].flip(0)).abs().max()),
            "spline_values": [round(float(v), 4) for v in theta[:n]],
            "spline_slopes": [round(float(v), 4) for v in theta[n:]],
        }
        return trace


def linear_bound_on_windows(wx, wy, wx_eval, wy_eval, test_x, test_y, taps):
    """LS FIR (taps) fitted on the output regions of the 400 windows (history from context)."""
    r_xx = np.zeros(taps); r_xy = np.zeros(taps)
    for x, y in zip(wx, wy):
        xd = x.astype(np.float64); yd = y.astype(np.float64)
        seg = xd[CONTEXT - taps + 1 :]
        for k in range(taps):
            lag = seg[taps - 1 - k : taps - 1 - k + OUTPUT]
            r_xy[k] += np.dot(lag, yd)
            if k == 0:
                base = lag
            r_xx[k] += np.dot(base, lag)
    r_xx[0] *= 1 + 1e-9
    h = solve_toeplitz(r_xx, r_xy)

    def apply(x):
        return np.convolve(x.astype(np.float64), h)[: x.size]

    def windows_esr(wxs, wys):
        err = 0.0; en = 0.0
        for x, y in zip(wxs, wys):
            p = apply(x)[-OUTPUT:]
            err += np.sum((p - y) ** 2); en += np.sum(y.astype(np.float64) ** 2)
        return err / en

    return {
        "taps": taps,
        "fit_windows_esr_1_400": windows_esr(wx, wy),
        "windows_101_200_esr": windows_esr(wx_eval, wy_eval),
        "test_esr": esr(apply(test_x), test_y),
        "max_abs_delta_from_delta": float(np.max(np.abs(h - np.eye(1, taps, 0)[0]))),
    }


def main():
    train_x, train_y = audio("train")
    test_x, test_y = audio("test")
    result = {}
    for seed, run_id in S3_RUNS.items():
        print(run_id, flush=True)
        model, code, _ = load_model(run_id)
        starts = replay_starts(seed, train_x.size)
        wx, wy, _ = windows_for_steps(train_x, train_y, starts, 1, 200)
        wx_eval, wy_eval, _ = windows_for_steps(train_x, train_y, starts, 101, 200)
        row = {"run_id": run_id}
        if seed == 0:
            result["linear_ls_on_400_windows"] = {
                f"taps_{t}": linear_bound_on_windows(wx, wy, wx_eval, wy_eval, test_x, test_y, t) for t in (33, 65, 97)
            }
            print(result["linear_ls_on_400_windows"], flush=True)
        row["checkpoint_windows_101_200_esr"] = esr(predict_windows(model, code, wx_eval).ravel(), wy_eval.ravel())
        refit = Refit(model, wx, wy)
        row["refit_objective_windows_1_400"] = refit.run()
        pred_eval = predict_windows(model, code, wx_eval)
        row["refit_windows_101_200_esr"] = esr(pred_eval.ravel(), wy_eval.ravel())
        row["refit_windows_101_200_bands"] = band_table(pred_eval.ravel().astype(np.float64) - wy_eval.ravel(), wy_eval.ravel())
        pred_test = predict_file(model, code, test_x)
        row["refit_test_esr"] = esr(pred_test, test_y)
        row["refit_test_bands"] = band_table(pred_test.astype(np.float64) - test_y, test_y)
        row["refit_test_residual_energy_ratio"] = float(model.energy_ratio(torch.from_numpy(test_x)))
        print({k: v for k, v in row.items() if isinstance(v, float)}, flush=True)
        result[f"S3_t33_seed{seed}"] = row
    dump("class_bounds.json", result)


if __name__ == "__main__":
    main()
