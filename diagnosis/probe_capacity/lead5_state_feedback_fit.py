"""Helpers of lead 5 (state and feedback): a multi-target ridge fit on frozen features,
level envelopes, two synthetic calibration devices, a NARX free-run loop.

MultiFit generalizes the library's normal equations (fssr_nam.product.probes
.normal_equations: Toeplitz Gram from FFT cross-correlations, a constant, a ridge
relative to the mean Gram diagonal) to a different FIR length per feature, and solves
several targets against one Gram. Its coefficients are G^-1 Phi^T y: linear in the
target as long as the features do not depend on it. With features x**1..x**5 and 256
taps it is Hammerstein 5 (checked against the harness in lead5_state_feedback.py).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import fft
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

FS = 48_000
WORKERS = 3  # FFT threads; the callers keep at most 3 threads in total


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


class MultiFit:
    """q = c + sum_k h_k * (f_k / s_k), h_k causal FIR of taps[k] samples, s_k the RMS
    of f_k on the fit section; ridge relative to the mean Gram diagonal (constant not
    penalized), exactly as probes.regularized."""

    def __init__(self, features: list[np.ndarray], taps: list[int], ridge=1e-4):
        self.taps = list(taps)
        self.scales = [max(rms(f), 1e-300) for f in features]
        u = [np.asarray(f) / s for f, s in zip(features, self.scales, strict=True)]
        n = self.n = len(u[0])
        if n < max(self.taps):
            raise ValueError(f"fit section ({n} samples) shorter than the FIR")
        self.size = fft.next_fast_len(n + max(self.taps))
        with fft.set_workers(WORKERS):
            self.spectra = [fft.rfft(v, self.size) for v in u]
        self.first = np.concatenate(([1], 1 + np.cumsum(self.taps)[:-1]))
        width = 1 + sum(self.taps)
        gram = np.empty((width, width))
        gram[0, 0] = n
        for k, v in enumerate(u):
            sums = np.concatenate(([0.0], np.cumsum(v)))[n - np.arange(self.taps[k])]
            a = self.first[k]
            gram[0, a : a + self.taps[k]] = sums
            gram[a : a + self.taps[k], 0] = sums
        with fft.set_workers(WORKERS):
            for k, sk in enumerate(self.spectra):
                for j in range(k, len(u)):
                    c = fft.irfft(np.conj(sk) * self.spectra[j], self.size)
                    lag = np.subtract.outer(
                        np.arange(self.taps[k]), np.arange(self.taps[j])
                    )
                    block = c[lag % self.size]
                    a, b = self.first[k], self.first[j]
                    gram[a : a + self.taps[k], b : b + self.taps[j]] = block
                    gram[b : b + self.taps[j], a : a + self.taps[k]] = block.T
        diagonal = np.diag(gram)[1:]
        gram[1:, 1:] += ridge * diagonal.mean() * np.eye(len(diagonal))
        self.factor = cho_factor(gram)

    def solve(self, target: np.ndarray) -> np.ndarray:
        """Coefficients for one target on the fit section (linear in the target)."""
        target = np.asarray(target, dtype=np.float64)
        with fft.set_workers(WORKERS):
            spectrum = fft.rfft(target, self.size)
            rhs = [np.array([target.sum()])] + [
                fft.irfft(np.conj(sk) * spectrum, self.size)[:m]
                for sk, m in zip(self.spectra, self.taps, strict=True)
            ]
        return cho_solve(self.factor, np.concatenate(rhs))

    def parts(
        self, theta: np.ndarray, features: list[np.ndarray]
    ) -> tuple[float, list[np.ndarray]]:
        """Constant and per-feature contributions on another section (zero state)."""
        out = []
        for k, f in enumerate(features):
            a = self.first[k]
            h = theta[a : a + self.taps[k]]
            out.append(fftconvolve(np.asarray(f) / self.scales[k], h)[: len(f)])
        return float(theta[0]), out

    def predict(self, theta: np.ndarray, features: list[np.ndarray]) -> np.ndarray:
        c, parts = self.parts(theta, features)
        return c + np.sum(parts, axis=0)


# ---------------------------------------------------------------- envelopes


def moving_power(v: np.ndarray, width: int, causal: bool = True) -> np.ndarray:
    """Mean of v**2 over the last `width` samples, current one included (causal), or
    over the next `width` samples (anticausal); samples outside v count as silence."""
    if not causal:
        return moving_power(np.asarray(v)[::-1], width)[::-1]
    p = np.square(np.asarray(v, dtype=np.float64))
    c = np.concatenate(([0.0], np.cumsum(p)))
    i = np.arange(1, len(p) + 1)
    return (c[i] - c[np.maximum(i - width, 0)]) / width


def level_db(power: np.ndarray, floor_db: float = -100.0) -> np.ndarray:
    return 10 * np.log10(np.maximum(power, 10 ** (floor_db / 10)))


def delayed(v: np.ndarray, d: int, fill: float) -> np.ndarray:
    """v delayed by d >= 0 samples, `fill` before its start."""
    out = np.full(len(v), fill)
    out[d:] = v[: len(v) - d]
    return out


# ---------------------------------------------------------------- synthetic devices


def device(x: np.ndarray, tau_s: float | None) -> np.ndarray:
    """Synthetic calibration device at 48 kHz (not a model of Nolly). High-pass
    150 Hz, drive 34 dB, tanh, low-pass 5 kHz, coupling high-pass 80 Hz, second tanh,
    low-pass 6 kHz: memory of a few ms, no slow state (tau_s None). With tau_s, the
    drive is divided by 1 + 30 * env, env = RMS of the high-passed input through a
    one-pole of time constant tau_s: a slow, input-driven state of known time
    constant (a feed-forward sag/compressor)."""
    v = sosfilt(butter(1, 150, "high", fs=FS, output="sos"), x)
    drive = np.full(len(v), 10 ** (34 / 20))
    if tau_s is not None:
        a = np.exp(-1 / (tau_s * FS))
        drive = drive / (1 + 30 * np.sqrt(lfilter([1 - a], [1, -a], v**2)))
    s = np.tanh(drive * v)
    s = sosfilt(butter(1, 5_000, "low", fs=FS, output="sos"), s)
    s = sosfilt(butter(1, 80, "high", fs=FS, output="sos"), s)
    return sosfilt(butter(2, 6_000, "low", fs=FS, output="sos"), 0.3 * np.tanh(3 * s))


# ---------------------------------------------------------------- NARX free run


@njit(cache=False)
def narx_free_run(u: np.ndarray, a: np.ndarray, limit: float) -> tuple:
    """q[n] = u[n] + sum_m sum_i a[m, i] * q[n-1-i]**(m+1), zero initial state.
    Stops at the first |q| > limit (or non-finite); returns (q, stop index or -1)."""
    q = np.zeros(len(u))
    powers, lags = a.shape
    for n in range(len(u)):
        v = u[n]
        for i in range(lags):
            if n - 1 - i < 0:
                break
            p = q[n - 1 - i]
            pm = p
            for m in range(powers):
                v += a[m, i] * pm
                pm *= p
        if not np.isfinite(v) or abs(v) > limit:
            return q, n
        q[n] = v
    return q, -1
