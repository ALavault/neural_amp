"""Analysis probes for audio-fidelity scores (product lane, no scientific claim).

A probe is an identification model fitted SEPARATELY to (x, y_ref) and (x, y_eval) on
estimation data A, then rendered on reading excitations B. The score compares the two
renderings, q_eval - q_ref, never the two fit errors: two different systems can have
the same fit error (F = x + 0.1 x^3 and G = x + 0.2 x^3 are both exact cubic
Hammersteins).

Probes here are linear in their parameters:
- `fir`: one causal FIR on x, a linear signature conditioned by the excitation (not
  "the linear part" of the device);
- `hammerstein`: parallel branches x^k, k = 1..degree, each through its own FIR.
Branch signals are scaled by their RMS on x_A, and the ridge is relative to the mean
diagonal of the Gram matrix: both depend on x_A only, so the fitted parameters are a
linear function of y, and fit(y_eval) - fit(y_ref) = fit(y_eval - y_ref) exactly.

Fit: least squares summed over the whole convolution support (rows 0 .. n + taps - 2,
y taken as zero past its end; the constant column only on rows 0 .. n - 1). The lagged
Gram matrix is then exactly block-Toeplitz and is built from FFT cross-correlations.
Rendering: causal convolution from a zero state, first n samples.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.linalg import solve
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class Probe:
    """Protocol of a linear-in-parameters probe; frozen before any test reading."""

    degree: int = 1  # 1: FIR probe; > 1: Hammerstein branches x^1 .. x^degree
    taps: int = 64
    ridge: float = 1e-6
    dc: bool = True

    @property
    def kind(self) -> str:
        return "fir" if self.degree == 1 else "hammerstein"

    def version(self) -> str:
        text = json.dumps({"kind": self.kind, **asdict(self)}, sort_keys=True)
        return f"{self.kind}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


@dataclass
class Fit:
    probe: Probe
    scales: np.ndarray  # RMS of each branch signal x^k on x_A
    theta: np.ndarray  # [dc] + degree * taps FIR coefficients


def branches(
    x: np.ndarray, degree: int, scales: np.ndarray | None = None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Branch signals x^k / scale_k; scales default to their RMS on this x."""
    powers = [x**k for k in range(1, degree + 1)]
    if scales is None:
        scales = np.array([max(np.sqrt(np.mean(p**2)), 1e-300) for p in powers])
    return [p / s for p, s in zip(powers, scales, strict=True)], scales


def _spectra(signals: list, size: int) -> list:
    return [rfft(s, size) for s in signals]


def normal_equations(probe: Probe, u: list, y: np.ndarray | None):
    """Gram matrix and, if y is given, Phi^T y for branch signals u."""
    n, m = len(u[0]), probe.taps
    size = next_fast_len(n + m)
    spectra = _spectra(u, size)
    lag = np.subtract.outer(np.arange(m), np.arange(m)) % size  # i - j
    width = len(u) * m + probe.dc
    gram = np.empty((width, width))
    first = int(probe.dc)
    for k, uk in enumerate(spectra):
        for j in range(k, len(u)):
            block = irfft(np.conj(uk) * spectra[j], size)[lag]
            rows, cols = first + k * m, first + j * m
            gram[rows : rows + m, cols : cols + m] = block
            gram[cols : cols + m, rows : rows + m] = block.T
    if probe.dc:
        gram[0, 0] = n
        for k, signal in enumerate(u):
            # sum over rows 0..n-1 of signal[row - i] = sum of signal[0 .. n-1-i]
            sums = np.concatenate(([0.0], np.cumsum(signal)))[n - np.arange(m)]
            gram[0, first + k * m : first + (k + 1) * m] = sums
            gram[first + k * m : first + (k + 1) * m, 0] = sums
    if y is None:
        return gram, None
    target = rfft(y, size)
    rhs = [irfft(np.conj(uk) * target, size)[:m] for uk in spectra]
    if probe.dc:
        rhs.insert(0, np.array([np.sum(y)]))
    return gram, np.concatenate(rhs)


def regularized(probe: Probe, gram: np.ndarray) -> np.ndarray:
    """Gram plus the relative ridge, on every column except the constant."""
    first = int(probe.dc)
    diagonal = np.diag(gram)[first:]
    out = gram.copy()
    out[first:, first:] += probe.ridge * diagonal.mean() * np.eye(len(diagonal))
    return out


def fit(probe: Probe, x: np.ndarray, y: np.ndarray) -> Fit:
    """Ridge least-squares fit of the probe on estimation data (x, y)."""
    u, scales = branches(x, probe.degree)
    gram, rhs = normal_equations(probe, u, y)
    theta = solve(regularized(probe, gram), rhs, assume_a="pos")
    return Fit(probe, scales, theta)


def contributions(result: Fit, x: np.ndarray) -> dict:
    """Rendered output of each branch (and the constant) on x, zero initial state."""
    probe, n = result.probe, len(x)
    u, _ = branches(x, probe.degree, result.scales)
    first, m = int(probe.dc), probe.taps
    out = {}
    if probe.dc:
        out["dc"] = np.full(n, result.theta[0])
    for k, signal in enumerate(u, start=1):
        h = result.theta[first + (k - 1) * m : first + k * m]
        out[f"order_{k}"] = fftconvolve(signal, h)[:n]
    return out


def render(result: Fit, x: np.ndarray) -> np.ndarray:
    """Probe output on x: the sum of its contributions."""
    return np.sum(list(contributions(result, x).values()), axis=0)
