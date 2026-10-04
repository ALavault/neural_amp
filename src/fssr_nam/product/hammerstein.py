"""Parallel Hammerstein identification from an exponential sine sweep.

Model: y = sum_{n=1..K} g_n * x^n, each power of the input followed by a linear
filter g_n. Deconvolving the output by the sweep separates, in time, the response to
each harmonic k of the sweep (Farina; Novak et al.): the k-th lands L ln k before the
linear response, L = T / ln(f1/f0). The harmonic responses D_k(F) mix the filters,
D_k(F) = sum_n M_kn(F) G_n(F).

M is not derived from trigonometry here: it is measured by passing the pure powers
x^n of the same reference through the same chain (deconvolution, windows, any
decimation done by the caller). Phases of a non-synchronised sweep, fades and windows
are then the same in M and in D, and cancel in the per-frequency solve.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

PRE_S, LENGTH_S = 0.01, 0.1  # window around each harmonic response


@dataclass
class Sweep:
    """Exponential sweep f0 -> f1 over T seconds (product_valvestate_session.sweep)."""

    f0: float = 20.0
    f1: float = 20_000.0
    seconds: float = 5.0

    @property
    def lag(self) -> float:
        return self.seconds / math.log(self.f1 / self.f0)


def deconvolve(y: np.ndarray, ref: np.ndarray, rate: int, sweep: Sweep) -> np.ndarray:
    """y deconvolved by ref (regularised spectral division, sweep band only).
    Returned circularly: negative times wrap to the end."""
    n = 1 << (len(y) + len(ref) - 1).bit_length()
    big_y, big_r = np.fft.rfft(y, n), np.fft.rfft(ref, n)
    freqs = np.fft.rfftfreq(n, 1 / rate)
    power = np.abs(big_r) ** 2
    inverse = np.conj(big_r) / (power + 1e-6 * power.max())
    inverse[(freqs < sweep.f0) | (freqs > sweep.f1)] = 0
    return np.fft.irfft(big_y * inverse, n)


def harmonic_spectra(
    h: np.ndarray, rate: int, sweep: Sweep, order: int, nfft: int
) -> np.ndarray:
    """(order, nfft//2+1) spectra of the windowed responses to harmonics 1..order."""
    pre, length = int(PRE_S * rate), int(LENGTH_S * rate)
    out = []
    for k in range(1, order + 1):
        start = -round(sweep.lag * math.log(k) * rate) - pre
        idx = (np.arange(length) + start) % len(h)
        out.append(np.fft.rfft(h[idx], nfft))
    return np.array(out)


@dataclass
class Hammerstein:
    freqs: np.ndarray  # Hz, rfft grid of nfft at rate
    g: np.ndarray  # (order, len(freqs)) complex, filter of x^n
    d: np.ndarray  # (order, len(freqs)) complex, measured harmonic responses


def calibration(
    ref: np.ndarray,
    to_rate: Callable[[np.ndarray], np.ndarray],
    rate: int,
    sweep: Sweep,
    order: int,
    nfft: int,
) -> np.ndarray:
    """M (len(freqs), order k, order n): harmonic responses of the pure powers ref^n.
    `ref` is the input at its native rate; `to_rate` brings a signal to the analysis
    rate the way the measured outputs were (identity, or a decimation)."""
    target = to_rate(ref)
    m = []
    for n in range(1, order + 1):
        h = deconvolve(to_rate(ref**n), target, rate, sweep)
        m.append(harmonic_spectra(h, rate, sweep, order, nfft))
    return np.transpose(np.array(m), (2, 1, 0))  # freq, k, n


def identify(
    y: np.ndarray,
    ref: np.ndarray,
    to_rate: Callable[[np.ndarray], np.ndarray],
    rate: int,
    sweep: Sweep,
    order: int = 5,
    nfft: int = 8192,
    m: np.ndarray | None = None,
) -> Hammerstein:
    """Filters G_1..G_order of the parallel Hammerstein model of y (at `rate`) driven
    by `ref` (native rate). Pass `m` to reuse a calibration of the same ref."""
    if m is None:
        m = calibration(ref, to_rate, rate, sweep, order, nfft)
    d = harmonic_spectra(
        deconvolve(y, to_rate(ref), rate, sweep), rate, sweep, order, nfft
    )
    g = np.array([np.linalg.lstsq(m[i], d[:, i], rcond=1e-9)[0] for i in range(len(m))])
    return Hammerstein(np.fft.rfftfreq(nfft, 1 / rate), g.T, d)
