#!/usr/bin/env python3
"""Lead 4: a measured linear ceiling from the multisines (best linear approximation).

Question: is the frozen FIR 256 capacity on di_test predicted by the share of output
power that the multisine at the same input RMS puts into nonlinear distortion (then
the ceiling is a property of the device and a probe can be judged against it), or does
the DI (non-Gaussian, high crest factor, non-stationary) behave differently?

Instruments, all at 48 kHz on the harness cache (cut then decimated per section):
- multisine framework (Pintelon and Schoukens): odd random-phase multisine, 1 Hz lines,
  4 realizations x periods 2-3 (period 1 is the transient). Per take and level, output
  power split into the BLA part, odd distortion on the excited lines (unbiased over
  realizations), odd detection lines, even lines, out-of-band bins (< 21 Hz, > 20 kHz)
  and noise (period 2 against period 3). D = 1 - BLA share; c = sqrt(D). D is the
  non-BLA power: nonlinear distortion plus the noise left in the 2-period mean (at
  most noise_share / 2, measured below 2e-5 here). Out of
  sample: BLA from 3 realizations read on the 4th (LOO), and the frozen `fir` spec
  (256 taps, relative ridge 1e-2, constant) fitted LOO on the same rows.
- DI at matched RMS: 50 ms windows of x (harness grid on B, from 0 on A) in bands of
  +-3 dB around -30, -24, -18 dBFS. Residual power share per band of the frozen probes
  (cached predictions, fitted on A), of a level-conditional FIR (frozen `fir` spec,
  fitted on the A rows of that band only, read on the B windows of the band), of the
  multisine BLA used as a filter (no DI fit at all) and of the multisine FIR.
- cross-checks: DI band FIR read on the multisine; pink noise -30 dBFS (FIR fitted on
  its first half, read on the second); B windows split by envelope trend (x only).

B is never fitted: every model read on B is fitted on A or on the multisines. All
probes here have frozen features (linear in y); the power shares are a ceiling, not a
score. EXCLUDED takes (gate 4) are measured and flagged.

    uv run python diagnosis/probe_capacity/lead4_bla_ceiling.py [--takes 20 30 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft
from scipy.linalg import solve
from scipy.signal import correlate, fftconvolve
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as h  # noqa: E402

nr = h.nr
LEVELS = (-30, -24, -18)  # multisine input RMS, dBFS
HALF_BAND = 3.0  # DI bands: window input RMS in [L - 3, L + 3) dB
NARROW = 1.0  # narrow bands [L - 1, L + 1) dB: within-band level compromise test
SEGMENT = h.FS  # local fits: 1 s segments of A, in sample (257 parameters / 48 000)
PERIOD = h.FS  # multisine period at 48 kHz (1 s, 1 Hz lines)
REALIZATIONS = (1, 2, 3, 4)
SPEC = h.FROZEN["fir"]  # 256 taps, relative ridge 1e-2, constant
TAPS, RIDGE = SPEC.taps, SPEC.ridge
ACAUSAL = 2400  # BLA filter: negative lags kept (h[-2400:]) when applied to the DI
TREND_DB, TREND_WINDOWS = 1.0, 4  # window level vs the mean power of the 4 before
CHUNK = 50_000  # rows per block of the explicit Gram
PROBES = ("fir", "hammerstein_5", "volterra_laguerre_3")
H2_PROBES = PROBES[1:]
PINK = "pink_noise_-30dBFS_rms"
OUT = HERE / "lead4_bla_ceiling.json"

LINE = np.arange(PERIOD // 2 + 1)
EXCITED = (LINE >= 21) & (LINE <= 19_999) & (LINE % 2 == 1) & (LINE % 8 != 3)
ODD_DET = (LINE >= 21) & (LINE <= 19_999) & (LINE % 8 == 3)
EVEN = (LINE >= 22) & (LINE <= 19_998) & (LINE % 2 == 0)
OOB = ~(EXCITED | ODD_DET | EVEN)
WEIGHT = np.where((LINE == 0) | (LINE == PERIOD // 2), 1.0, 2.0)  # one-sided rfft


def power(spectrum: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean square per period carried by the rfft bins of `mask` (last axis)."""
    w = WEIGHT if mask is None else np.where(mask, WEIGHT, 0.0)
    return (w * np.abs(spectrum) ** 2).sum(-1) / PERIOD**2


def section(tag: int | str, name: str) -> np.ndarray:
    return np.asarray(h.load(tag, (name,))[name])


def multisine_name(level: int, r: int) -> str:
    return f"multisine_{level}dBFS_rms_r{r}"


# ---------------------------------------------------------------- multisine framework


def periods(tag: int | str, level: int) -> np.ndarray:
    """rfft of periods 2-3 of every realization: (realization, period, line)."""
    return np.stack(
        [
            fft.rfft(
                section(tag, multisine_name(level, r))[PERIOD:].reshape(-1, PERIOD)
            )
            for r in REALIZATIONS
        ]
    )


def decomposition(x: np.ndarray, y: np.ndarray) -> dict:
    """Power shares of the output of one take at one level (x, y from `periods`)."""
    m = len(REALIZATIONS)
    u, y_mean = x.mean(1), y.mean(1)
    total = power(y).mean()
    g = y_mean[:, EXCITED] / u[:, EXCITED]
    g_bla, g_var = g.mean(0), g.var(0, ddof=1)
    distortion = y_mean[:, EXCITED] - g_bla * u[:, EXCITED]
    noise = power(y[:, 0] - y[:, 1]).mean() / 2  # per-period noise variance
    exc = 2 * np.abs(distortion) ** 2 * m / (m - 1)  # unbiased S_exc per line
    lin = 2 * (np.abs(g_bla) ** 2 - g_var / m) * np.abs(u[:, EXCITED]) ** 2
    shares = {
        "bla": lin.sum(-1).mean() / PERIOD**2 / total,
        "odd_excited": exc.sum(-1).mean() / PERIOD**2 / total,
        "odd_detection": power(y_mean, ODD_DET).mean() / total,
        "even": power(y_mean, EVEN).mean() / total,
        "out_of_band": power(y_mean, OOB).mean() / total,
        "noise_in_mean": noise / 2 / total,  # what the 2-period mean still holds
    }
    loo = []  # BLA from the other realizations, read on every period of this one
    for k in range(m):
        g_other = np.delete(g, k, axis=0).mean(0)
        residual = y[k].copy()
        residual[:, EXCITED] -= g_other * x[k][:, EXCITED]
        loo.append(power(residual).mean())
    d_in, d_loo = 1 - shares["bla"], float(np.mean(loo)) / total
    det_level = power(y_mean, ODD_DET) / ODD_DET.sum()
    return {
        "output_dbfs": 10 * np.log10(total),
        "shares": shares,
        "closure": sum(shares.values()),
        "noise_share": noise / total,
        "D": d_in,
        "c": np.sqrt(d_in),
        "D_loo": d_loo,
        "c_loo": np.sqrt(d_loo),
        "odd_db": float(np.mean(10 * np.log10(det_level / exc_level(y_mean)))),
        "even_db": float(
            np.mean(10 * np.log10(power(y_mean, EVEN) / EVEN.sum() / exc_level(y_mean)))
        ),
        "drift": float(
            max(nr.rms(fft.irfft(a[0] - a[1])) / nr.rms(fft.irfft(a)) for a in y)
        ),
        "bla_lines": g_bla,
    }


def exc_level(y_mean: np.ndarray) -> np.ndarray:
    return power(y_mean, EXCITED) / EXCITED.sum()


def bla_filter(g_bla: np.ndarray) -> np.ndarray:
    """Two-sided impulse response (ACAUSAL negative lags first) of the BLA measured on
    the excited lines, interpolated on the other lines and held outside 21-19999 Hz."""
    k = LINE[EXCITED]
    full = np.interp(LINE, k, g_bla.real) + 1j * np.interp(LINE, k, g_bla.imag)
    full[-1] = full[-1].real
    return np.roll(fft.irfft(full, PERIOD), ACAUSAL)


def apply_filter(two_sided: np.ndarray, x: np.ndarray) -> np.ndarray:
    return fftconvolve(x, two_sided)[ACAUSAL : ACAUSAL + len(x)]


# ---------------------------------------------------------------- FIR on chosen rows


def gram(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """sum over rows n of r_n r_n^T, r_n = [1, x[n], x[n-1], ..., x[n-255]]."""
    view = sliding_window_view(np.concatenate((np.zeros(TAPS - 1), x)), TAPS)
    out = np.zeros((TAPS + 1, TAPS + 1))
    for start in range(0, len(rows), CHUNK):
        block = view[rows[start : start + CHUNK]][:, ::-1]
        design = np.hstack((np.ones((len(block), 1)), block))
        out += design.T @ design
    return out


def rhs(x_spectrum: np.ndarray, size: int, y: np.ndarray, mask: np.ndarray):
    """[sum y, sum_n y[n] x[n - j]] over the rows of `mask` (FFT correlation)."""
    z = np.where(mask, y, 0.0)
    corr = fft.irfft(fft.rfft(z, size, workers=3) * np.conj(x_spectrum), size)
    return np.concatenate(([z.sum()], corr[:TAPS]))


def solve_fir(g: np.ndarray, b: np.ndarray) -> np.ndarray:
    reg = g.copy()
    reg[1:, 1:] += RIDGE * np.diag(g)[1:].mean() * np.eye(TAPS)
    return solve(reg, b, assume_a="pos")


def render(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return theta[0] + fftconvolve(x, theta[1:])[: len(x)]


class RowFit:
    """Frozen `fir` spec fitted on chosen rows of one input; Gram built once."""

    def __init__(self, x: np.ndarray, mask: np.ndarray) -> None:
        self.mask, self.size = mask, fft.next_fast_len(len(x) + TAPS)
        self.spectrum = fft.rfft(x, self.size, workers=3)
        self.gram = gram(x, np.flatnonzero(mask))

    def rhs(self, y: np.ndarray) -> np.ndarray:
        return rhs(self.spectrum, self.size, y, self.mask)


class LocalFits:
    """Frozen `fir` spec fitted IN SAMPLE on every 1 s segment of A (rows with their
    real history): how much of the output a linear model adapted to the content of
    one second leaves. A only; the in-sample optimism is about 257 / 48 000."""

    def __init__(self, x: np.ndarray) -> None:
        count = len(x) // SEGMENT
        self.padded = np.concatenate((np.zeros(TAPS - 1), x))
        view = sliding_window_view(self.padded, TAPS)
        self.grams = np.empty((count, TAPS + 1, TAPS + 1))
        for k in range(count):
            block = view[k * SEGMENT : (k + 1) * SEGMENT][:, ::-1]
            design = np.hstack((np.ones((SEGMENT, 1)), block))
            self.grams[k] = design.T @ design
        ms = np.mean(x[: count * SEGMENT].reshape(count, SEGMENT) ** 2, 1)
        self.level = 10 * np.log10(ms)

    def residual_shares(self, y: np.ndarray) -> np.ndarray:
        """(residual power, output power) of every segment after its own fit."""
        out = np.empty((len(self.grams), 2))
        for k, g in enumerate(self.grams):
            ys = y[k * SEGMENT : (k + 1) * SEGMENT]
            xe = self.padded[k * SEGMENT : (k + 1) * SEGMENT + TAPS - 1]
            b = np.concatenate(([ys.sum()], correlate(xe, ys, "valid")[::-1]))
            theta = solve_fir(g, b)
            yy = ys @ ys
            out[k] = yy - 2 * theta @ b + theta @ g @ theta, yy
        return out


def local_fits(ctx: Context, take: int) -> dict:
    """Pooled in-sample residual share of the 1 s local FIRs per matched band."""
    shares = ctx.local.residual_shares(section(take, h.FIT))
    out = {}
    for lvl in LEVELS:
        pick = band(ctx.local.level, lvl)
        out[f"{lvl}"] = {
            "segments": int(pick.sum()),
            "residual_share": float(shares[pick, 0].sum() / shares[pick, 1].sum()),
        }
    return out


# ---------------------------------------------------------------- windows and bands


def window_levels(x: np.ndarray, start: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean square and level (dBFS) of the 50 ms windows of x from `start`."""
    count = (len(x) - start) // h.WINDOW
    ms = np.mean(x[start : start + count * h.WINDOW].reshape(count, h.WINDOW) ** 2, 1)
    return ms, 10 * np.log10(np.maximum(ms, 1e-30))


def trend(ms: np.ndarray) -> np.ndarray:
    """Window level minus the level of the mean power of the TREND_WINDOWS before."""
    before = np.convolve(ms, np.ones(TREND_WINDOWS) / TREND_WINDOWS)[: len(ms)]
    out = np.full(len(ms), np.nan)
    out[TREND_WINDOWS:] = 10 * np.log10(
        ms[TREND_WINDOWS:] / before[TREND_WINDOWS - 1 : -1]
    )
    return out


def band(level: np.ndarray, centre: float, half: float = HALF_BAND) -> np.ndarray:
    return (level >= centre - half) & (level < centre + half)


def schedule(level: np.ndarray) -> np.ndarray:
    """Band index of every window: 0 below -33, 1..3 the matched bands, 4 from -15."""
    edges = [LEVELS[0] - HALF_BAND, *(c + HALF_BAND for c in LEVELS)]
    return np.searchsorted(edges, level, side="right")


def classes(level: np.ndarray, change: np.ndarray, centre: float) -> dict:
    """Window masks of one band: all, and split by envelope trend (x only)."""
    inside = band(level, centre)
    return {
        "all": inside,
        "stationary": inside & (np.abs(change) <= TREND_DB),
        "rising": inside & (change > TREND_DB),
        "falling": inside & (change < -TREND_DB),
    }


def rows_of(windows: np.ndarray, start: int, length: int) -> np.ndarray:
    """Sample mask of the selected 50 ms windows (window i covers start + i * W)."""
    mask = np.zeros(length, bool)
    mask[start : start + len(windows) * h.WINDOW] = np.repeat(windows, h.WINDOW)
    return mask


def pooled(y: np.ndarray, q: np.ndarray, windows: np.ndarray) -> float:
    """Residual power share over the selected B windows (harness grid from SKIP)."""
    count = len(windows)
    frame = slice(h.SKIP, h.SKIP + count * h.WINDOW)
    ys = y[frame].reshape(count, -1)[windows]
    rs = (y - q)[frame].reshape(count, -1)[windows]
    return float(np.sum(rs**2) / np.sum(ys**2)) if windows.any() else float("nan")


def descriptors(x: np.ndarray, windows: np.ndarray, start: int) -> dict:
    """Median window crest factor and input power split by frequency (20 Hz bins)."""
    count = len(windows)
    xs = x[start : start + count * h.WINDOW].reshape(count, -1)[windows]
    crest = 20 * np.log10(np.abs(xs).max(1) / np.sqrt(np.mean(xs**2, 1)))
    hz = fft.rfftfreq(h.WINDOW, 1 / h.FS)
    weight = np.where((hz == 0) | (hz == h.FS / 2), 1.0, 2.0)  # one-sided Parseval
    spectrum = weight * (np.abs(fft.rfft(xs, axis=1)) ** 2).sum(0)
    total = spectrum.sum()
    return {
        "windows": int(windows.sum()),
        "crest_db_median": float(np.median(crest)),
        "share_below_1k": float(spectrum[hz < 1000].sum() / total),
        "share_1k_5k": float(spectrum[(hz >= 1000) & (hz < 5000)].sum() / total),
        "share_above_5k": float(spectrum[hz >= 5000].sum() / total),
    }


# ---------------------------------------------------------------- shared context


class Context:
    """Everything that depends on x only: multisine spectra and Grams, DI windows,
    band Grams on A, pink-noise Gram."""

    def __init__(self) -> None:
        t0 = time.time()
        self.ms_x = {lvl: periods("x", lvl) for lvl in LEVELS}
        self.ms_sections = {
            lvl: [section("x", multisine_name(lvl, r)) for r in REALIZATIONS]
            for lvl in LEVELS
        }
        rows = np.arange(len(self.ms_sections[LEVELS[0]][0])) >= PERIOD
        self.ms_fits = {
            lvl: [RowFit(s, rows) for s in sections]
            for lvl, sections in self.ms_sections.items()
        }
        self.ms_rows = rows
        self.x_a, self.x_b = section("x", h.FIT), section("x", h.READ)
        self.x_v = section("x", h.VALID)
        ms_a, level_a = window_levels(self.x_a, 0)
        ms_b, level_b = window_levels(self.x_b, h.SKIP)
        self.level_b, self.level_v = level_b, window_levels(self.x_v, h.SKIP)[1]
        self.classes_a = {c: classes(level_a, trend(ms_a), c) for c in LEVELS}
        self.classes_b = {c: classes(level_b, trend(ms_b), c) for c in LEVELS}
        sched_a = schedule(level_a)
        self.schedule = {h.READ: schedule(level_b), h.VALID: schedule(self.level_v)}
        self.band_fits = [  # 5 bands of `schedule`, fitted on A rows only
            RowFit(self.x_a, rows_of(sched_a == i, 0, len(self.x_a)))
            for i in range(len(LEVELS) + 2)
        ]
        self.windows_a = np.bincount(sched_a, minlength=len(LEVELS) + 2).tolist()
        self.narrow_b = {c: band(level_b, c, NARROW) for c in LEVELS}
        narrow_a = {c: band(level_a, c, NARROW) for c in LEVELS}
        self.narrow_fits = {
            c: RowFit(self.x_a, rows_of(m, 0, len(self.x_a)))
            for c, m in narrow_a.items()
        }
        self.narrow_windows = {
            c: [int(narrow_a[c].sum()), int(self.narrow_b[c].sum())] for c in LEVELS
        }
        self.local = LocalFits(self.x_a)
        self.pink = section("x", PINK)
        half = len(self.pink) // 2
        pink_rows = (np.arange(len(self.pink)) >= h.SKIP) & (
            np.arange(len(self.pink)) < half
        )
        self.pink_fit, self.pink_half = RowFit(self.pink, pink_rows), half
        print(f"context {time.time() - t0:.0f} s", flush=True)

    def excitation(self) -> dict:
        out = {"multisine": {}, "di_test": {}, "di_train_windows": {}}
        for lvl in LEVELS:
            x = self.ms_sections[lvl][0][PERIOD:]
            out["multisine"][f"{lvl}"] = descriptors(
                x, np.ones(len(x) // h.WINDOW, bool), 0
            )
            out["di_test"][f"{lvl}"] = {
                kind: descriptors(self.x_b, mask, h.SKIP)
                for kind, mask in self.classes_b[lvl].items()
            }
            out["di_train_windows"][f"{lvl}"] = {
                kind: int(mask.sum()) for kind, mask in self.classes_a[lvl].items()
            }
        pink = self.pink[self.pink_half :]
        out["pink_read_half"] = descriptors(
            pink, np.ones(len(pink) // h.WINDOW, bool), 0
        )
        out["di_train_windows_by_schedule"] = self.windows_a
        out["narrow_windows_a_b"] = {f"{c}": n for c, n in self.narrow_windows.items()}
        out["di_test_all"] = descriptors(
            self.x_b, np.ones(len(self.level_b), bool), h.SKIP
        )
        return out


# ---------------------------------------------------------------- per take


def multisine_fir(ctx: Context, take: int, lvl: int) -> tuple[dict, np.ndarray]:
    """Frozen `fir` spec on the multisine at one level: LOO over realizations (fit on
    3, read periods 2-3 of the 4th) and the fit on all 4 (returned for the DI)."""
    ys = [section(take, multisine_name(lvl, r)) for r in REALIZATIONS]
    fits = ctx.ms_fits[lvl]
    b = [f.rhs(y) for f, y in zip(fits, ys, strict=True)]
    residual = total = 0.0
    for k in range(len(REALIZATIONS)):
        theta = solve_fir(
            sum(f.gram for i, f in enumerate(fits) if i != k),
            sum(v for i, v in enumerate(b) if i != k),
        )
        q = render(theta, ctx.ms_sections[lvl][k])
        residual += np.sum((ys[k] - q)[PERIOD:] ** 2)
        total += np.sum(ys[k][PERIOD:] ** 2)
    theta_all = solve_fir(sum(f.gram for f in fits), sum(b))
    return {"D_fir_loo": residual / total, "c_fir_loo": np.sqrt(residual / total)}, (
        theta_all
    )


def on_multisine(ctx: Context, take: int, lvl: int, predict) -> float:
    """Residual power share of a fixed model on periods 2-3 of the 4 realizations."""
    residual = total = 0.0
    for r, x in zip(REALIZATIONS, ctx.ms_sections[lvl], strict=True):
        y = section(take, multisine_name(lvl, r))
        residual += np.sum((y - predict(x))[PERIOD:] ** 2)
        total += np.sum(y[PERIOD:] ** 2)
    return residual / total


def scheduled(
    thetas: list, x: np.ndarray, which: np.ndarray, causal: bool = False
) -> np.ndarray:
    """Level-scheduled FIR: every 50 ms window (grid from SKIP) takes the output of
    the band FIR of its own input level (x only, 50 ms look-ahead: non-causal), or,
    causal, of the level of the window before it."""
    if causal:
        which = np.concatenate((which[:1], which[:-1]))
    renders = np.stack([render(t, x) for t in thetas])
    per_sample = np.full(len(x), which[-1])
    per_sample[: h.SKIP] = which[0]
    per_sample[h.SKIP : h.SKIP + len(which) * h.WINDOW] = np.repeat(which, h.WINDOW)
    return renders[per_sample, np.arange(len(x))]


def share(y: np.ndarray, q: np.ndarray, part: slice) -> float:
    return float(np.sum((y - q)[part] ** 2) / np.sum(y[part] ** 2))


def validation(ctx: Context, take: int, thetas: list) -> dict:
    """The band FIRs (fitted on A) read on V: per matched band and scheduled; V is
    where a per-device threshold would be measured without touching B. The two
    halves of V (15 s each) give the section-to-section spread of that ceiling."""
    y_v = section(take, h.VALID)
    which = ctx.schedule[h.VALID]
    q = scheduled(thetas, ctx.x_v, which)
    q_causal = scheduled(thetas, ctx.x_v, which, causal=True)
    out = {
        "scheduled_fir": h.ratio(y_v, q) ** 2,
        "scheduled_fir_causal": h.ratio(y_v, q_causal) ** 2,
    }
    half = len(y_v) // 2
    parts = (slice(h.SKIP, None), slice(h.SKIP, half), slice(half, None))
    out["scheduled_fir_halves"] = [share(y_v, q, part) for part in parts[1:]]
    if take in h.GAIN_TAKES.values():  # paired E of the frozen probes on V
        out["probes"] = {}
        first = np.arange(len(which)) < len(which) // 2  # windows of the first half
        for name in ("fir", *H2_PROBES):
            q_p = np.asarray(h.frozen_prediction(name, take, read=h.VALID))
            shares = [share(y_v, q_p, part) for part in parts]
            out["probes"][name] = {
                "share": shares[0],
                "halves": shares[1:],
                "band_halves": {
                    f"{lvl}": [
                        pooled(y_v, q_p, (which == i) & side)
                        for side in (first, ~first)
                    ]
                    for i, lvl in enumerate(LEVELS, start=1)
                },
                "paired_E_halves": [
                    1 - s / share(y_v, q, part)
                    for s, part in zip(shares[1:], parts[1:], strict=True)
                ],
                "paired_E_halves_causal": [
                    1 - s / share(y_v, q_causal, part)
                    for s, part in zip(shares[1:], parts[1:], strict=True)
                ],
            }
    for i, lvl in enumerate(LEVELS, start=1):
        out[f"{lvl}"] = pooled(y_v, render(thetas[i], ctx.x_v), which == i)
    ms_v, _ = window_levels(y_v, h.SKIP)
    out["output_power_by_band"] = (
        np.bincount(which, ms_v, minlength=len(thetas)) / ms_v.sum()
    ).tolist()
    return out


def di_bands(ctx: Context, take: int, models: dict) -> dict:
    """Per band and trend class: residual power share on B of the frozen probes, of
    the level-conditional FIR (fitted on A), of the multisine BLA and FIR."""
    y_a, y_b = section(take, h.FIT), section(take, h.READ)
    frozen = {p: np.asarray(h.frozen_prediction(p, take)) for p in PROBES}
    ms_b, _ = window_levels(y_b, h.SKIP)
    out = {"output_power_share_outside_bands": 0.0}
    inside = np.zeros(len(ms_b), bool)
    thetas = [solve_fir(f.gram, f.rhs(y_a)) for f in ctx.band_fits]
    for i, lvl in enumerate(LEVELS, start=1):
        theta = thetas[i]
        models[lvl]["di_band_fir"] = theta
        preds = {f"frozen_{p}": q for p, q in frozen.items()}
        preds["band_fir"] = render(theta, ctx.x_b)
        preds["bla_filter"] = apply_filter(models[lvl]["bla_filter"], ctx.x_b)
        preds["multisine_fir"] = render(models[lvl]["multisine_fir"], ctx.x_b)
        entry = out[f"{lvl}"] = {}
        for kind, mask in ctx.classes_b[lvl].items():
            entry[kind] = {name: pooled(y_b, q, mask) for name, q in preds.items()}
        entry["output_power_share"] = float(
            ms_b[ctx.classes_b[lvl]["all"]].sum() / ms_b.sum()
        )
        fit = ctx.narrow_fits[lvl]
        preds["narrow_fir"] = render(solve_fir(fit.gram, fit.rhs(y_a)), ctx.x_b)
        entry["narrow"] = {  # B windows within +-1 dB; narrow_fir fitted on A +-1 dB
            name: pooled(y_b, preds[name], ctx.narrow_b[lvl])
            for name in ("narrow_fir", "band_fir", "frozen_fir", "multisine_fir")
        }
        inside |= ctx.classes_b[lvl]["all"]
    out["output_power_share_outside_bands"] = float(ms_b[~inside].sum() / ms_b.sum())
    out["global"] = {p: h.ratio(y_b, q) for p, q in frozen.items()}
    q = scheduled(thetas, ctx.x_b, ctx.schedule[h.READ])
    out["global"]["scheduled_fir"] = h.ratio(y_b, q)
    q = scheduled(thetas, ctx.x_b, ctx.schedule[h.READ], causal=True)
    out["global"]["scheduled_fir_causal"] = h.ratio(y_b, q)
    out["validation"] = validation(ctx, take, thetas)
    out["_window_ms"] = ms_b
    out["_fir_prediction"] = frozen["fir"]
    return out


def interpolated(curve: dict, level: np.ndarray) -> np.ndarray:
    """A multisine curve (value per level) at other input levels, linear in dB,
    held outside -30 .. -18 dBFS."""
    return np.interp(level, LEVELS, [curve[lvl] for lvl in LEVELS])


def predictions(ctx: Context, take: int, curves: dict, di: dict) -> dict:
    """Frozen FIR capacity on B, globally and per harness quintile, against the
    multisine curves read at each window's input level."""
    y_b = section(take, h.READ)
    ms_b, q = di.pop("_window_ms"), di.pop("_fir_prediction")
    measured = h.level_capacity_from(ctx.x_b, y_b, q)
    edges = np.quantile(ctx.level_b, np.linspace(0, 1, len(measured) + 1))
    which = np.searchsorted(edges[1:-1], ctx.level_b, side="right")  # as the harness
    out = {"quintiles": [], "global": {"fir_capacity2": h.ratio(y_b, q) ** 2}}
    for i, b in enumerate(measured):
        median = float(np.median(ctx.level_b[which == i]))
        out["quintiles"].append(
            {
                "input_db": b["input_db"],
                "median_input_db": median,
                "fir_capacity2": b["capacity"] ** 2,
                **{
                    f"{name}_at_median": float(interpolated(c, np.array([median]))[0])
                    for name, c in curves.items()
                },
            }
        )
    ms_v, _ = window_levels(section(take, h.VALID), h.SKIP)
    for name, c in curves.items():
        for tag, ms, level in (("B", ms_b, ctx.level_b), ("V", ms_v, ctx.level_v)):
            out["global"][f"{name}_weighted_by_{tag}_output"] = float(
                np.sum(ms * interpolated(c, level)) / ms.sum()
            )
    return out


def pink(ctx: Context, take: int, models: dict) -> dict:
    """Pink noise -30 dBFS: frozen `fir` spec on its first half, read on the second;
    and the -30 dBFS models of the multisine and of the DI band, not refitted."""
    y = section(take, PINK)
    half = ctx.pink_half
    theta = solve_fir(ctx.pink_fit.gram, ctx.pink_fit.rhs(y))
    out = {}
    for name, q in {
        "pink_fir_half": render(theta, ctx.pink),
        "bla_filter": apply_filter(models[-30]["bla_filter"], ctx.pink),
        "multisine_fir": render(models[-30]["multisine_fir"], ctx.pink),
        "di_band_fir": render(models[-30]["di_band_fir"], ctx.pink),
    }.items():
        out[name] = float(np.sum((y - q)[half:] ** 2) / np.sum(y[half:] ** 2))
    out["output_dbfs"] = nr.db(nr.rms(y[half:]))
    return out


def one_take(ctx: Context, take: int) -> dict:
    t0 = time.time()
    row = h.ROWS[take]
    out = {
        "excluded_gate4": take in h.EXCLUDED,
        "setting": row["setting"],
        "G": float(row["G"]),
        "cab": row["cab"],
        "multisine": {},
    }
    models = {}
    for lvl in LEVELS:
        dec = decomposition(ctx.ms_x[lvl], periods(take, lvl))
        g_bla = dec.pop("bla_lines")
        fir, theta = multisine_fir(ctx, take, lvl)
        out["multisine"][f"{lvl}"] = dec | fir
        models[lvl] = {"bla_filter": bla_filter(g_bla), "multisine_fir": theta}
    out["di_test"] = di_bands(ctx, take, models)
    curves = {
        key: {lvl: out["multisine"][f"{lvl}"][key] for lvl in LEVELS}
        for key in ("D", "D_loo", "D_fir_loo")
    }
    out["prediction"] = predictions(ctx, take, curves, out["di_test"])
    out["cross"] = {
        f"{lvl}": {
            "di_band_fir_on_multisine": on_multisine(
                ctx, take, lvl, lambda x, t=models[lvl]["di_band_fir"]: render(t, x)
            )
        }
        for lvl in LEVELS
    }
    out["pink"] = pink(ctx, take, models)
    out["local_A"] = local_fits(ctx, take)
    print(f"take {take}: {time.time() - t0:.0f} s", flush=True)
    return out


# ---------------------------------------------------------------- checks and summary


def instrument_check(result: dict, ctx: Context) -> dict:
    """48 kHz decomposition against the frozen 192 kHz reading (odd/even levels and
    period drift), closure, decimation edge of the source, and the row-masked FIR
    against the library fit on di_validation (all rows)."""
    gaps = {"odd_db": [], "even_db": [], "drift": []}
    for take, levels in h.READING["multisine"].items():
        if int(take) not in result["takes"]:
            continue
        for lvl, frozen in levels.items():
            mine = result["takes"][int(take)]["multisine"][lvl]
            gaps["odd_db"].append(abs(mine["odd_db"] - frozen["odd_db"]))
            gaps["even_db"].append(abs(mine["even_db"] - frozen["even_db"]))
            gaps["drift"].append(abs(mine["drift"] - frozen["period_drift"]))
    closure = [
        abs(m["closure"] - 1)
        for t in result["takes"].values()
        for m in t["multisine"].values()
    ]
    x = ctx.ms_sections[LEVELS[0]][0]
    edge = np.mean((x[PERIOD : 2 * PERIOD] - x[2 * PERIOD :]) ** 2) / 2
    out = {
        "nolly_multisine_compared": len(gaps["odd_db"]),
        **{f"max_abs_{k}": max(v, default=float("nan")) for k, v in gaps.items()},
        "closure_max_abs": max(closure),
        "source_period_edge_share": float(edge / np.mean(x[PERIOD:] ** 2)),
    }
    out |= library_check()
    out["pass"] = bool(
        out["max_abs_odd_db"] <= 0.1
        and out["max_abs_even_db"] <= 0.1
        and out["closure_max_abs"] <= 1e-5
        and out["masked_fir_vs_library_on_V"] <= 1e-6
    )
    return out


def library_check(take: int = 20) -> dict:
    """RowFit with every row of di_validation against fit_probe (frozen `fir`): the
    library also sums the 255 tail rows (y = 0 past the end), hence a small gap."""
    x, y = section("x", h.VALID), section(take, h.VALID)
    mine = render(solve_fir(*_all_rows(x, y)), x)
    lib = h.FrozenProbe("fir").fit({"x": x}, y).predict({"x": x})
    return {
        "masked_fir_vs_library_on_V": float(nr.rms(mine - lib) / nr.rms(lib)),
        "masked_fir_capacity_on_V": h.ratio(y, mine),
        "library_fir_capacity_on_V": h.ratio(y, lib),
    }


def _all_rows(x: np.ndarray, y: np.ndarray):
    fit = RowFit(x, np.ones(len(x), bool))
    return fit.gram, fit.rhs(y)


def correlation(a: list[float], b: list[float]) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    rank = [rankdata(v) for v in (a, b)]  # average ranks for ties
    return {
        "n": len(a),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(np.corrcoef(*rank)[0, 1]),
        "mean_ratio_b_over_a": float(np.mean(b / a)),
        "mean_diff_b_minus_a": float(np.mean(b - a)),
        "max_abs_diff": float(np.max(np.abs(b - a))),
    }


def band_values(v: dict, lvl: str) -> dict:
    """The per-band quantities of one take compared in the summary."""
    di, ms = v["di_test"][lvl], v["multisine"][lvl]
    out = {
        "multisine_D": ms["D"],
        "multisine_D_fir_loo": ms["D_fir_loo"],
        "multisine_D_bla_loo": ms["D_loo"],
        "di_band_fir": di["all"]["band_fir"],
        "di_band_fir_stationary": di["stationary"]["band_fir"],
        "di_band_fir_rising": di["rising"]["band_fir"],
        "di_narrow_fir": di["narrow"]["narrow_fir"],
        "di_band_fir_on_narrow": di["narrow"]["band_fir"],
        "di_band_fir_V": v["di_test"]["validation"][lvl],
        "multisine_fir_on_di": di["all"]["multisine_fir"],
        "di_local_A_1s_in_sample": v["local_A"][lvl]["residual_share"],
        "di_band_fir_on_multisine": v["cross"][lvl]["di_band_fir_on_multisine"],
    }
    out |= {p: di["all"][f"frozen_{p}"] for p in PROBES}
    if lvl == "-30":
        out["pink_fir_half"] = v["pink"]["pink_fir_half"]
    return out


def group_summary(takes: dict) -> dict:
    """Median and range over a group of takes of every band quantity."""
    out = {"takes": sorted(takes), "bands": {}}
    if not takes:
        return out
    for lvl in map(str, LEVELS):
        values = [band_values(v, lvl) for v in takes.values()]
        out["bands"][lvl] = {
            k: {
                "median": float(np.median([x[k] for x in values])),
                "min": float(np.min([x[k] for x in values])),
                "max": float(np.max([x[k] for x in values])),
            }
            for k in values[0]
        }
    glob = {
        "frozen_fir_capacity2": [
            v["di_test"]["global"]["fir"] ** 2 for v in takes.values()
        ],
        "multisine_D_fir_loo_weighted_B": [
            v["prediction"]["global"]["D_fir_loo_weighted_by_B_output"]
            for v in takes.values()
        ],
        "scheduled_fir_B": [
            v["di_test"]["global"]["scheduled_fir"] ** 2 for v in takes.values()
        ],
    }
    out["global"] = {k: float(np.median(x)) for k, x in glob.items()}
    return out


def summary(result: dict) -> dict:
    """Usable takes: group medians (G = 0; G >= 2.5) and, per band, correlations of
    the multisine D (in sample; FIR LOO) with the DI residual shares on B, over all
    usable takes and over G >= 2.5 only (the G = 0 takes form a separate cluster)."""
    usable = {t: v for t, v in result["takes"].items() if not v["excluded_gate4"]}
    loud = {t: v for t, v in usable.items() if v["G"] > 0}
    out = {
        "groups": {
            "G0": group_summary({t: v for t, v in usable.items() if v["G"] == 0}),
            "G_ge_2.5": group_summary(loud),
        },
        "noise_share_max": max(
            m["noise_share"] for v in usable.values() for m in v["multisine"].values()
        ),
        # descriptive: regularized LOO FIR 256 against the variance-corrected BLA
        "fir_loo_minus_bla_D_max_abs": max(
            abs(m["D_fir_loo"] - m["D"])
            for v in usable.values()
            for m in v["multisine"].values()
        ),
        "correlations": {},
        "margin_V_halves": {
            k: float(f(margins))
            for margins in [
                [
                    abs(a - b) / ((a + b) / 2)
                    for a, b in (
                        v["di_test"]["validation"]["scheduled_fir_halves"]
                        for v in usable.values()
                    )
                ]
            ]
            for k, f in (("median", np.median), ("max", np.max))
        },
    }
    pairs = [
        ("multisine_D", "di_band_fir"),
        ("multisine_D", "di_band_fir_stationary"),
        ("multisine_D", "di_narrow_fir"),
        ("multisine_D", "di_local_A_1s_in_sample"),
        ("multisine_D", "fir"),
        ("multisine_D_fir_loo", "di_band_fir"),
        ("di_band_fir_V", "di_band_fir"),
    ]
    for name, group in (("usable", usable), ("G_ge_2.5", loud)):
        if len(group) < 3:  # partial runs (--takes): no correlation to report
            continue
        entry = out["correlations"][name] = {}
        for lvl in map(str, LEVELS):
            values = [band_values(v, lvl) for v in group.values()]
            for a, b in pairs + ([("pink_fir_half", "di_band_fir")] * (lvl == "-30")):
                entry[f"{lvl}|{a}_vs_{b}"] = correlation(
                    [x[a] for x in values], [x[b] for x in values]
                )
        glob = [v["prediction"]["global"] for v in group.values()]
        entry["global|multisine_D_fir_loo_weighted_B_vs_frozen_fir"] = correlation(
            [g["D_fir_loo_weighted_by_B_output"] for g in glob],
            [g["fir_capacity2"] for g in glob],
        )
        entry["global|scheduled_fir_V_vs_B"] = correlation(
            [v["di_test"]["validation"]["scheduled_fir"] for v in group.values()],
            [v["di_test"]["global"]["scheduled_fir"] ** 2 for v in group.values()],
        )
    return out


def h2_band(t: dict, lvl: str) -> dict:
    """Candidate ceilings (power shares) of one band and the probes against them.
    rule_V_beats_linear: the probe's share on B below the ceiling measured on V (a
    threshold set without B), not a comparison on the same samples."""
    ceilings = {
        "multisine_fir_loo": t["multisine"][lvl]["D_fir_loo"],
        "di_band_fir_V": t["di_test"]["validation"][lvl],
        "di_band_fir_B": t["di_test"][lvl]["all"]["band_fir"],
    }
    probes = {p: t["di_test"][lvl]["all"][f"frozen_{p}"] for p in H2_PROBES}
    return {
        "ceiling_D": ceilings,
        "probe_capacity2": probes,
        "explained_share": {
            f"{p}_vs_{c}": 1 - v / d
            for p, v in probes.items()
            for c, d in ceilings.items()
        },
        "rule_V_beats_linear": {
            p: bool(v < ceilings["di_band_fir_V"]) for p, v in probes.items()
        },
    }


def against_fir(t: dict) -> dict:
    """Paired rule against the frozen FIR (the linear reference fitted on A, read on
    the same samples): E = 1 - capacity^2(probe) / capacity^2(FIR) on B, globally
    and per band; margin = |E(V first half) - E(V second half)|; pass if E_B > it."""
    probes = t["di_test"]["validation"]["probes"]
    out = {}
    for p in H2_PROBES:
        entry = out[p] = {}
        b_global = (
            1 - t["di_test"]["global"][p] ** 2 / t["di_test"]["global"]["fir"] ** 2
        )
        v_global = [
            1 - a / b
            for a, b in zip(probes[p]["halves"], probes["fir"]["halves"], strict=True)
        ]
        cells = {"global": (b_global, v_global)}
        for lvl in map(str, LEVELS):
            band = t["di_test"][lvl]["all"]
            cells[lvl] = (
                1 - band[f"frozen_{p}"] / band["frozen_fir"],
                [
                    1 - a / b
                    for a, b in zip(
                        probes[p]["band_halves"][lvl],
                        probes["fir"]["band_halves"][lvl],
                        strict=True,
                    )
                ],
            )
        for cell, (e_b, e_v) in cells.items():
            spread = abs(e_v[0] - e_v[1])
            entry[cell] = {
                "E_B": e_b,
                "E_V_halves": e_v,
                "margin": spread,
                "pass": bool(e_b > spread),
            }
    return out


def h2_rule(result: dict) -> dict:
    """Per H2 take, per band and globally: candidate ceilings and the frozen probes'
    explained share of them, E = 1 - capacity^2 / ceiling^2 (> 0: below the ceiling).
    Global ceilings: the multisine D read at each window level and weighted by the
    output power of V; the level-scheduled FIR on V (both without B)."""
    out = {}
    for gain, take in h.GAIN_TAKES.items():
        t = result["takes"][take]
        ms_weighted = t["prediction"]["global"]["D_fir_loo_weighted_by_V_output"]
        ceilings = {
            "multisine_fir_loo_weighted_V": ms_weighted,
            "scheduled_fir_V": t["di_test"]["validation"]["scheduled_fir"],
            "scheduled_fir_B": t["di_test"]["global"]["scheduled_fir"] ** 2,
        }
        probes = {p: t["di_test"]["global"][p] ** 2 for p in (*H2_PROBES, "fir")}
        first, second = t["di_test"]["validation"]["scheduled_fir_halves"]
        margin = abs(first - second) / ((first + second) / 2)
        paired = {}
        for p, v in t["di_test"]["validation"]["probes"].items():
            paired[p] = {}
            for tag in ("", "_causal"):
                e_halves = v[f"paired_E_halves{tag}"]
                spread = abs(e_halves[0] - e_halves[1])
                ref = t["di_test"]["global"][f"scheduled_fir{tag}"] ** 2
                e_b = 1 - t["di_test"]["global"][p] ** 2 / ref
                paired[p] |= {
                    f"E_B{tag}": e_b,
                    f"E_V_halves{tag}": e_halves,
                    f"E_V_spread{tag}": spread,
                    f"beats_linear_by_spread{tag}": bool(e_b > spread),
                }
        out[f"{gain:g}"] = {
            "take": take,
            "bands": {lvl: h2_band(t, lvl) for lvl in map(str, LEVELS)},
            "global": {
                "ceiling_D": ceilings,
                "probe_capacity2": probes,
                "explained_share": {
                    f"{p}_vs_{c}": 1 - v / d
                    for p, v in probes.items()
                    for c, d in ceilings.items()
                },
                "frozen_rule_refused": {p: bool(v > 0.3**2) for p, v in probes.items()},
                # the frozen 0.3 read in this frame: the share of the beyond-linear
                # power (level-scheduled FIR on V) a probe must explain to pass it
                "frozen_0.3_required_explained_share": 1
                - 0.3**2 / ceilings["scheduled_fir_V"],
                "rule_V_beats_linear": {
                    p: bool(v < ceilings["scheduled_fir_V"]) for p, v in probes.items()
                },
                # margin: relative gap of the ceiling between the two halves of V
                "margin_V_halves": margin,
                "rule_V_beats_linear_by_margin": {
                    p: bool(1 - v / ceilings["scheduled_fir_V"] > margin)
                    for p, v in probes.items()
                },
                # paired rule: probe and level-scheduled FIR (A fit) on the same B
                # samples; margin = spread of that paired E between the V halves
                "paired": paired,
            },
            "against_frozen_fir": against_fir(t),
            "pink_-30": t["pink"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--takes", type=int, nargs="*", default=sorted(h.ROWS))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="recompute summary and h2_rule from the existing --out JSON",
    )
    args = parser.parse_args()
    if args.summary_only:
        result = json.loads(args.out.read_text())
        result["takes"] = {int(t): v for t, v in result["takes"].items()}
        result["summary"] = summary(result)
        if all(t in result["takes"] for t in h.GAIN_TAKES.values()):
            result["h2_rule"] = h2_rule(result)
        args.out.write_text(json.dumps(nr.finite(to_plain(result)), indent=1) + "\n")
        return
    ctx = Context()
    result = {
        "protocol": {
            "rate_hz": h.FS,
            "levels_dbfs": list(LEVELS),
            "band_half_width_db": HALF_BAND,
            "window_samples": h.WINDOW,
            "skip_b": h.SKIP,
            "fir_spec": {"taps": TAPS, "ridge_relative": RIDGE, "constant": True},
            "bla_acausal_lags": ACAUSAL,
            "trend": {"db": TREND_DB, "previous_windows": TREND_WINDOWS},
            "fit_sections": "A = di_train (DI models); multisine realizations; "
            "first half of the pink noise. B = di_test is read only.",
        },
        "excitation": ctx.excitation(),
        "takes": {t: one_take(ctx, t) for t in args.takes},
    }
    result["instrument_check"] = instrument_check(result, ctx)
    result["summary"] = summary(result)
    if all(t in result["takes"] for t in h.GAIN_TAKES.values()):
        result["h2_rule"] = h2_rule(result)
    args.out.write_text(json.dumps(nr.finite(to_plain(result)), indent=1) + "\n")
    print(json.dumps(result["instrument_check"], indent=1))


def to_plain(value):
    """numpy scalars to floats, int keys to strings."""
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
