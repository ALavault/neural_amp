"""History-conditioned gain (product lane): how much the output level depends on the
input level of the recent past, at a fixed present input level. Works on any program
material (DI guitar), unlike E2 (dynamics.py), which needs level steps.

Per frame of FRAME_S (hop HOP_S): present level a = input power (dB) over the frame,
output level o = output power (dB) over the frame, history level h_T = input power
(dB) over the T seconds that end GAP_S before the frame. Frames more than FLOOR_DB
below the loudest input frame are dropped. The static part of o is its mean per
BIN_DB bin of a; the history slope gamma_T is the pooled within-bin regression slope
of o on d = h_T - a (dB of output per dB of past excess). A system without memory
longer than GAP_S gives gamma near 0 whatever its static curve; a gain that recovers
slowly after loud passages (sag, slow compression) gives gamma < 0. Spectral effects
(a filter before a saturation) and linear tails longer than GAP_S (a cabinet) also
move gamma: read it against a memoryless reference on the same input, or between a
reference and an evaluated output (`history_deviation`).
"""

from __future__ import annotations

import numpy as np

FRAME_S = 0.010
HOP_S = 0.005
GAP_S = 0.010
HISTORY_S = (0.030, 0.100, 0.300)
BIN_DB = 2.0
FLOOR_DB = 40.0
MIN_PER_BIN = 20


def _power_db(signal: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    """Mean power (dB) of signal[s : s + width] per start, from a cumulative sum."""
    cumulative = np.concatenate(([0.0], np.cumsum(signal.astype(np.float64) ** 2)))
    power = (cumulative[starts + width] - cumulative[starts]) / width
    return 10 * np.log10(np.maximum(power, 1e-30))


def frames(x: np.ndarray, y: np.ndarray, fs: float) -> dict:
    """Present level a, output level o and history levels h_T per kept frame."""
    width, hop, gap = (round(v * fs) for v in (FRAME_S, HOP_S, GAP_S))
    longest = round(max(HISTORY_S) * fs)
    starts = np.arange(gap + longest, len(x) - width, hop)
    a = _power_db(x, starts, width)
    keep = a >= a.max() - FLOOR_DB
    out = {"a": a[keep], "o": _power_db(y, starts, width)[keep]}
    for t in HISTORY_S:
        span = round(t * fs)
        out[t] = _power_db(x, starts - gap - span, span)[keep]
    return out


def history_slopes(x: np.ndarray, y: np.ndarray, fs: float) -> dict:
    """gamma_T per history length, with the share of the within-bin variance of o
    that the slope explains."""
    f = frames(x, y, fs)
    bins = np.floor(f["a"] / BIN_DB).astype(int)
    labels, counts = np.unique(bins, return_counts=True)
    kept = np.isin(bins, labels[counts >= MIN_PER_BIN])
    out = {"frames": int(kept.sum())}
    for t in HISTORY_S:
        o, d, b = f["o"][kept], (f[t] - f["a"])[kept], bins[kept]
        oc, dc = o.copy(), d.copy()
        for label in np.unique(b):
            m = b == label
            oc[m] -= o[m].mean()
            dc[m] -= d[m].mean()
        gamma = float(oc @ dc / max(dc @ dc, 1e-30))
        residual = oc - gamma * dc
        out[t] = {
            "gamma": gamma,
            "explained": 1 - float(residual @ residual) / max(float(oc @ oc), 1e-30),
        }
    return out


def history_deviation(x, y_ref, y_eval, fs: float) -> dict:
    """Difference of history slopes, evaluated minus reference, per history length."""
    ref, ev = history_slopes(x, y_ref, fs), history_slopes(x, y_eval, fs)
    return {
        t: {
            "gamma_ref": ref[t]["gamma"],
            "gamma_eval": ev[t]["gamma"],
            "delta": ev[t]["gamma"] - ref[t]["gamma"],
        }
        for t in HISTORY_S
    }
