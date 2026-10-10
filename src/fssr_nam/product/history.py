"""History-conditioned gain (product lane): how much the output level depends on the
input level of the recent past, at a fixed present input level. Works on any program
material (DI guitar), unlike E2 (dynamics.py), which needs level steps.

Per frame of FRAME_S (hop HOP_S): present level a = input power (dB) over the frame,
output level o = output power (dB) over the frame, history level h_T = input power
(dB) over the T seconds that end GAP_S before the frame. Frames more than FLOOR_DB
below the loudest input frame are dropped, and BIN_DB bins of a with fewer than
MIN_PER_BIN frames. Within each bin, o, a, (a - bin mean)^2 and h_T are centred; the
history slope gamma_T is the coefficient of h_T in the pooled least-squares fit of o
on those three (dB of output per dB of history at a fixed present level), and
`explained` the share of the residual variance left by a and a^2 that h_T removes.
Holding a in the fit matters: centring per bin alone leaves o and h - a both tied to
the within-bin spread of a (the identity gave gamma = -0.5 to -0.9 on noise; Codex
review, demo/greybox_deep/HISTORY_MEASURE.md). Too few frames: NaN.

A gain that recovers slowly after loud passages (sag, slow compression) gives
gamma < 0. A memoryless system gives gamma = 0 only up to waveform effects (the same
level carried by an attack or by a decay drives a saturation differently), and linear
tails longer than GAP_S (a cabinet) move gamma: read it between a reference and an
evaluated output on the same input (`history_deviation`), beside a memoryless check.
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
    keep = a >= a.max() - FLOOR_DB if len(a) else np.zeros(0, bool)
    out = {"a": a[keep], "o": _power_db(y, starts, width)[keep]}
    for t in HISTORY_S:
        span = round(t * fs)
        out[t] = _power_db(x, starts - gap - span, span)[keep]
    return out


def _centred(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for label in np.unique(bins):
        m = bins == label
        out[m] -= values[m].mean()
    return out


def history_slopes(x: np.ndarray, y: np.ndarray, fs: float) -> dict:
    """gamma_T and the share of variance it explains, per history length."""
    f = frames(x, y, fs)
    bins = np.floor(f["a"] / BIN_DB).astype(int)
    labels, counts = np.unique(bins, return_counts=True)
    kept = np.isin(bins, labels[counts >= MIN_PER_BIN])
    out = {"frames": int(kept.sum())}
    b = bins[kept]
    a = _centred(f["a"][kept], b)
    held = np.column_stack([a, _centred(a**2, b)]) if len(b) else np.zeros((0, 2))
    o = _centred(f["o"][kept], b)
    for t in HISTORY_S:
        h = _centred(f[t][kept], b)
        design = np.column_stack([held, h])
        if len(b) < 10 or np.linalg.matrix_rank(design) < 3:
            out[t] = {"gamma": float("nan"), "explained": float("nan")}
            continue
        coefs, *_ = np.linalg.lstsq(design, o, rcond=None)
        base, *_ = np.linalg.lstsq(held, o, rcond=None)
        full = o - design @ coefs
        reduced = o - held @ base
        # Nothing left once a is held (o a function of a): no share to explain.
        left = float(reduced @ reduced)
        share = 1 - float(full @ full) / left if left > 1e-12 else float("nan")
        out[t] = {"gamma": float(coefs[2]), "explained": share}
    return out


def history_deviation(x, y_ref, y_eval, fs: float) -> dict:
    """Difference of history slopes, evaluated minus reference, per history length."""
    ref, ev = history_slopes(x, y_ref, fs), history_slopes(x, y_eval, fs)
    return {
        "frames": {"ref": ref["frames"], "eval": ev["frames"]},
        **{
            t: {
                "gamma_ref": ref[t]["gamma"],
                "gamma_eval": ev[t]["gamma"],
                "delta": ev[t]["gamma"] - ref[t]["gamma"],
            }
            for t in HISTORY_S
        },
    }
