"""Envelope-dynamics deviation on level steps (ad hoc score E2, product lane).

For each step of the input level, the short-time gain trajectory
20 log10(RMS_w(y) / RMS_w(x)) is followed from the step on; trajectories of the same
step type (loud -> quiet: release; quiet -> loud: attack) are AVERAGED over steps, and
each average is referred to its settled value, so that static changes (gain,
saturation curve) cancel and only the transient remains. The score is the RMS, in dB,
of the difference between the evaluated and the reference transients over a scoring
window disjoint from the settling window. Pre-registered in
diagnosis/probe_metrics/NOTE.md (E2).
"""

from __future__ import annotations

import numpy as np

WINDOW_S = 0.005
SCORE_S = (0.010, 0.800)
SETTLED_S = (1.000, 1.200)


def gain_track(x: np.ndarray, y: np.ndarray, start: int, fs: float) -> np.ndarray:
    """Short-time gain (dB) in WINDOW_S windows from `start` to the settling end."""
    width = round(WINDOW_S * fs)
    count = int(SETTLED_S[1] / WINDOW_S)
    segment = slice(start, start + width * count)
    xs = x[segment].reshape(count, width)
    ys = y[segment].reshape(count, width)
    power = np.sqrt(np.mean(ys**2, axis=1) / np.maximum(np.mean(xs**2, axis=1), 1e-30))
    return 20 * np.log10(np.maximum(power, 1e-15))


def transients(x: np.ndarray, y: np.ndarray, starts, fs: float) -> np.ndarray:
    """Step-averaged gain trajectory minus its settled value (dB)."""
    mean = np.mean([gain_track(x, y, s, fs) for s in starts], axis=0)
    times = (np.arange(len(mean)) + 0.5) * WINDOW_S
    settled = (times >= SETTLED_S[0]) & (times < SETTLED_S[1])
    return mean - mean[settled].mean()


def step_kinds(x: np.ndarray, boundaries, fs: float) -> dict:
    """Split step boundaries into release (level drops) and attack (level rises), by
    the input RMS over the second before and after each boundary."""
    span = round(1.0 * fs)
    kinds = {"release": [], "attack": []}
    for b in boundaries:
        before = np.mean(x[max(0, b - span) : b] ** 2)
        after = np.mean(x[b : b + span] ** 2)
        kinds["release" if after < before else "attack"].append(int(b))
    return kinds


def dynamic_deviation(x, y_ref, y_eval, fs: float, boundaries) -> dict:
    """Release and attack scores (dB RMS) of y_eval against y_ref, with the averaged
    transients for inspection."""
    times = (np.arange(int(SETTLED_S[1] / WINDOW_S)) + 0.5) * WINDOW_S
    scored = (times >= SCORE_S[0]) & (times < SCORE_S[1])
    out = {"times_s": times.tolist()}
    for kind, starts in step_kinds(x, boundaries, fs).items():
        ref = transients(x, y_ref, starts, fs)
        ev = transients(x, y_eval, starts, fs)
        out[kind] = {
            "score_db": float(np.sqrt(np.mean((ev - ref)[scored] ** 2))),
            "steps": len(starts),
            "reference_db": ref.tolist(),
            "evaluated_db": ev.tolist(),
        }
    return out
