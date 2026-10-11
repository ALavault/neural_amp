#!/usr/bin/env python3
"""Lead 5 of the probe-capacity studies: state and feedback (exploration, CPU only).

Question: does the output of the Nolly renders depend on input history beyond the
frozen probes' memory (256 taps = 5.3 ms at 48 kHz), and does such a state explain
the H2 capacity gap (Hammerstein 5 leaves 0.48 at gain 0, about 0.80 at gains 2.5-7.5,
0.75 at 10; baseline.json)?

Stages (each one replaces its key of lead5_state_feedback.json, the others are kept):
  clicks     model-free, renders at 192 kHz: the two clicks of each marker section
             (0.5 s apart; start section at 3 s, end section at 444 s), all 42 takes:
             same input, different histories.
  gaps       model-free, 192 kHz: output in the 0.5 s digital-silence gaps between
             sections and in the final 6 s pad, against the preceding input level.
  multisine  model-free, cache: first period against the third (relaxation after
             the 0.5 s gap that precedes every multisine) per 10 ms frame; 2 vs 3.
  steps      model-free, cache: output level after the 20 dB level steps (stationary
             Gaussian coloured noise, 1.2 s segments) against the segment's end.
  history    (a) residual of the frozen H5 on B against the input level over the
             10-1000 ms before the current 50 ms, at matched current level.
  lpv        (b) Hammerstein 5 with branches modulated by a slow input level:
             features x^k z^j (k 1..5, j 0..2, 256 taps each), z = standardized log
             level of x over tau. Causal; controls: anticausal level (non-causal, a
             control, not a probe) and placebo (causal level delayed by 10 s); oracle:
             level of the true past output (reads y on B, nonlinear in y).
  narx       (c) Hammerstein 5 plus past outputs (ARX, polynomial NARX), equation-
             error fit on A; one-step on B is an oracle; free run on V and B. Both are
             nonlinear in y (the regressors contain the target).
Synthetic calibration devices (lead5_state_feedback_fit.device): S0 without slow state,
S1 with an input-driven sag of known time constant 0.5 s, on the same input, through
multisine, steps, history and lpv.

Rules: B (di_test) never enters a fit; tau is selected on V (di_validation), then B is
reported; oracles are labelled; the gate-4 excluded takes (every presence-10 take is
among them) are flagged. Inputs of the level features may read x before B (V is
contiguous before B, A before V): input only, no leakage.

    uv run python diagnosis/probe_capacity/lead5_state_feedback.py \
        [--stages clicks gaps multisine steps history lpv narx]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import correlate, welch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as h  # noqa: E402
import lead5_state_feedback_fit as lf  # noqa: E402

from fssr_nam.product import nolly  # noqa: E402

OUT = HERE / "lead5_state_feedback.json"
nr = h.nr
STAGES = ("clicks", "gaps", "multisine", "steps", "history", "lpv", "narx", "summary")
GAINS = h.GAIN_TAKES  # {0: 30, 2.5: 16, 5: 20, 7.5: 6, 10: 7}
HIGH = [GAINS[g] for g in (2.5, 5.0, 7.5, 10.0)]
PRESENCE10 = tuple(o for o, r in sorted(h.ROWS.items()) if float(r["P"]) == 10)
PRESENCE10_HIGH = tuple(o for o in PRESENCE10 if float(h.ROWS[o]["G"]) >= 5)
STUDY = [*GAINS.values(), *(o for o in PRESENCE10 if h.ROWS[o]["cab"] == "off")]
SYNTH = {"S0": None, "S1": 0.5}  # device(x, tau_s): no slow state / 0.5 s sag
TAUS_MS = (10, 30, 100, 300, 1000, 3000)
ORACLE_TAUS_MS = (100, 1000)
PLACEBO_DELAY = 10 * h.FS
POWERS = 3  # z**0, z**1, z**2
BANDS_MS = ((0, 10), (10, 30), (30, 100), (100, 300), (300, 1000))
HORIZONS_MS = (10, 30, 100, 300, 1000)
FRAME = 480  # 10 ms at 48 kHz
CURRENT = 2400  # current level: the 50 ms ending at the window's end
# Pre-registered reading of (b), fixed before B was read: drops are H5 minus LPV on B
# at the causal tau selected on V (mean V capacity over the gains 2.5-10).
BASIS_FLOOR = 0.04  # largest gain of degree/basis changes in the scratch study
PREREGISTERED = {
    "state_explains_part": "min drop over G 2.5-10 > 0.04, placebo drop < 0.01 at "
    "every G, mean causal drop exceeds mean anticausal drop by > 0.01",
    "level_normalization_not_history": "min drop > 0.04 but the anticausal drop is "
    "within 0.01 of the causal drop (or larger)",
    "refuted_for_H2": "max drop over G 2.5-10 <= 0.04",
    "inconclusive": "anything else",
}


def db_power(p: float) -> float:
    return float(10 * np.log10(max(p, 1e-300)))


def log(*args) -> None:
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def take_info(o: int) -> dict:
    r = h.ROWS[o]
    return {"setting": r["setting"], "excluded_gate4": o in h.EXCLUDED}


# ---------------------------------------------------------------- clicks (192 kHz)

CLICKS = (96_000, 192_384)  # onsets of the two 2 ms clicks in a marker section
CLICK_LEN = 48_000  # 0.25 s from the onset (the response decays below -90 dBFS)
ONSET = 960  # first 5 ms
CLICK_FRAME = 1920  # 10 ms at 192 kHz


def fractional_lag(a: np.ndarray, ref: np.ndarray, search: int = 16) -> float:
    """Lag of a against ref (samples, parabolic peak of the cross-correlation)."""
    c = correlate(a, ref, mode="full", method="fft")
    mid = len(ref) - 1
    window = c[mid - search : mid + search + 1]
    k = int(np.argmax(window))
    if 0 < k < len(window) - 1:
        y0, y1, y2 = window[k - 1 : k + 2]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return float(k - search)


def shifted(v: np.ndarray, d: float) -> np.ndarray:
    """v advanced by d samples (fractional, FFT phase on a zero-padded copy)."""
    n = 2 * len(v)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(v, n) * np.exp(2j * np.pi * f * d), n)[: len(v)]


def compare(a: np.ndarray, ref: np.ndarray) -> dict:
    """Difference of two click responses relative to ref (dB): whole 0.25 s, onset
    (first 5 ms), tail (5-250 ms), and after removing a fractional lag and a gain."""
    lag = fractional_lag(a, ref)
    b = shifted(a, lag)
    g = float(np.dot(b, ref) / np.dot(b, b))
    rel = lambda u, v: nr.db(nr.rms(u - v) / nr.rms(v))  # noqa: E731
    return {
        "total_db": rel(a, ref),
        "onset_db": rel(a[:ONSET], ref[:ONSET]),
        "tail_db": rel(a[ONSET:], ref[ONSET:]),
        "lag_samples_192k": lag,
        "gain": g,
        "after_lag_gain_db": rel(g * b, ref),
        "tail_after_lag_gain_db": rel(g * b[ONSET:], ref[ONSET:]),
    }


def clicks() -> dict:
    s0, s1, quiet = (
        nolly.SECTIONS[n] for n in ("markers_start", "markers_end", "silence")
    )
    out = {}
    for t in nolly.takes():
        o, path = t["order"], nolly.RENDERS / t["file"]
        w = {}
        for tag, s in (("s", s0), ("e", s1)):
            seg = nolly.read(path, s["start"], s["start"] + s["samples"])
            for i, c in enumerate(CLICKS, 1):
                w[f"{tag}{i}"] = seg[c : c + CLICK_LEN]
            if tag == "e":
                pre = seg[CLICKS[0] // 2 : CLICKS[0]]  # 0.75-1 s after steps_-18
            else:  # what is left of click 1 in the 100 ms before click 2
                residue = seg[CLICKS[1] - nolly.RATE // 10 : CLICKS[1]]
        silence = nolly.read(path, quiet["start"], quiet["start"] + quiet["samples"])
        f, p = welch(pre, fs=nolly.RATE, nperseg=8192)
        entry = take_info(o) | {
            "end_preclick_dbfs": nr.db(nr.rms(pre)),
            "start_before_click2_dbfs": nr.db(nr.rms(residue)),
            "start_click1_response_dbfs": nr.db(nr.rms(w["s1"])),
            "start_click1_onset_dbfs": nr.db(nr.rms(w["s1"][:ONSET])),
            "end_preclick_peak_hz": float(f[np.argmax(p)]),
            "silence_dbfs": nr.db(nr.rms(silence)),
        }
        for a, b in (("s2", "s1"), ("e2", "e1"), ("e1", "s1"), ("e2", "s2")):
            entry[f"{a}_vs_{b}"] = compare(w[a], w[b])
        out[str(o)] = entry
        log("clicks", o)
    summary = {}
    groups = {
        "presence10_gain_ge5": [o for o in PRESENCE10_HIGH],
        "others": [o for o in sorted(h.ROWS) if o not in PRESENCE10_HIGH],
    }
    for name, members in groups.items():
        summary[name] = {"takes": members}
        for key in ("s2_vs_s1", "e2_vs_e1", "e1_vs_s1", "e2_vs_s2"):
            for field in (
                "total_db",
                "onset_db",
                "tail_db",
                "after_lag_gain_db",
                "lag_samples_192k",
            ):
                v = [out[str(o)][key][field] for o in members]
                summary[name][f"{key}.{field}"] = {
                    "min": min(v),
                    "median": float(np.median(v)),
                    "max": max(v),
                }
        for field in (
            "end_preclick_dbfs",
            "silence_dbfs",
            "start_before_click2_dbfs",
            "start_click1_onset_dbfs",
        ):
            v = [out[str(o)][field] for o in members]
            summary[name][field] = {
                "min": min(v),
                "median": float(np.median(v)),
                "max": max(v),
            }
    v = {f: [out[str(o)][f] for o in sorted(h.ROWS)] for f in ("end_preclick_dbfs",)}
    e1s1 = [out[str(o)]["e1_vs_s1"]["total_db"] for o in sorted(h.ROWS)]
    summary["spearman_e1_vs_s1_against_end_preclick_all_takes"] = float(
        spearmanr(e1s1, v["end_preclick_dbfs"]).statistic
    )
    return {
        "note": "192 kHz renders; click windows 0.25 s from the onset; relative "
        "differences in dB re the second named response; history: s1 after 3.5 s of "
        "silence from the file start, s2 and e2 0.5 s after a click, e1 after steps_"
        "-18dBFS_rms and 1 s of silence; same input samples in all four",
        "takes": out,
        "summary": summary,
    }


# ---------------------------------------------------------------- gaps (192 kHz)


def gaps() -> dict:
    names = list(nolly.SECTIONS)
    spans = []
    for a, b in itertools.pairwise(names):
        end = nolly.SECTIONS[a]["start"] + nolly.SECTIONS[a]["samples"]
        if nolly.SECTIONS[b]["start"] - end > 0:
            spans.append((a, end, nolly.SECTIONS[b]["start"]))
    level = {}
    for a, end, _ in spans:
        src = nolly.read(nolly.SOURCE, end - nolly.RATE, end)
        level[a] = nr.db(nr.rms(src))
    pad = nolly.SECTIONS["pad_end"]
    steps18 = nolly.SECTIONS["steps_-18dBFS_rms"]
    after_steps = steps18["start"] + steps18["samples"]
    before_click = nolly.SECTIONS["markers_end"]["start"] + CLICKS[0]
    out = {}
    for t in nolly.takes():
        o, path = t["order"], nolly.RENDERS / t["file"]
        rows = []
        for a, end, stop in spans:
            y = nolly.read(path, end, stop)
            half = len(y) // 2
            rows.append(
                {
                    "after": a,
                    "input_last_1s_dbfs": level[a],
                    "first_half_dbfs": nr.db(nr.rms(y[:half])),
                    "second_half_dbfs": nr.db(nr.rms(y[half:])),
                }
            )
        tail = nolly.read(path, pad["start"], pad["start"] + pad["samples"])
        second = [nr.db(nr.rms(c)) for c in tail.reshape(-1, nolly.RATE)]
        # output from the end of steps_-18dBFS_rms to the first end click (1 s)
        y = nolly.read(path, after_steps, before_click)
        frames = y[: len(y) // CLICK_FRAME * CLICK_FRAME].reshape(-1, CLICK_FRAME)
        trace = 10 * np.log10(np.maximum(np.mean(frames**2, axis=1), 1e-300))
        low = int(np.argmin(trace))
        ok = [r for r in rows if r["input_last_1s_dbfs"] > -250]
        lv = np.array([r["input_last_1s_dbfs"] for r in ok])
        fl = np.array([r["second_half_dbfs"] for r in ok])
        out[str(o)] = take_info(o) | {
            "gaps": rows,
            "pad_end_per_second_dbfs": second,
            "after_steps_dbfs_every_50ms": [float(v) for v in trace[::5]],
            "after_steps_min_dbfs": float(trace[low]),
            "after_steps_min_at_s": low * CLICK_FRAME / nolly.RATE,
            "after_steps_regrowth_db": float(trace[low:].max() - trace[low]),
            "after_steps_last_100ms_dbfs": nr.db(nr.rms(y[-nolly.RATE // 10 :])),
            "floor_max_dbfs": float(fl.max()),
            "floor_median_dbfs": float(np.median(fl)),
            "spearman_floor_vs_input_level": float(spearmanr(lv, fl).statistic),
        }
        log("gaps", o)
    return {
        "note": "0.5 s digital-silence gaps of the source between sections; output "
        "RMS in the first and second 0.25 s; input level over the last 1 s of the "
        "preceding section; pad_end: 6 s of silence after markers_end, per second; "
        "after_steps: 10 ms frames from the end of steps_-18dBFS_rms to the first "
        "click of markers_end (1 s of digital silence), every 5th frame, its minimum "
        "and the regrowth after it (dB, > 0: the output grows again in silence)",
        "takes": out,
    }


# ---------------------------------------------------------------- synthetic signals


def synthetic(name: str, section: str, warm: int = h.FS // 2) -> np.ndarray:
    """Output of a synthetic device on a cached section, after `warm` samples of
    silence (the 0.5 s gap that precedes the section in the layout)."""
    x = np.asarray(h.load("x", (section,))[section])
    y = lf.device(np.concatenate((np.zeros(warm), x)), SYNTH[name])
    return y[warm:]


def synthetic_contiguous(name: str) -> dict[str, np.ndarray]:
    """A synthetic device run once over A, V, B (contiguous in the layout)."""
    x = load_avb("x")
    y = lf.device(np.concatenate((np.zeros(h.FS // 2), np.concatenate(x))), SYNTH[name])
    y = y[h.FS // 2 :]
    a, v = len(x[0]), len(x[1])
    return {h.FIT: y[:a], h.VALID: y[a : a + v], h.READ: y[a + v :]}


def load_avb(take) -> list[np.ndarray]:
    d = h.load(take, (h.FIT, h.VALID, h.READ))
    return [np.asarray(d[s]) for s in (h.FIT, h.VALID, h.READ)]


def outputs(tags) -> dict:
    """{tag: {section: y}} for takes (int) and synthetic devices (str)."""
    out = {}
    for t in tags:
        if isinstance(t, str):
            out[t] = synthetic_contiguous(t)
        else:
            out[t] = dict(zip((h.FIT, h.VALID, h.READ), load_avb(t), strict=True))
    return out


# ---------------------------------------------------------------- multisine (cache)


def bands(per_frame: np.ndarray, limit_frames: int) -> dict:
    out = {}
    for lo, hi in BANDS_MS:
        a, b = lo * h.FS // 1000 // FRAME, min(hi * h.FS // 1000 // FRAME, limit_frames)
        if b > a:
            label = f"{lo}-{b * FRAME * 1000 // h.FS}ms"  # clipped at limit_frames
            out[label] = db_power(float(np.mean(per_frame[a:b])))
    return out


def multisine() -> dict:
    period = h.FS
    tags = ["x", *sorted(h.ROWS), *SYNTH]
    out = {}
    for t in tags:
        levels = {}
        for name in nr.MULTISINES:
            y = (
                synthetic(t, name)
                if t in SYNTH
                else np.asarray(h.load(t, (name,))[name])
            )
            p = y[: 3 * period].reshape(3, period)
            ref = np.mean(p[2] ** 2)
            frames = ((p[0] - p[2]) ** 2).reshape(-1, FRAME).mean(axis=1) / ref
            entry = levels.setdefault(
                f"{nr.SECTIONS[name]['rms_dbfs']:g}", {"frames": [], "p2p3": []}
            )
            entry["frames"].append(frames)
            entry["p2p3"].append(np.mean((p[1] - p[2]) ** 2) / ref)
        out[str(t)] = (take_info(t) if isinstance(t, int) else {}) | {
            level: {
                "p1_vs_p3_db": bands(np.mean(v["frames"], axis=0), period // FRAME),
                "p2_vs_p3_db": db_power(float(np.mean(v["p2p3"]))),
            }
            for level, v in levels.items()
        }
        log("multisine", t)
    return {
        "note": "48 kHz cache; per level, mean over the 4 realizations of the power "
        "of (period 1 - period 3) per 10 ms frame re the mean power of period 3, "
        "averaged over the frames of each band (dB); every multisine follows 0.5 s of "
        "digital silence; synthetic devices start from 0.5 s of silence",
        "tags": out,
    }


# ---------------------------------------------------------------- steps (cache)


def steps() -> dict:
    tags = ["x", *sorted(h.ROWS), *SYNTH]
    out = {}
    for t in tags:
        result = {}
        for name in nr.STEPS:
            s = nr.SECTIONS[name]
            y = (
                synthetic(t, name)
                if t in SYNTH
                else np.asarray(h.load(t, (name,))[name])
            )
            edges = [0, *(o // 4 for o in s["onsets"]), len(y)]
            levels = s["segment_rms_dbfs"]
            kinds = {"release": [], "attack": []}
            for i in range(1, len(edges) - 1):
                seg = y[edges[i] : edges[i + 1]]
                frames = (seg[: len(seg) // FRAME * FRAME] ** 2).reshape(-1, FRAME)
                power = frames.mean(axis=1)
                ref = power[60:120].mean()  # 0.6-1.2 s into the segment
                kind = "release" if levels[i] < levels[i - 1] else "attack"
                kinds[kind].append(power / ref)
            result[name] = {
                k: bands(np.mean(v, axis=0), 60) | {"segments": len(v)}
                for k, v in kinds.items()
            }
        out[str(t)] = (take_info(t) if isinstance(t, int) else {}) | result
        log("steps", t)
    return {
        "note": "48 kHz cache; power per 10 ms frame after each level step re the "
        "segment's power at 0.6-1.2 s, mean over the 21-22 steps of each kind (dB, "
        "bands up to 600 ms); release: loud to quiet (-20 dB +-2), attack: quiet to "
        "loud; 'x' is the input itself (control)",
        "tags": out,
    }


# ---------------------------------------------------------------- history (a)


def h5_fit_predict(y: dict) -> np.ndarray:
    """Hammerstein 5 (MultiFit, identical to the frozen probe) fitted on A, read B."""
    xa = np.asarray(h.load("x", (h.FIT,))[h.FIT])
    xb = np.asarray(h.load("x", (h.READ,))[h.READ])
    fit = lf.MultiFit([xa**k for k in range(1, 6)], [256] * 5)
    return fit.predict(fit.solve(y[h.FIT]), [xb**k for k in range(1, 6)])


def history() -> dict:
    xv = np.asarray(h.load("x", (h.VALID,))[h.VALID])
    xb = np.asarray(h.load("x", (h.READ,))[h.READ])
    x = np.concatenate((xv, xb))  # history of the first B windows read from V
    c = np.concatenate(([0.0], np.cumsum(x**2)))
    starts = np.arange(h.SKIP, len(xb) - FRAME + 1, FRAME)
    ends = len(xv) + starts + FRAME  # window ends in concatenated coordinates

    def level(a, b):
        return lf.level_db((c[b] - c[a]) / (b - a))

    current = level(ends - CURRENT, ends)
    past = {
        t: level(ends - CURRENT - t * h.FS // 1000, ends - CURRENT) for t in HORIZONS_MS
    }
    qbins = np.quantile(current, np.linspace(0, 1, 6))
    cbin = np.clip(np.searchsorted(qbins[1:-1], current, side="right"), 0, 4)
    out = {}
    for t in [*STUDY, *SYNTH]:
        if t in SYNTH:
            y = synthetic_contiguous(t)
            yb, q = y[h.READ], h5_fit_predict(y)
        else:
            yb = np.asarray(h.load(t, (h.READ,))[h.READ])
            q = np.asarray(h.frozen_prediction("hammerstein_5", t))
        idx = starts[:, None] + np.arange(FRAME)
        ey = np.sum(yb[idx] ** 2, axis=1)
        ee = np.sum((yb - q)[idx] ** 2, axis=1)
        entry = (take_info(t) if isinstance(t, int) else {}) | {
            "h5_capacity_B": h.ratio(yb, q)
        }
        for hz, lp in past.items():
            d = lp - current
            cells, contrast = [], []
            for b in range(5):
                pick = cbin == b
                terc = np.quantile(d[pick], [1 / 3, 2 / 3])
                which = np.searchsorted(terc, d[pick], side="right")
                row = [
                    float(
                        np.sqrt(ee[pick][which == k].sum() / ey[pick][which == k].sum())
                    )
                    for k in range(3)
                ]
                cells.append(row)
                contrast.append(row[2] - row[0])
            r = 10 * np.log10(np.maximum(ee, 1e-300) / np.maximum(ey, 1e-300))
            design = np.column_stack(
                [(cbin == b).astype(float) for b in range(5)] + [lp]
            )
            coef = np.linalg.lstsq(design, r, rcond=None)[0]
            entry[f"{hz}ms"] = {
                "capacity_by_current_bin_and_past_tercile": cells,
                "contrast_past_louder_minus_quieter": contrast,
                "mean_contrast": float(np.mean(contrast)),
                "slope_db_per_db_past_level": float(coef[-1]),
            }
        out[str(t)] = entry
        log("history", t)
    return {
        "note": "B, 10 ms windows after SKIP; current level: x over the 50 ms ending "
        "at the window's end; past level: x over the T ms before that (from V for the "
        "first windows); 5 equal-count current-level bins x terciles of past minus "
        "current level; capacity = sqrt(sum e^2 / sum y^2) per cell; contrast = "
        "louder-past tercile minus quieter-past tercile; slope: per-window residual "
        "share (dB) on current-bin dummies + past level. Frozen H5 predictions for "
        "the takes, the same H5 refitted for the synthetic devices",
        "current_bin_edges_db": [float(v) for v in qbins],
        "windows": len(starts),
        "tags": out,
    }


# ---------------------------------------------------------------- lpv (b)


def level_z(power: np.ndarray, a: int) -> np.ndarray:
    lv = lf.level_db(power)
    return (lv - lv[:a].mean()) / lv[:a].std()


def lpv_features(x: np.ndarray, z: np.ndarray, powers: int = POWERS) -> list:
    return [x**k * z**j for k in range(1, 6) for j in range(powers)]


def split(v: np.ndarray, a: int, b: int) -> tuple:
    return v[:a], v[a : a + b], v[a + b :]


def capacities(fit, theta, feats_v, feats_b, y) -> dict:
    return {
        "V": h.ratio(y[h.VALID], fit.predict(theta, feats_v)),
        "B": h.ratio(y[h.READ], fit.predict(theta, feats_b)),
    }


def lpv() -> dict:
    xs = load_avb("x")
    a, v = len(xs[0]), len(xs[1])
    x = np.concatenate(xs)
    tags = [*STUDY, *SYNTH]
    ys = outputs(tags)
    targets = {t: ys[t][h.FIT] for t in tags}
    # linearity in the target, pair (1, 2): fit(y2) - fit(y1) - fit(y2 - y1)
    pair = {o: np.asarray(h.load(o, (h.FIT,))[h.FIT]) for o in (1, 2)}
    result = {"families": {}}

    def run(label: str, z: np.ndarray | None) -> dict:
        t0 = time.time()
        powers = 1 if z is None else POWERS
        zz = np.ones_like(x) if z is None else z
        fa, fv, fb = (
            lpv_features(s, w, powers)
            for s, w in zip(split(x, a, v), split(zz, a, v), strict=True)
        )
        fit = lf.MultiFit(fa, [256] * len(fa))
        entry = {"capacity": {}}
        for t in tags:
            entry["capacity"][str(t)] = capacities(
                fit, fit.solve(targets[t]), fv, fb, ys[t]
            )
        q = {
            k: fit.predict(fit.solve(y), fb)
            for k, y in (("1", pair[1]), ("2", pair[2]), ("d", pair[2] - pair[1]))
        }
        gap = q["2"] - q["1"] - q["d"]
        entry["linearity_gap_pair_1_2"] = nr.rms(gap[h.SKIP :]) / nr.rms(
            q["d"][h.SKIP :]
        )
        log("lpv", label, f"{time.time() - t0:.0f} s")
        return entry

    result["h5"] = run("h5", None)
    causal = {tau: lf.moving_power(x, tau * h.FS // 1000) for tau in TAUS_MS}
    for family in ("causal", "anticausal", "placebo"):
        result["families"][family] = {}
        for tau in TAUS_MS:
            w = tau * h.FS // 1000
            if family == "causal":
                power = causal[tau]
            elif family == "anticausal":
                power = lf.moving_power(x, w, causal=False)
            else:
                fill = float(np.median(causal[tau][:a]))
                power = lf.delayed(causal[tau], PLACEBO_DELAY, fill)
            result["families"][family][f"{tau}ms"] = run(
                f"{family} {tau}", level_z(power, a)
            )
    # selection on V: mean V capacity over the gains 2.5-10
    selected = {}
    for family, by_tau in result["families"].items():
        score = {
            k: float(np.mean([e["capacity"][str(o)]["V"] for o in HIGH]))
            for k, e in by_tau.items()
        }
        selected[family] = {"tau": min(score, key=score.get), "mean_V_high": score}
    result["selected_on_V"] = selected
    # oracle: level of the true past output (strictly past samples)
    result["oracle_output_level"] = oracle(xs, ys)
    result["reading"] = lpv_reading(result)
    result["note"] = (
        "features x^k z^j, k 1..5, j 0..2, 256 taps each, ridge 1e-4 (MultiFit); z: "
        "standardized log moving RMS of x over tau (causal: past tau incl. current "
        "sample; anticausal: next tau, a non-causal control; placebo: causal level "
        "delayed by 10 s), computed on x over A+V+B contiguous (input only); fit on "
        "A, capacity on V and B (SKIP 2400, floor 1e-4); S0/S1 synthetic devices; "
        "takes 9, 13, 18, 23, 24, 38 are gate-4 excluded (presence 10)"
    )
    result["preregistered"] = PREREGISTERED
    return result


def oracle(xs: list, ys: dict) -> dict:
    """ORACLE: level features from the true output's strictly past samples (on B it
    reads y_B). The features depend on y: the probe is not linear in y."""
    a, v = len(xs[0]), len(xs[1])
    x = np.concatenate(xs)
    out = {}
    for t in [*HIGH, GAINS[0.0], 23, "S1"]:
        y = np.concatenate([ys[t][s] for s in (h.FIT, h.VALID, h.READ)])
        entry = {}
        for tau in ORACLE_TAUS_MS:
            t0 = time.time()
            power = lf.moving_power(lf.delayed(y, 1, 0.0), tau * h.FS // 1000)
            z = level_z(power, a)
            fa, fv, fb = (
                lpv_features(s, w)
                for s, w in zip(split(x, a, v), split(z, a, v), strict=True)
            )
            fit = lf.MultiFit(fa, [256] * len(fa))
            entry[f"{tau}ms"] = capacities(fit, fit.solve(ys[t][h.FIT]), fv, fb, ys[t])
            log("oracle", t, tau, f"{time.time() - t0:.0f} s")
        out[str(t)] = entry
    return out


def lpv_reading(result: dict) -> dict:
    tau = result["selected_on_V"]["causal"]["tau"]
    cap = lambda fam, o: result["families"][fam][tau]["capacity"][str(o)]["B"]  # noqa: E731
    h5 = {o: result["h5"]["capacity"][str(o)]["B"] for o in HIGH}
    drop = {f: {o: h5[o] - cap(f, o) for o in HIGH} for f in result["families"]}
    c, ac, pl = (np.array(list(drop[f].values())) for f in drop)
    if c.max() <= BASIS_FLOOR:
        verdict = "refuted_for_H2"
    elif c.min() > BASIS_FLOOR and (pl < 0.01).all() and c.mean() - ac.mean() > 0.01:
        verdict = "state_explains_part"
    elif c.min() > BASIS_FLOOR and ac.mean() >= c.mean() - 0.01:
        verdict = "level_normalization_not_history"
    else:
        verdict = "inconclusive"
    return {
        "tau_selected_on_V": tau,
        "drop_B_at_tau": {
            f: {str(o): float(v) for o, v in d.items()} for f, d in drop.items()
        },
        "verdict": verdict,
    }


# ---------------------------------------------------------------- narx (c)

NARX = {
    "arx_na2": (2, 1),  # (lags, powers of the past output)
    "arx_na8": (8, 1),
    "narx_poly3_na2": (2, 3),
}


def narx() -> dict:
    xs = load_avb("x")
    feats = {
        s: [xi**k for k in range(1, 6)]
        for s, xi in zip((h.FIT, h.VALID, h.READ), xs, strict=True)
    }
    out = {}
    for o in GAINS.values():
        y = dict(zip((h.FIT, h.VALID, h.READ), load_avb(o), strict=True))
        limit = 100 * float(np.max(np.abs(y[h.FIT])))
        entry = take_info(o)
        for name, (lags, powers) in NARX.items():
            t0 = time.time()
            past = {s: lf.delayed(y[s], 1, 0.0) for s in y}
            ylag = {s: [past[s] ** m for m in range(1, powers + 1)] for s in y}
            fit = lf.MultiFit(feats[h.FIT] + ylag[h.FIT], [256] * 5 + [lags] * powers)
            theta = fit.solve(y[h.FIT])
            res = {
                "one_step_B_ORACLE": h.ratio(
                    y[h.READ], fit.predict(theta, feats[h.READ] + ylag[h.READ])
                )
            }
            a = np.array(
                [
                    theta[fit.first[5 + m] : fit.first[5 + m] + lags]
                    / fit.scales[5 + m]
                    for m in range(powers)
                ]
            )
            if powers == 1:
                roots = np.roots(np.concatenate(([1.0], -a[0])))
                res["ar_max_root_modulus"] = float(np.max(np.abs(roots)))
            for s in (h.VALID, h.READ):
                c, parts = fit.parts(theta, feats[s] + [np.zeros(len(y[s]))] * powers)
                u = c + np.sum(parts[:5], axis=0)
                q, stop = lf.narx_free_run(u, a, limit)
                res[f"free_run_{s}"] = {
                    "capacity": h.ratio(y[s], q) if stop < 0 else float("nan"),
                    "diverged_at_sample": int(stop),
                }
            entry[name] = res
            log("narx", o, name, f"{time.time() - t0:.0f} s")
        entry["h5_B"] = h.ratio(
            y[h.READ], np.asarray(h.frozen_prediction("hammerstein_5", o))
        )
        out[str(GAINS_BY_TAKE[o])] = entry
    return {
        "note": "features x^1..x^5 (256 taps) plus past outputs y[n-1..n-lags] (and "
        "their squares and cubes for poly3), equation-error ridge fit on A (MultiFit, "
        "ridge 1e-4); one_step_B_ORACLE uses the true past outputs of B; free run: the "
        "model fed back its own past outputs from a zero state, stopped when |q| > "
        "100 max|y_A|; nonlinear in y (the regressors contain the target)",
        "by_gain": out,
    }


GAINS_BY_TAKE = {o: f"{g:g}" for g, o in GAINS.items()}


# ---------------------------------------------------------------- summary


def summary() -> dict:
    """Arithmetic on the stored results of the other stages (no new measurement):
    the drops, bounds and contrasts quoted in the conclusions."""
    r = json.loads(OUT.read_text())
    high, levels = [str(o) for o in HIGH], ("-30", "-24", "-18")
    h5 = {t: e["B"] for t, e in r["lpv"]["h5"]["capacity"].items()}
    lpv_drops = {}
    for family, by_tau in r["lpv"]["families"].items():
        lpv_drops[family] = {}
        for tau, e in by_tau.items():
            d = {t: h5[t] - e["capacity"][t]["B"] for t in h5}
            lpv_drops[family][tau] = {
                "gain_ge_2.5_min": min(d[t] for t in high),
                "gain_ge_2.5_max": max(d[t] for t in high),
                "gain_0": d[str(GAINS[0.0])],
                "presence10": {str(o): d[str(o)] for o in STUDY if o in PRESENCE10},
                "S0": d["S0"],
                "S1": d["S1"],
                "S1_minus_S0": d["S1"] - d["S0"],
            }
    oracle = {
        tau: {
            t: h5[t] - e[tau]["B"] for t, e in r["lpv"]["oracle_output_level"].items()
        }
        for tau in (f"{t}ms" for t in ORACLE_TAUS_MS)
    }
    ms = r["multisine"]["tags"]
    relaxation = {}
    for group, tags in (
        ("gain_ge_2.5", high),
        ("gain_0", [str(GAINS[0.0])]),
        ("S0", ["S0"]),
        ("S1", ["S1"]),
    ):
        entry = {}
        for band in ("10-30ms", "30-100ms", "100-300ms", "300-1000ms"):
            v = max(ms[t][lv]["p1_vs_p3_db"][band] for t in tags for lv in levels)
            entry[band] = {"max_db": v, "max_amplitude_ratio": 10 ** (v / 20)}
        entry["p2_vs_p3_max_db"] = max(
            ms[t][lv]["p2_vs_p3_db"] for t in tags for lv in levels
        )
        relaxation[group] = entry
    st = r["steps"]["tags"]
    envelope = {}
    for group, tags in (
        ("gain_ge_2.5", high),
        ("input_x", ["x"]),
        ("S0", ["S0"]),
        ("S1", ["S1"]),
    ):
        envelope[group] = {
            band: max(
                abs(st[t][n][k][band])
                for t in tags
                for n in nr.STEPS
                for k in ("release", "attack")
            )
            for band in ("10-30ms", "30-100ms", "100-300ms")
        }
    narx_drop = {
        g: {
            k: e["h5_B"] - e[k]["free_run_di_test"]["capacity"]
            if e[k]["free_run_di_test"]["capacity"] is not None
            else None
            for k in NARX
        }
        for g, e in r["narx"]["by_gain"].items()
    }
    return {
        "note": "drops = H5 capacity on B minus the probe's; a component of relative "
        "RMS s removed from the residual lowers the capacity by at most s (triangle "
        "inequality), so reaching the H2 threshold 0.3 needs a mechanism carrying at "
        "least H5 - 0.3 of the output RMS; multisine: max over the gain takes 2.5-10 "
        "and the three levels of RMS(period 1 - period 3) / RMS(period 3) per band "
        "(after 0.5 s of silence); steps: max |envelope deviation| (dB)",
        "h5_B": {f"{g:g}": h5[str(o)] for g, o in GAINS.items()},
        "needed_share_for_threshold_0.3": {
            f"{g:g}": h5[str(o)] - 0.3 for g, o in GAINS.items()
        },
        "lpv_drops": lpv_drops,
        "oracle_output_level_drops": oracle,
        "multisine_relaxation": relaxation,
        "steps_envelope_abs_db": envelope,
        "narx_free_run_drop_B": narx_drop,
    }


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args()
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    result["protocol"] = (
        "fit di_train (A), select on di_validation (V), read di_test (B) at 48 kHz, "
        "SKIP 2400, floor 1e-4 (harness.py); model-free stages read the renders"
    )
    result["groups"] = {
        "gain_takes": {f"{g:g}": o for g, o in GAINS.items()},
        "presence10": list(PRESENCE10),
        "presence10_gain_ge5": list(PRESENCE10_HIGH),
        "excluded_gate4": list(h.EXCLUDED),
    }
    for stage in args.stages:
        t0 = time.time()
        result[stage] = globals()[stage]()
        result[stage]["seconds"] = round(time.time() - t0, 1)
        OUT.write_text(json.dumps(nr.finite(result), indent=1) + "\n")
        log("stage", stage, "written")


if __name__ == "__main__":
    main()
