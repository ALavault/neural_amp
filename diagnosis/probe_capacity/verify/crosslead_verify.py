#!/usr/bin/env python3
"""Cross-lead skeptic check of the probe-capacity leads 1-6 and of the shared harness.

Independent recomputations (own code; harness primitives and cached arrays only):
- instrument: hammerstein_5 and volterra_laguerre_3 refitted from scratch with the
  library (fit_probe + probe_contributions, no prediction cache) on two gains, scored
  with an own ratio, against readings.H2_capacity.by_gain of nolly_reading.json;
  FeatureProbe [x**1..x**5] against that refit on a take no lead used for parity;
- protocol: SKIP, floor, decimation (cache against a fresh cut + nr.dec of the 192 kHz
  files), H2 ladder takes;
- quotes: every frozen/scratch baseline the lead reports quote, looked up in
  baseline.json, nolly_reading.json and the scratch capacity_split.json;
- scratch Wiener-HP300 scaling: the scratch features (z scaled by the RMS of the
  section being read, i.e. by x_B on B) against the same features scaled by the RMS
  on A (in protocol), takes 30 and 20; linearity in y of the in-protocol variant;
- lead 4: multisine D (own BLA code), frozen probes read in the matched bands (own
  window code on cached predictions), and a discriminating schedule test: the band
  FIRs (own row-masked Gram, fitted on A) switched by the current 50 ms window
  (lead 4 "non-causal"), the previous window (lead 4 "causal"), and per-sample
  trailing / leading running levels of 10 and 50 ms;
- H5 on V for take 20 (lead 5 quotes 0.842).

CPU only, from the repository root:
    uv run python diagnosis/probe_capacity/verify/crosslead_verify.py \
        --scratch <session scratchpad>/nl_research/capacity_split.json
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
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as h  # noqa: E402

from fssr_nam.product.fidelity import Protocol  # noqa: E402
from fssr_nam.product.probes import fit_probe, probe_contributions  # noqa: E402

nr = h.nr
ROOT = h.ROOT
OUT = HERE / "crosslead_verify.json"
TMP = h.CACHE / "verify"
BASE = json.loads((HERE.parent / "baseline.json").read_text())
LEADS = {
    i: json.loads((HERE.parent / f"{name}.json").read_text())
    for i, name in (
        (1, "lead1_output_referenced"),
        (4, "lead4_bla_ceiling"),
        (5, "lead5_state_feedback"),
    )
}


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def own_ratio(y: np.ndarray, q: np.ndarray, floor: float = 1e-4) -> float:
    y, q = np.asarray(y), np.asarray(q)
    return rms((y - q)[2400:]) / max(rms(y[2400:]), floor)


def sec(tag, name):
    return np.asarray(h.load(tag, (name,))[name])


# ---------------------------------------------------------------- instrument


def instrument() -> dict:
    out = {"H2": {}, "feature_probe": {}}
    xa, xb = sec("x", h.FIT), sec("x", h.READ)
    for probe in ("hammerstein_5", "volterra_laguerre_3"):
        for gain, take in (("0", 30), ("10", 7)):
            t0 = time.time()
            result = fit_probe(h.FROZEN[probe], xa, sec(take, h.FIT))
            q = sum(probe_contributions(result, xb).values())
            mine = own_ratio(sec(take, h.READ), q)
            frozen = h.READING["readings"]["H2_capacity"][probe]["by_gain"][gain]
            cached = own_ratio(sec(take, h.READ), h.frozen_prediction(probe, take))
            out["H2"][f"{probe}|G{gain}|take{take}"] = {
                "fresh_refit": mine,
                "nolly_reading": frozen,
                "rel_diff": abs(mine - frozen) / frozen,
                "harness_cache": cached,
                "seconds": round(time.time() - t0, 1),
            }
            if probe == "hammerstein_5" and take == 7:
                fp = h.FeatureProbe(lambda i: [i["x"] ** k for k in range(1, 6)])
                fp.fit({"x": xa}, sec(take, h.FIT))
                qf = fp.predict({"x": xb})
                out["feature_probe"] = {
                    "take": take,
                    "feature_probe_x1_x5": own_ratio(sec(take, h.READ), qf),
                    "hammerstein_5_refit": mine,
                    "max_rel_prediction_gap": float(
                        np.max(np.abs(qf - q)) / np.max(np.abs(q))
                    ),
                }
    return out


# ---------------------------------------------------------------- protocol


def protocol() -> dict:
    out = {
        "SKIP": h.SKIP,
        "SKIP_is_0.05s_at_48k": h.SKIP == round(0.05 * 48_000) == nr.SKIP,
        "FLOOR": h.FLOOR,
        "FLOOR_is_Protocol_floor": h.FLOOR
        == Protocol.__dataclass_fields__["floor"].default
        == 1e-4,
        "FS": h.FS,
        "DOWN": nr.DOWN,
        "gain_takes": {f"{g:g}": t for g, t in h.GAIN_TAKES.items()},
        "gain_takes_from_settings": {
            f"{g:g}": t for g, t in sorted(nr.gain_takes(h.ROWS).items())
        },
    }
    fresh = {}
    for tag, path in (("x", h.SOURCE), (20, h.RENDERS / h.ROWS[20]["file"])):
        raw = nr.read_sections(path, (h.READ,))[h.READ]
        mine = resample_poly(raw, 1, 4)
        cached = sec(tag, h.READ)
        fresh[str(tag)] = {
            "samples_192k": len(raw),
            "samples_48k": len(mine),
            "max_abs_diff_vs_cache": float(np.max(np.abs(mine - cached))),
        }
    out["decimation_di_test"] = fresh
    return out


# ---------------------------------------------------------------- quotes


def quotes(scratch: dict | None) -> dict:  # noqa: C901 (flat lookup table)
    P = BASE["probes"]

    def gain(probe, g):
        return P[probe]["capacity_by_gain"][g]

    def share(probe, ref, ev):
        for s in P[probe]["difference_share_one_knob"]:
            if s["ref"] == ref and s["eval"] == ev:
                return s["share_from_x"]
        return None

    def take(probe, t):
        return P[probe]["capacity_by_take"][str(t)]["value"]

    def level(probe, g, b):
        return P[probe]["capacity_by_level"][g][b]["capacity"]

    gains = ("0", "2.5", "5", "7.5", "10")
    rows = []

    def q(lead, label, quoted, actual, source, note=""):
        rows.append(
            {
                "lead": lead,
                "label": label,
                "quoted": quoted,
                "actual": actual,
                "source": source,
                "abs_diff": None if actual is None else abs(quoted - actual),
                "exists": actual is not None,
                "note": note,
            }
        )

    ladder = {
        "hammerstein_5": (0.476, 0.794, 0.803, 0.800, 0.747),
        "volterra_laguerre_3": (0.497, 0.786, 0.793, 0.790, 0.754),
        "fir": (0.511, 0.825, 0.831, 0.830, 0.800),
    }
    for lead in (1, 2, 3, 5, 6):
        for probe, vals in ladder.items():
            if probe == "fir" or (lead == 5 and probe != "hammerstein_5"):
                continue
            for g, v in zip(gains, vals, strict=True):
                q(lead, f"{probe} G{g}", v, gain(probe, g), "baseline.json")
    for g, v in zip(gains, ladder["fir"], strict=True):
        q("harness", f"fir G{g}", v, gain("fir", g), "baseline.json")
    # lead 4 power shares = capacity^2
    l4 = {
        "fir": (0.261, 0.681, 0.691, 0.689, 0.640),
        "hammerstein_5": (0.227, 0.631, 0.644, 0.640, 0.558),
        "volterra_laguerre_3": (0.247, 0.618, 0.630, 0.625, 0.568),
    }
    for probe, vals in l4.items():
        for g, v in zip(gains, vals, strict=True):
            q(4, f"{probe}^2 G{g} (power)", v, gain(probe, g) ** 2, "baseline.json")
    for b, v in enumerate((0.864, 0.764, 0.700, 0.567, 0.592)):
        q(4, f"fir^2 G5 quintile {b + 1}", v, level("fir", "5", b) ** 2, "baseline")
    for b, v in enumerate((0.313, 0.331, 0.324, 0.235, 0.219)):
        q(4, f"fir^2 G0 quintile {b + 1}", v, level("fir", "0", b) ** 2, "baseline")
    usable_hi = [t for t in h.USABLE if float(h.ROWS[t]["G"]) >= 2.5]
    q(
        4,
        "median fir^2 over usable G>=2.5 takes",
        0.687,
        float(np.median([take("fir", t) ** 2 for t in usable_hi])),
        "baseline.json",
        f"n = {len(usable_hi)}",
    )
    # H5 level bins (leads 2, 3, 6 and the harness)
    h5_bins = {
        "0": (0.412, 0.456, 0.485, 0.456, 0.502),
        "2.5": (0.881, 0.818, 0.788, 0.733, 0.765),
        "5": (0.895, 0.830, 0.806, 0.739, 0.750),
        "7.5": (0.898, 0.826, 0.803, 0.733, 0.744),
        "10": (0.880, 0.784, 0.732, 0.653, 0.677),
    }
    for g, vals in h5_bins.items():
        for b, v in enumerate(vals):
            q("2,3,6", f"H5 G{g} bin {b + 1}", v, level("hammerstein_5", g, b), "base")
    # one-knob shares quoted by leads 1, 2, 6
    h5_vl3 = {
        (1, 2): (0.829, 0.886),
        (1, 3): (0.809, 0.772),
        (1, 4): (0.877, 0.863),
        (1, 5): (0.888, 0.868),
        (1, 6): (0.979, 0.981),
        (1, 7): (0.905, 0.905),
        (1, 10): (0.767, 0.764),
        (20, 11): (0.872, 0.850),
        (20, 16): (0.970, 0.972),
        (20, 21): (0.828, 0.882),
        (20, 22): (0.808, 0.829),
        (20, 26): (0.801, 0.764),
        (20, 28): (0.914, 0.898),
        (39, 30): (0.951, 0.955),
        (39, 31): (0.875, 0.858),
        (39, 33): (0.873, 0.850),
        (39, 34): (0.794, 0.760),
        (39, 35): (0.794, 0.764),
        (39, 36): (0.882, 0.862),
        (20, 23): (0.955, 0.948),
    }
    for (r, e), (a, b) in h5_vl3.items():
        q("1,2,3,6", f"H5 share {r}->{e}", a, share("hammerstein_5", r, e), "base")
        q("2,3,6", f"VL3 share {r}->{e}", b, share("volterra_laguerre_3", r, e), "base")
    # lead 2 take-level quotes
    usable = [take("hammerstein_5", t) for t in h.USABLE]
    for label, v, a in (
        ("H5 usable min", 0.476, min(usable)),
        ("H5 usable median", 0.799, float(np.median(usable))),
        ("H5 usable max", 0.836, max(usable)),
        ("H5 gate-4 min", 0.756, min(take("hammerstein_5", t) for t in h.EXCLUDED)),
        ("H5 gate-4 max", 0.874, max(take("hammerstein_5", t) for t in h.EXCLUDED)),
    ):
        q(2, label, v, a, "baseline.json")
    for t, v in ((37, 0.764), (7, 0.747), (15, 0.743), (19, 0.717), (29, 0.821)):
        q(2, f"H5 take {t}", v, take("hammerstein_5", t), "baseline.json")
    g5 = [t for t in h.USABLE if float(h.ROWS[t]["G"]) == 5.0 and t not in (1, 20, 39)]
    g5_vals = [take("hammerstein_5", t) for t in g5]
    q(2, "H5 other G5 takes min ('0.77')", 0.77, min(g5_vals), "baseline", f"{g5}")
    q(2, "H5 other G5 takes max ('0.84')", 0.84, max(g5_vals), "baseline", f"{g5}")
    # lead 5 presence-10 takes
    for t, v in (
        (9, 0.836),
        (13, 0.874),
        (18, 0.756),
        (23, 0.826),
        (24, 0.814),
        (38, 0.829),
    ):
        q(5, f"H5 take {t} (presence 10, gate 4)", v, take("hammerstein_5", t), "base")
    # lead 1 class medians of the frozen share, recomputed from nolly_reading.json
    reading = {(p["ref"], p["eval"]): p for p in h.READING["pairs"]}
    by_class = LEADS[1]["nolly"]["by_class"]
    for cls, (v5, v3) in {
        "tone": (0.829, 0.850),
        "gain": (0.933, 0.934),
        "cab": (0.830, 0.858),
    }.items():
        pairs = [tuple(map(int, s.split("->"))) for s in by_class[cls]["pairs"]]
        for probe, v in (("hammerstein_5", v5), ("volterra_laguerre_3", v3)):
            vals, missing = [], [p for p in pairs if p not in reading]
            for p in pairs:
                if p in reading:
                    seg = reading[p]["probes"][probe][h.READ]
                    vals.append(seg["unrepresented"] / seg["d_audio_norm"])
            q(
                1,
                f"median frozen {probe} share, class {cls}",
                v,
                float(np.median(vals)),
                "nolly_reading.json pairs",
                f"n = {len(vals)}, missing from reading: {missing}",
            )
    # scratch numbers
    if scratch is not None:
        S, SP = scratch["structure"], scratch["pairs_vs_01"]
        for key, vals, leads in (
            ("wsat_bla_256", (0.381, 0.729, 0.748, 0.748, 0.683), "2 (3: 0.748)"),
            ("wsat_hp300_256", (0.520, 0.614, 0.692, 0.696, 0.710), "2 (3: 0.6919)"),
        ):
            for g, v in zip(gains, vals, strict=True):
                t = h.GAIN_TAKES[float(g)]
                q(leads, f"scratch {key} G{g}", v, S[str(t)][key], "capacity_split")
        for label, v, ev, key in (
            ("yfir 1->2", 0.488, "2", "yref_fir256"),
            ("yfir1024 1->2", 0.411, "2", "yref_fir1024"),
            ("yfir 1->4", 0.461, "4", "yref_fir256"),
            ("yfir 1->7", 0.664, "7", "yref_fir256"),
            ("yfir 39->30 (scratch: 1->30)", 0.461, "30", "yref_fir256"),
        ):
            q("1,6", label, v, SP[ev][key], "capacity_split pairs_vs_01")
        out_scratch = {
            "pairs_exposed_on_B_by_scratch": sorted(int(k) for k in SP),
            "scratch_reference_take": 1,
            "lead1_says": "6 pairs: 1->2, 1->4, 1->7, 39->30, 39->35, 39->36",
        }
    else:
        out_scratch = None
    # lead 1 vs lead 6: FIR fed y_ref
    l1 = {(p["ref"], p["eval"]): p for p in LEADS[1]["nolly"]["pairs"]}
    l6_yref = {
        (1, 3): 0.149,
        (20, 26): 0.191,
        (39, 34): 0.250,
        (39, 35): 0.298,
        (1, 5): 0.308,
        (20, 28): 0.421,
        (39, 36): 0.287,
        (1, 4): 0.461,
        (20, 11): 0.385,
        (39, 31): 0.438,
        (39, 33): 0.359,
        (1, 2): 0.488,
        (1, 10): 0.174,
        (20, 21): 0.472,
        (20, 22): 0.360,
        (1, 6): 0.671,
        (1, 7): 0.664,
        (20, 16): 0.840,
        (39, 30): 0.461,
    }
    for p, v in l6_yref.items():
        a = l1[p]["share"]["yfir"][h.READ] if p in l1 else None
        q(6, f"FIR fed y_ref {p[0]}->{p[1]} vs lead 1 yfir", v, a, "lead1 JSON")
    bad = [r for r in rows if (not r["exists"]) or r["abs_diff"] > 0.005]
    rounding = [r for r in rows if r["exists"] and 0.0006 < r["abs_diff"] <= 0.005]
    return {
        "checked": len(rows),
        "mismatch_over_0.005_or_absent": bad,
        "rounding_gaps_0.0006_to_0.005": rounding,
        "scratch_exposure": out_scratch,
        "rows": rows,
    }


# ---------------------------------------------------------------- scratch scaling


def scratch_scaling() -> dict:
    """Scratch Wiener-HP300 (capacity_split.py hpfeat): z = hp(x) / RMS(hp(x)) of the
    section being processed (x_B on B), 7 tanh/sign features + x, 256 taps, ridge
    1e-4. In protocol: the same with the RMS of hp(x_A)."""
    sos = butter(2, 300, "highpass", fs=48_000, output="sos")
    xa = sec("x", h.FIT)
    scale_a = rms(sosfilt(sos, xa))

    def feats(x, scale):
        u = sosfilt(sos, x)
        z = u / (rms(u) if scale is None else scale)
        return [x] + [np.tanh(a * z) for a in (0.5, 1, 2, 4, 8, 16)] + [np.sign(z)]

    xb = sec("x", h.READ)
    out = {
        "scale_A": scale_a,
        "scale_B": rms(sosfilt(sos, xb)),
        "scale_ratio_B_over_A_db": 20 * np.log10(rms(sosfilt(sos, xb)) / scale_a),
    }
    for take in (30, 20):
        row = {}
        for label, scale in (("per_section_scratch", None), ("A_scaled", scale_a)):
            probe = h.FeatureProbe(lambda i, s=scale: feats(i["x"], s))
            row[label] = h.output_capacity(probe, take)
        out[f"take_{take}"] = row
    probe = h.FeatureProbe(lambda i: feats(i["x"], scale_a))
    out["linearity_A_scaled_pair_1_7"] = h.linearity_check(probe, 1, 7)
    return out


# ---------------------------------------------------------------- lead 4


def multisine_d(take: int) -> dict:
    """Own BLA: 4 realizations, mean of periods 2-3, excited lines found from |U|,
    BLA = mean over realizations of Y/U; D_plain = 1 - BLA power share (no bias
    correction, biased low by the stochastic share / 4); D_bias_corrected subtracts
    var(G) / 4 from |mean G|^2 (lead 4's estimator); D_loo = residual power share with
    the BLA of the other three (biased high by the stochastic share / 3)."""
    out = {}
    for lvl in (-30, -24, -18):
        X, Y = [], []
        for r in (1, 2, 3, 4):
            name = f"multisine_{lvl}dBFS_rms_r{r}"
            x = sec("x", name)[48_000:].reshape(2, 48_000)
            y = sec(take, name)[48_000:].reshape(2, 48_000)
            X.append(np.fft.rfft(x, axis=1))
            Y.append(np.fft.rfft(y, axis=1))
        X, Y = np.array(X), np.array(Y)  # (r, period, line)
        U, Ym = X.mean(1), Y.mean(1)
        mag = np.abs(U)
        excited = np.all(mag > 1e-3 * mag.max(axis=1, keepdims=True), axis=0)
        w = np.full(U.shape[1], 2.0)
        w[0] = w[-1] = 1.0
        G = Ym[:, excited] / U[:, excited]
        total = np.mean(np.sum(w * np.abs(Y) ** 2, axis=-1))
        bla = np.mean(
            np.sum(w[excited] * np.abs(G.mean(0) * U[:, excited]) ** 2, axis=-1)
        )
        g_var = np.var(G, axis=0, ddof=1)  # |mean G|^2 is biased up by var / 4
        bla_corr = np.mean(
            np.sum(
                w[excited]
                * (np.abs(G.mean(0)) ** 2 - g_var / 4)
                * np.abs(U[:, excited]) ** 2,
                axis=-1,
            )
        )
        loo = []
        for k in range(4):
            g = np.delete(G, k, axis=0).mean(0)
            res = Y[k].copy()
            res[:, excited] -= g * X[k][:, excited]
            loo.append(np.mean(np.sum(w * np.abs(res) ** 2, axis=-1)))
        out[str(lvl)] = {
            "excited_lines": int(excited.sum()),
            "D_plain": 1 - bla / total,
            "D_bias_corrected": 1 - bla_corr / total,
            "D_loo": float(np.mean(loo)) / total,
        }
    return out


def windows(x: np.ndarray, start: int) -> np.ndarray:
    n = (len(x) - start) // h.WINDOW
    ms = np.mean(x[start : start + n * h.WINDOW].reshape(n, h.WINDOW) ** 2, 1)
    return 10 * np.log10(np.maximum(ms, 1e-30))


EDGES = (-33.0, -27.0, -21.0, -15.0)  # lead 4 schedule: 5 bands


def band_of(level: np.ndarray) -> np.ndarray:
    return np.searchsorted(EDGES, level, side="right")


def frozen_in_band(take: int, level_b: np.ndarray) -> dict:
    y = sec(take, h.READ)
    n = len(level_b)
    yw = y[h.SKIP : h.SKIP + n * h.WINDOW].reshape(n, h.WINDOW)
    out = {}
    for probe in ("fir", "hammerstein_5", "volterra_laguerre_3"):
        q = np.asarray(h.frozen_prediction(probe, take))
        rw = (y - q)[h.SKIP : h.SKIP + n * h.WINDOW].reshape(n, h.WINDOW)
        out[probe] = {}
        for c in (-30, -24, -18):
            m = (level_b >= c - 3) & (level_b < c + 3)
            out[probe][str(c)] = float(np.sum(rw[m] ** 2) / np.sum(yw[m] ** 2))
    return out


def running_level(x: np.ndarray, length: int, lead: bool) -> np.ndarray:
    """Per-sample level (dBFS) of the mean square over `length` samples ending at n
    (trailing, causal) or starting at n (leading, anticausal)."""
    c = np.concatenate(([0.0], np.cumsum(x**2)))
    n = np.arange(len(x))
    if lead:
        lo, hi = n, np.minimum(n + length, len(x))
    else:
        lo, hi = np.maximum(n - length + 1, 0), n + 1
    ms = (c[hi] - c[lo]) / (hi - lo)
    return 10 * np.log10(np.maximum(ms, 1e-30))


def band_firs(takes: tuple[int, ...]) -> dict:  # noqa: C901
    """Own reimplementation of lead 4's band FIRs: frozen `fir` spec (256 taps,
    constant, relative ridge 1e-2 of the band Gram's mean diagonal), fitted on the A
    rows of each schedule band (50 ms windows from 0), read on B with several band
    selectors. Gram from x only: linear in y by construction (checked)."""
    taps, ridge, chunk = 256, 1e-2, 40_000
    xa, xb = sec("x", h.FIT), sec("x", h.READ)
    level_a = windows(xa, 0)
    lab_a = np.full(len(xa), -1)
    lab_a[: len(level_a) * h.WINDOW] = np.repeat(band_of(level_a), h.WINDOW)
    t0 = time.time()
    view = sliding_window_view(np.concatenate((np.zeros(taps - 1), xa)), taps)
    grams = np.zeros((5, taps + 1, taps + 1))
    for start in range(0, len(xa), chunk):
        block = view[start : start + chunk][:, ::-1]
        lab = lab_a[start : start + chunk]
        for b in range(5):
            sel = block[lab == b]
            if len(sel) == 0:
                continue
            s = sel.sum(0)
            grams[b, 0, 0] += len(sel)
            grams[b, 0, 1:] += s
            grams[b, 1:, 0] += s
            grams[b, 1:, 1:] += sel.T @ sel
    gram_seconds = time.time() - t0
    size = fft.next_fast_len(len(xa) + taps)
    xs = fft.rfft(xa, size)

    def thetas(y):
        out = []
        for b in range(5):
            z = np.where(lab_a == b, y, 0.0)
            corr = fft.irfft(fft.rfft(z, size) * np.conj(xs), size)[:taps]
            rhs = np.concatenate(([z.sum()], corr))
            g = grams[b].copy()
            g[1:, 1:] += ridge * np.diag(grams[b])[1:].mean() * np.eye(taps)
            out.append(solve(g, rhs, assume_a="pos"))
        return out

    def render(theta, x):
        return theta[0] + fftconvolve(x, theta[1:])[: len(x)]

    level_b = windows(xb, h.SKIP)
    cur = band_of(level_b)
    prev = np.concatenate((cur[:1], cur[:-1]))

    def per_window(which):
        s = np.full(len(xb), which[-1])
        s[: h.SKIP] = which[0]
        s[h.SKIP : h.SKIP + len(which) * h.WINDOW] = np.repeat(which, h.WINDOW)
        return s

    selectors = {
        "window_current_noncausal (lead 4)": per_window(cur),
        "window_previous (lead 4 causal)": per_window(prev),
        "trailing_50ms_causal": band_of(running_level(xb, 2400, lead=False)),
        "trailing_10ms_causal": band_of(running_level(xb, 480, lead=False)),
        "leading_50ms_anticausal": band_of(running_level(xb, 2400, lead=True)),
        "leading_10ms_anticausal": band_of(running_level(xb, 480, lead=True)),
    }
    agree = {
        k: float(
            np.mean((v == selectors["window_current_noncausal (lead 4)"])[h.SKIP :])
        )
        for k, v in selectors.items()
    }
    out = {
        "gram_seconds": round(gram_seconds, 1),
        "selector_agreement_with_current": agree,
        "takes": {},
    }
    for take in takes:
        ya = sec(take, h.FIT)
        th = thetas(ya)
        renders = np.stack([render(t, xb) for t in th])
        yb = sec(take, h.READ)
        row = {"lead4_reported": None}
        for k, s in selectors.items():
            qb = renders[s, np.arange(len(xb))]
            row[k] = h.ratio(yb, qb) ** 2
        glob = grams.sum(0)  # rows in some window (tail rows of A excluded)
        g = glob.copy()
        g[1:, 1:] += ridge * np.diag(glob)[1:].mean() * np.eye(taps)
        z = np.where(lab_a >= 0, ya, 0.0)
        corr = fft.irfft(fft.rfft(z, size) * np.conj(xs), size)[:taps]
        th_g = solve(g, np.concatenate(([z.sum()], corr)), assume_a="pos")
        row["global_fir_own"] = h.ratio(yb, render(th_g, xb)) ** 2
        row["frozen_fir_baseline"] = h.ratio(yb, h.frozen_prediction("fir", take)) ** 2
        out["takes"][str(take)] = row
    # linearity in y of the band FIRs (pair 1 -> 7), current-window selector
    s = selectors["window_current_noncausal (lead 4)"]

    def predict(y):
        r = np.stack([render(t, xb) for t in thetas(y)])
        return r[s, np.arange(len(xb))]

    y1, y7 = sec(1, h.FIT), sec(7, h.FIT)
    qd = predict(y7 - y1)
    gap = predict(y7) - predict(y1) - qd
    out["linearity_pair_1_7"] = rms(gap[h.SKIP :]) / rms(qd[h.SKIP :])
    return out


def lead4() -> dict:
    lead4_json = LEADS[4]
    out = {"multisine_D": {}, "frozen_in_band": {}}
    for take in (20, 7, 30):
        out["multisine_D"][str(take)] = multisine_d(take)
    level_b = windows(sec("x", h.READ), h.SKIP)
    out["band_windows_B"] = {
        str(c): int(np.sum((level_b >= c - 3) & (level_b < c + 3)))
        for c in (-30, -24, -18)
    }
    for take in (20, 7):
        out["frozen_in_band"][str(take)] = frozen_in_band(take, level_b)
    # group medians over the 32 usable takes at G >= 2.5 (lead 4 table C)
    usable_hi = [t for t in h.USABLE if float(h.ROWS[t]["G"]) >= 2.5]
    med = {}
    per = {t: frozen_in_band(t, level_b) for t in usable_hi}
    for probe in ("fir", "hammerstein_5", "volterra_laguerre_3"):
        med[probe] = {
            str(c): float(np.median([per[t][probe][str(c)] for t in usable_hi]))
            for c in (-30, -24, -18)
        }
    out["group_median_in_band_G_ge_2.5"] = {"n": len(usable_hi), **med}
    out["band_firs"] = band_firs((20, 7, 16, 6, 30))
    out["lead4_json_keys"] = list(lead4_json.keys())
    return out


# ---------------------------------------------------------------- other


def h5_on_validation() -> dict:
    take = 20
    probe = h.FrozenProbe("hammerstein_5").fit({"x": sec("x", h.FIT)}, sec(take, h.FIT))
    out = {}
    for name in (h.VALID, h.READ):
        out[name] = own_ratio(sec(take, name), probe.predict({"x": sec("x", name)}))
    return out


def units() -> dict:
    """Lead 4 power shares converted to amplitude ratios (sqrt)."""
    power = {
        "lead4 non-causal scheduled FIR (G0..G10)": (0.193, 0.598, 0.612, 0.611, 0.542),
        "lead4 causal scheduled FIR (G0..G10)": (0.224, 0.707, 0.732, 0.733, 0.706),
        "lead4 multisine D read at window levels, B-weighted (G0..G10)": (
            0.071,
            0.416,
            0.431,
            0.450,
            0.475,
        ),
        "lead4 multisine D at -30/-24/-18, G5": (0.415, 0.426, 0.440),
        "lead4 DI band FIR group median -30/-24/-18": (0.655, 0.628, 0.560),
        "lead4 frozen FIR group median -30/-24/-18": (0.831, 0.715, 0.551),
    }
    amplitude = {
        k: [round(float(np.sqrt(v)), 4) for v in vals] for k, vals in power.items()
    }
    to_power = {
        "lead2 WH3 frozen from take 20, B (G0..G10)": (
            0.328,
            0.392,
            0.258,
            0.357,
            0.507,
        ),
        "lead3 hp300 union (G0..G10)": (0.260, 0.545, 0.612, 0.626, 0.654),
        "lead5 LPV causal 10 ms (G0..G10)": (0.437, 0.758, 0.765, 0.762, 0.704),
        "lead5 LPV anticausal 10 ms (G0..G10)": (0.447, 0.747, 0.762, 0.761, 0.704),
        "lead6 Nolly fir_pen (G0..G10)": (0.488, 0.666, 0.702, 0.719, 0.765),
    }
    power_of = {k: [round(v * v, 4) for v in vals] for k, vals in to_power.items()}
    return {
        "lead4_power_as_amplitude": amplitude,
        "others_amplitude_as_power": power_of,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--parts", nargs="*", default=None)
    args = parser.parse_args()
    scratch = json.loads(args.scratch.read_text()) if args.scratch else None
    parts = {
        "protocol": protocol,
        "quotes": lambda: quotes(scratch),
        "units": units,
        "h5_on_validation": h5_on_validation,
        "scratch_scaling": scratch_scaling,
        "lead4": lead4,
        "instrument": instrument,
    }
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, job in parts.items():
        if args.parts and name not in args.parts:
            continue
        t0 = time.time()
        result[name] = job()
        result.setdefault("seconds", {})[name] = round(time.time() - t0, 1)
        OUT.write_text(json.dumps(nr.finite(result), indent=1) + "\n")
        TMP.mkdir(parents=True, exist_ok=True)
        (TMP / OUT.name).write_text(json.dumps(nr.finite(result), indent=1) + "\n")
        print(name, result["seconds"][name], "s", flush=True)


if __name__ == "__main__":
    main()
