#!/usr/bin/env python3
"""Skeptic verification of lead 5 (state and feedback), CPU only, own code.

Recomputes, without calling the lead's stage functions:
  bound      arithmetic of the refutation (triangle and orthogonal bounds)
  multisine  period 1 vs period 3 (and 2 vs 3) per band from the cached sections, at
             the four gain takes 2.5-10 and take 30; fine 1 ms curve 0-40 ms, band
             5.33-10 ms (just beyond the 256-tap probe memory); where the 2-vs-3 floor
             sits in the period; synthetic S0/S1 (the lead's device helper, reused)
  clicks     192 kHz markers: click 2 vs click 1, click-1 residue before click 2,
             click 1 end vs start, for a few takes; click onsets in the source
  drift      presence-10 takes: whole marker section end vs start (as
             nolly_marker_drift), per-click split, spectra of the click difference
             and of the pre-click regrowth
  taps       Hammerstein 5 with 256 and 1024 taps (fit A, read B): memory 5.3-21 ms
  lpv        LPV causal / anticausal / placebo at 10 ms and causal 100 ms for the
             four gain takes 2.5-10 (fit A, V and B), own envelopes and own fit with
             the library normal equations; linearity identity on pair (1, 2)
  levels     short-term input level of B (50 ms windows) against the multisine RMS

    uv run python diagnosis/probe_capacity/verify/lead5_verify.py [--stages ...]
Writes verify/lead5_verify.json (each stage replaces its own key).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import fftconvolve, welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as h  # noqa: E402

from fssr_nam.product import nolly  # noqa: E402
from fssr_nam.product.probes import Probe, normal_equations, regularized  # noqa: E402

nr = h.nr
OUT = HERE / "lead5_verify.json"
LEAD = json.loads((HERE.parent / "lead5_state_feedback.json").read_text())
GAINS = h.GAIN_TAKES  # {0: 30, 2.5: 16, 5: 20, 7.5: 6, 10: 7}
HIGH = [GAINS[g] for g in (2.5, 5.0, 7.5, 10.0)]
FS = h.FS
STAGES = ("bound", "multisine", "clicks", "drift", "regrowth", "taps", "lpv", "levels")


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pdb(p: float) -> float:
    return float(10 * np.log10(max(p, 1e-300)))


def rms(v) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


# ---------------------------------------------------------------- bound


def bound() -> dict:
    h5 = LEAD["summary"]["h5_B"]
    out = {}
    for g, c in h5.items():
        out[g] = {
            "h5_B": c,
            "triangle_needed_share": c - 0.3,  # worst case: s aligned with the residual
            "orthogonal_needed_share": float(np.sqrt(max(c**2 - 0.09, 0.0))),
            "capacity_left_if_s_0.041_triangle": c - 0.041,
            "capacity_left_if_s_0.041_orthogonal": float(np.sqrt(c**2 - 0.041**2)),
            "capacity_left_if_s_0.19_triangle": c - 0.19,
            "capacity_left_if_s_0.19_orthogonal": float(np.sqrt(c**2 - 0.19**2)),
        }
    # Synthetic calibration: S1 state share 0.112 (100-300 ms) on H5 residual 0.588
    s1 = LEAD["summary"]["multisine_relaxation"]["S1"]["100-300ms"][
        "max_amplitude_ratio"
    ]
    c1 = LEAD["lpv"]["h5"]["capacity"]["S1"]["B"]
    out["S1_pythagoras"] = {
        "h5_B": c1,
        "state_share_100_300ms": s1,
        "predicted_drop_if_orthogonal": c1 - float(np.sqrt(c1**2 - s1**2)),
        "predicted_drop_if_aligned": s1,
        "measured_S1_minus_S0_causal": {
            t: e["S1_minus_S0"]
            for t, e in LEAD["summary"]["lpv_drops"]["causal"].items()
        },
    }
    return out


# ---------------------------------------------------------------- multisine


def periods(y: np.ndarray) -> np.ndarray:
    return np.asarray(y)[: 3 * FS].reshape(3, FS)


def synth(tau, name: str) -> np.ndarray:
    sys.path.insert(0, str(HERE.parent))
    import lead5_state_feedback_fit as lf  # helper (device), reused on purpose

    x = np.asarray(h.load("x", (name,))[name])
    return lf.device(np.concatenate((np.zeros(FS // 2), x)), tau)[FS // 2 :]


def ms_one(tag) -> dict:
    """Per level: mean over 4 realizations of (p1-p3)^2 per sample re mean p3 power."""
    out = {}
    for lv in ("-30", "-24", "-18"):
        d13, d23, p3 = [], [], []
        for r in range(1, 5):
            name = f"multisine_{lv}dBFS_rms_r{r}"
            if tag in ("S0", "S1"):
                y = synth(None if tag == "S0" else 0.5, name)
            else:
                y = h.load(tag, (name,))[name]
            p = periods(y)
            ref = np.mean(p[2] ** 2)
            d13.append((p[0] - p[2]) ** 2 / ref)
            d23.append((p[1] - p[2]) ** 2 / ref)
        d13, d23 = np.mean(d13, axis=0), np.mean(d23, axis=0)

        def band(lo_ms, hi_ms, d=d13):
            a, b = round(lo_ms * FS / 1000), round(hi_ms * FS / 1000)
            return pdb(float(np.mean(d[a:b])))

        fine = [band(t, t + 1) for t in range(0, 40)]
        out[lv] = {
            "p1p3_db": {
                "0-10ms": band(0, 10),
                "5.33-10ms": band(256 / 48, 10),
                "5.33-6.33ms": band(256 / 48, 256 / 48 + 1),
                "10-30ms": band(10, 30),
                "30-31ms": band(30, 31),
                "30-100ms": band(30, 100),
                "100-300ms": band(100, 300),
                "300-1000ms": band(300, 1000),
                "300-990ms": band(300, 990),
            },
            "p1p3_fine_1ms_db_0_40ms": fine,
            "p2p3_db": pdb(float(np.mean(d23))),
            "p2p3_db_without_last_10ms": pdb(float(np.mean(d23[: FS - 480]))),
            "p2p3_share_of_power_in_last_1ms": float(d23[-48:].sum() / d23.sum()),
        }
    return out


def multisine() -> dict:
    out = {}
    for t in [*HIGH, GAINS[0.0], "x", "S0", "S1"]:
        out[str(t)] = ms_one(t)
        log("multisine", t)
    # max over the four gain takes and three levels, as the lead's summary
    agg = {}
    for band in (
        "10-30ms",
        "30-100ms",
        "100-300ms",
        "300-1000ms",
        "5.33-10ms",
        "5.33-6.33ms",
        "0-10ms",
        "30-31ms",
    ):
        v = max(out[str(t)][lv]["p1p3_db"][band] for t in HIGH for lv in out[str(t)])
        agg[band] = {"max_db": v, "amplitude": 10 ** (v / 20)}
    agg["p2p3_max_db"] = max(
        out[str(t)][lv]["p2p3_db"] for t in HIGH for lv in out[str(t)]
    )
    # lead values to compare
    lead_ms = LEAD["multisine"]["tags"]
    compare = {}
    for t in [*HIGH, GAINS[0.0], "S0", "S1"]:
        for lv in ("-30", "-24", "-18"):
            for band in ("10-30ms", "30-100ms", "100-300ms", "300-1000ms"):
                compare[f"{t}/{lv}/{band}"] = {
                    "lead": lead_ms[str(t)][lv]["p1_vs_p3_db"][band],
                    "mine": out[str(t)][lv]["p1p3_db"][band],
                }
            compare[f"{t}/{lv}/p2p3"] = {
                "lead": lead_ms[str(t)][lv]["p2_vs_p3_db"],
                "mine": out[str(t)][lv]["p2p3_db"],
            }
    finite = [
        abs(v["lead"] - v["mine"])
        for v in compare.values()
        if v["lead"] > -2000 and v["mine"] > -2000
    ]
    return {
        "tags": out,
        "max_over_high_gain": agg,
        "compare_with_lead": compare,
        "max_abs_gap_db_finite": max(finite),
    }


# ---------------------------------------------------------------- clicks (192 kHz)

C1, C2, LEN, ONSET = 96_000, 192_384, 48_000, 960


def rel_db(a, b) -> float:
    return float(20 * np.log10(rms(a - b) / rms(b)))


def marker(path, which: str) -> np.ndarray:
    s = nolly.SECTIONS[which]
    return nolly.read(path, s["start"], s["start"] + s["samples"])


def clicks() -> dict:
    src = marker(nolly.SOURCE, "markers_start")
    src_e = marker(nolly.SOURCE, "markers_end")
    nz = np.flatnonzero(src)
    runs = np.split(nz, np.flatnonzero(np.diff(nz) > 1) + 1)
    source = {
        "click_runs_start_end": [[int(r[0]), int(r[-1])] for r in runs],
        "markers_end_identical_to_start": bool(np.array_equal(src, src_e)),
        "click1_equals_click2": bool(
            np.array_equal(src[C1 : C1 + LEN], src[C2 : C2 + LEN])
        ),
    }
    rows = {t["order"]: t for t in nolly.takes()}
    out = {}
    for o in (1, 7, 30, 16, 13):
        path = nolly.RENDERS / rows[o]["file"]
        s, e = marker(path, "markers_start"), marker(path, "markers_end")
        s1, s2 = s[C1 : C1 + LEN], s[C2 : C2 + LEN]
        e1 = e[C1 : C1 + LEN]
        lead = LEAD["clicks"]["takes"][str(o)]
        out[str(o)] = {
            "setting": rows[o]["setting"],
            "s2_vs_s1_total_db": rel_db(s2, s1),
            "s2_vs_s1_onset_db": rel_db(s2[:ONSET], s1[:ONSET]),
            "s2_vs_s1_tail_db": rel_db(s2[ONSET:], s1[ONSET:]),
            "e1_vs_s1_total_db": rel_db(e1, s1),
            "residue_before_click2_dbfs": float(
                20 * np.log10(rms(s[C2 - 19_200 : C2]))
            ),
            "click1_onset_dbfs": float(20 * np.log10(rms(s1[:ONSET]))),
            "diff_s2_s1_onset_dbfs": float(20 * np.log10(rms((s2 - s1)[:ONSET]))),
            "lead": {
                "s2_vs_s1_total_db": lead["s2_vs_s1"]["total_db"],
                "s2_vs_s1_onset_db": lead["s2_vs_s1"]["onset_db"],
                "s2_vs_s1_tail_db": lead["s2_vs_s1"]["tail_db"],
                "e1_vs_s1_total_db": lead["e1_vs_s1"]["total_db"],
                "residue_before_click2_dbfs": lead["start_before_click2_dbfs"],
                "click1_onset_dbfs": lead["start_click1_onset_dbfs"],
            },
        }
        log("clicks", o)
    return {"source": source, "takes": out}


# ---------------------------------------------------------------- drift (presence 10)


def drift() -> dict:
    md = json.loads(
        (h.ROOT / "diagnosis/probe_metrics/nolly_marker_drift.json").read_text()
    )
    rows = {t["order"]: t for t in nolly.takes()}
    out = {}
    for o in (9, 13, 23, 24, 38, 41, 18, 1):
        path = nolly.RENDERS / rows[o]["file"]
        s, e = marker(path, "markers_start"), marker(path, "markers_end")
        d = e - s
        w1 = slice(C1, C1 + LEN)
        w2 = slice(C2, C2 + LEN)
        # pre-click output (0.5-1 s after steps_-18, silence)
        pre = e[C1 // 2 : C1]
        # whole-section Welch exactly as nolly_marker_drift (onset inside segments)
        f, pd = welch(d, fs=nolly.RATE, nperseg=8192)
        f2, pp = welch(pre, fs=nolly.RATE, nperseg=8192)
        # onset-only spectrum: first 5 ms of the click-1 difference, zero-padded
        spec = np.abs(np.fft.rfft(d[C1 : C1 + ONSET], 8192)) ** 2
        fo = np.fft.rfftfreq(8192, 1 / nolly.RATE)
        out[str(o)] = {
            "setting": rows[o]["setting"],
            "whole_section_end_vs_start_db": float(20 * np.log10(rms(d) / rms(s))),
            "marker_drift_json_db": md[str(o)]["difference_db"],
            "share_of_difference_energy_in_click1_window": float(
                np.sum(d[w1] ** 2) / np.sum(d**2)
            ),
            "share_in_click2_window": float(np.sum(d[w2] ** 2) / np.sum(d**2)),
            "section_diff_welch_peak_hz": float(f[np.argmax(pd)]),
            "marker_drift_json_peak_hz": md[str(o)]["difference_peak_hz"],
            "click1_onset_diff_peak_hz": float(fo[np.argmax(spec)]),
            "click1_onset_diff_share_800_1600hz": float(
                spec[(fo > 800) & (fo < 1600)].sum() / spec.sum()
            ),
            "share_of_click1_diff_energy_in_first_5ms": float(
                np.sum(d[C1 : C1 + ONSET] ** 2) / np.sum(d[w1] ** 2)
            ),
            "preclick_dbfs": float(20 * np.log10(max(rms(pre), 1e-300))),
            "preclick_peak_hz": float(f2[np.argmax(pp)]),
            "preclick_share_below_100hz": float(
                pp[f2 < 100].sum() / max(pp.sum(), 1e-300)
            ),
            "click1_diff_dbfs": float(20 * np.log10(rms(d[w1]))),
            "end_click1_gain_ls": float(np.dot(e[w1], s[w1]) / np.dot(s[w1], s[w1])),
        }
        log("drift", o)
    return out


def regrowth() -> dict:
    """Output in the 1 s of digital silence between steps_-18dBFS_rms and the first
    end click: 10 ms frames, minimum, regrowth after it; and pad_end per second."""
    rows = {t["order"]: t for t in nolly.takes()}
    st = nolly.SECTIONS["steps_-18dBFS_rms"]
    a = st["start"] + st["samples"]
    b = nolly.SECTIONS["markers_end"]["start"] + C1
    src = nolly.read(nolly.SOURCE, a, b)
    pad = nolly.SECTIONS["pad_end"]
    out = {"source_max_abs_in_window": float(np.max(np.abs(src)))}
    for o in (9, 13, 23, 24, 38, 41, 18, 7, 30, 1, 40):
        y = nolly.read(nolly.RENDERS / rows[o]["file"], a, b)
        fr = y[: len(y) // 1920 * 1920].reshape(-1, 1920)
        tr = 10 * np.log10(np.maximum(np.mean(fr**2, axis=1), 1e-300))
        k = int(np.argmin(tr))
        tail = nolly.read(
            nolly.RENDERS / rows[o]["file"], pad["start"], pad["start"] + pad["samples"]
        )
        out[str(o)] = {
            "setting": rows[o]["setting"],
            "min_dbfs": float(tr[k]),
            "min_at_s": k * 1920 / nolly.RATE,
            "regrowth_db": float(tr[k:].max() - tr[k]),
            "last_100ms_dbfs": float(20 * np.log10(max(rms(y[-19_200:]), 1e-300))),
            "pad_end_per_second_dbfs": [
                float(10 * np.log10(max(np.mean(c**2), 1e-300)))
                for c in tail[: len(tail) // nolly.RATE * nolly.RATE].reshape(
                    -1, nolly.RATE
                )
            ],
            "lead_regrowth_db": LEAD["gaps"]["takes"][str(o)][
                "after_steps_regrowth_db"
            ],
        }
        log("regrowth", o)
    return out


# ---------------------------------------------------------------- taps (memory)


def h5_taps(take: int, taps: int) -> float:
    probe = h.FeatureProbe(
        lambda i: [np.asarray(i["x"]) ** k for k in range(1, 6)], taps
    )
    return h.output_capacity(probe, take)


def taps() -> dict:
    out = {}
    for o in (20, 7):
        out[str(o)] = {}
        for m in (256, 1024):
            t0 = time.time()
            out[str(o)][str(m)] = h5_taps(o, m)
            log("taps", o, m, f"{time.time() - t0:.0f} s", out[str(o)][str(m)])
    return out


# ---------------------------------------------------------------- lpv (own code)


def mpower(x: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate(([0.0], np.cumsum(x * x)))
    i = np.arange(1, len(x) + 1)
    return (c[i] - c[np.maximum(i - w, 0)]) / w


def zlevel(power: np.ndarray, a: int) -> np.ndarray:
    lv = 10 * np.log10(np.maximum(power, 1e-10))
    return (lv - lv[:a].mean()) / lv[:a].std()


def lpv() -> dict:
    xs = [np.asarray(h.load("x", (s,))[s]) for s in (h.FIT, h.VALID, h.READ)]
    a, v = len(xs[0]), len(xs[1])
    x = np.concatenate(xs)
    ys = {
        o: {s: np.asarray(h.load(o, (s,))[s]) for s in (h.FIT, h.VALID, h.READ)}
        for o in [*HIGH, 1, 2]
    }
    zs = {}
    w10 = 10 * FS // 1000
    zs["causal_10ms"] = zlevel(mpower(x, w10), a)
    zs["anticausal_10ms"] = zlevel(mpower(x[::-1], w10)[::-1], a)
    p = mpower(x, w10)
    delay = 10 * FS
    plc = np.full(len(p), float(np.median(p[:a])))
    plc[delay:] = p[: len(p) - delay]
    zs["placebo_10ms"] = zlevel(plc, a)
    zs["causal_100ms"] = zlevel(mpower(x, 100 * FS // 1000), a)
    zs["h5"] = None
    probe_taps = 256
    out = {}
    for label, z in zs.items():
        t0 = time.time()
        parts = []
        for k in range(1, 6):
            for j in range(1 if z is None else 3):
                parts.append(x**k if z is None else x**k * z**j)
        scales = [rms(f[:a]) for f in parts]
        ua = [f[:a] / s for f, s in zip(parts, scales, strict=True)]
        probe = Probe(degree=len(ua), taps=probe_taps, ridge=1e-4)
        gram, _ = normal_equations(probe, ua, None)
        factor = cho_factor(regularized(probe, gram))
        size = len(ua[0])
        from scipy.fft import irfft, next_fast_len, rfft

        n_fft = next_fast_len(size + probe_taps)
        spectra = [rfft(u, n_fft) for u in ua]

        def solve(y):
            ty = rfft(y, n_fft)
            rhs = [np.array([np.sum(y)])] + [
                irfft(np.conj(sk) * ty, n_fft)[:probe_taps] for sk in spectra
            ]
            return cho_solve(factor, np.concatenate(rhs))

        def predict(theta, lo, hi):
            q = np.full(hi - lo, theta[0])
            for kk, (f, s) in enumerate(zip(parts, scales, strict=True)):
                hh = theta[1 + kk * probe_taps : 1 + (kk + 1) * probe_taps]
                q += fftconvolve(f[lo:hi] / s, hh)[: hi - lo]
            return q

        entry = {"capacity": {}}
        for o in HIGH:
            th = solve(ys[o][h.FIT])
            entry["capacity"][str(o)] = {
                "V": h.ratio(ys[o][h.VALID], predict(th, a, a + v)),
                "B": h.ratio(ys[o][h.READ], predict(th, a + v, len(x))),
            }
        q1 = predict(solve(ys[1][h.FIT]), a + v, len(x))
        q2 = predict(solve(ys[2][h.FIT]), a + v, len(x))
        qd = predict(solve(ys[2][h.FIT] - ys[1][h.FIT]), a + v, len(x))
        entry["linearity_gap_pair_1_2"] = rms((q2 - q1 - qd)[h.SKIP :]) / rms(
            qd[h.SKIP :]
        )
        lead_key = {
            "causal_10ms": ("causal", "10ms"),
            "anticausal_10ms": ("anticausal", "10ms"),
            "placebo_10ms": ("placebo", "10ms"),
            "causal_100ms": ("causal", "100ms"),
        }
        if label == "h5":
            lead_cap = LEAD["lpv"]["h5"]["capacity"]
        else:
            fam, tau = lead_key[label]
            lead_cap = LEAD["lpv"]["families"][fam][tau]["capacity"]
        entry["lead"] = {str(o): lead_cap[str(o)] for o in HIGH}
        entry["max_abs_gap"] = max(
            abs(entry["capacity"][str(o)][s] - lead_cap[str(o)][s])
            for o in HIGH
            for s in ("V", "B")
        )
        out[label] = entry
        log("lpv", label, f"{time.time() - t0:.0f} s", entry["max_abs_gap"])
        del gram, factor, spectra, ua, parts
    return out


# ---------------------------------------------------------------- levels


def levels() -> dict:
    out = {}
    for sec in (h.FIT, h.READ):
        xb = np.asarray(h.load("x", (sec,))[sec])
        n = len(xb) // 2400
        lv = 10 * np.log10(
            np.maximum(np.mean(xb[: n * 2400].reshape(n, 2400) ** 2, axis=1), 1e-30)
        )
        out[sec] = {
            "quantiles_dbfs_5_25_50_75_95_99": [
                float(q) for q in np.percentile(lv, [5, 25, 50, 75, 95, 99])
            ],
            "share_windows_above_-18": float(np.mean(lv > -18)),
            "share_windows_above_-30": float(np.mean(lv > -30)),
            "share_windows_below_-50": float(np.mean(lv < -50)),
            "overall_rms_dbfs": float(20 * np.log10(rms(xb))),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args()
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    for st in args.stages:
        t0 = time.time()
        result[st] = globals()[st]()
        if isinstance(result[st], dict):
            result[st]["seconds"] = round(time.time() - t0, 1)
        OUT.write_text(json.dumps(nr.finite(result), indent=1) + "\n")
        log("stage", st, "written")


if __name__ == "__main__":
    main()
