#!/usr/bin/env python3
"""Skeptic check of lead 1 (output-referenced probe), CPU only, read-only on the lead.

Recomputed here without the lead's functions (features built inline, harness
primitives and cached arrays only):
- nolly: B (and V) shares of FIR 256 on y_ref for 3 tone, 2 gain and 1 cab pairs
  (harness.difference_share and a manual fit read on V and B), FIR 1024 on the cab
  pair 39->42, x5_yh5 on 39->30, frozen H5 shares from the cached predictions;
- identity: an independent direct least-squares solver (lagged design matrix built in
  chunks, no FFT Gram) fits y_ref, y_eval and d with FIR 256 on y_ref, ref take 1:
  r_eval / unrepresented, r_ref / RMS(y_ref), coverage RMS(r_eval - r_ref) / RMS(d);
- linearity: harness.linearity_check of two variants the lead did not measure
  (FIR 256 on y_ref advanced by 16 samples, y_ref with tanh features);
- json: class medians, V selection, instrument agreement, identity baseline of the
  conditional capacity (q = y_ref), recomputed from the lead's per-pair JSON;
- lags: VS100R model renders cached by the lead (no inference), on di_validation and
  di_test: ESR-optimal fractional delay (phase ramp on the zero-padded model output and
  the bench's scored window, as coreading_delay.py), cross-correlation peak with
  parabolic and band-limited refinement, and the cross-spectrum phase slope on
  80 Hz - 2 kHz and 2 - 6 kHz; sign tested on a synthetic +2.0-sample shift.

    uv run python diagnosis/probe_capacity/verify/lead1_verify.py --stage json
    (stages: json, nolly, identity, linearity, lags; each updates lead1_verify.json)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import next_fast_len
from scipy.optimize import minimize_scalar
from scipy.signal import csd, fftconvolve, welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as h  # noqa: E402

OUT = HERE / "lead1_verify.json"
LEAD = json.loads((HERE.parent / "lead1_output_referenced.json").read_text())
VS = h.CACHE / "lead1" / "vs100r"
RIDGE = 1e-4
FS = h.FS
med = statistics.median


def rms(v) -> float:
    return float(np.sqrt(np.mean(np.asarray(v) ** 2)))


def share(d: np.ndarray, q: np.ndarray) -> float:
    """RMS((d - q)[SKIP:]) / RMS(d[SKIP:]), written here (no floor)."""
    return rms((d - q)[h.SKIP :]) / rms(d[h.SKIP :])


def lead_pair(ref: int, ev: int) -> dict:
    (row,) = [
        p
        for p in LEAD["nolly"]["pairs"]
        if (p["ref"], p["eval"]) == (ref, ev) and not p["kind"].endswith("reversed")
    ]
    return row


# ---------------------------------------------------------------- stage json


def my_class(p: dict) -> str:
    if p["kind"] in ("repeat", "cab"):
        return p["kind"]
    g = float(h.ROWS[p["ref"]]["G"]) != float(h.ROWS[p["eval"]]["G"])
    return "gain" if g else "tone"


def stage_json() -> dict:
    pairs = LEAD["nolly"]["pairs"]
    out: dict = {}
    # class medians, own classification; reversed tested BEFORE excluded (fixed group)
    groups: dict[str, list] = {}
    lead_groups: dict[str, list] = {}
    for p in pairs:
        c = my_class(p)
        assert c == p["class"], (p["ref"], p["eval"], c, p["class"])
        rev = p["kind"] == "one_knob_reversed"
        key = f"{c}_reversed" if rev else c
        if p["excluded"]:
            key += "_gate4_excluded"
        groups.setdefault(key, []).append(p)
        lk = f"{c}_gate4_excluded" if p["excluded"] else (f"{c}_reversed" if rev else c)
        lead_groups.setdefault(lk, []).append(p)
    variants = ("yfir", "yfir1024", "yh5_1024", "x5_yh5")
    rows = {}
    max_gap = 0.0
    for key, members in sorted(lead_groups.items()):
        mine = {v: med([m["share"][v][h.READ] for m in members]) for v in variants}
        mine["H5_x"] = med([m["frozen_share_B"]["hammerstein_5"] for m in members])
        lead = LEAD["nolly"]["by_class"][key]
        theirs = {v: lead["median_share_B"][v] for v in variants}
        theirs["H5_x"] = lead["median_frozen_share_B"]["hammerstein_5"]
        gap = max(abs(mine[k] - theirs[k]) for k in mine)
        max_gap = max(max_gap, gap)
        rows[key] = {"n": len(members), "recomputed": mine, "max_abs_gap": gap}
    out["class_medians_from_pairs"] = {"by_lead_group": rows, "max_abs_gap": max_gap}
    out["grouping_note"] = {
        "lead_tone_gate4_excluded": sorted(
            f"{p['ref']}->{p['eval']}" for p in lead_groups["tone_gate4_excluded"]
        ),
        "reversed_pairs_in_gate4_group": [
            f"{p['ref']}->{p['eval']}"
            for p in lead_groups["tone_gate4_excluded"]
            if p["kind"] == "one_knob_reversed"
        ],
        "tone_gate4_excluded_forward_only_median": {
            v: med(
                [m["share"][v][h.READ] for m in groups["tone_gate4_excluded"]]
            )
            for v in variants
        },
    }
    # selection on V
    pool = [
        p
        for p in pairs
        if not p["excluded"] and p["class"] != "repeat"
        and p["kind"] != "one_knob_reversed"
    ]
    names = list(LEAD["variants"])
    med_v = {v: med([p["share"][v][h.VALID] for p in pool]) for v in names}
    med_b = {v: med([p["share"][v][h.READ] for p in pool]) for v in names}
    out["selection"] = {
        "pool_n": len(pool),
        "median_V": med_v,
        "argmin_V": min(med_v, key=med_v.get),
        "lead_selected": LEAD["nolly"]["selection"]["selected"],
        "max_abs_gap_median_V": max(
            abs(med_v[v] - LEAD["nolly"]["selection"]["median_V"][v]) for v in names
        ),
        "median_B_same_pool_for_reference": med_b,
        "argmin_B_same_pool": min(med_b, key=med_b.get),
        "rank_V": sorted(names, key=med_v.get),
        "rank_B": sorted(names, key=med_b.get),
    }
    # instrument: frozen shares from cache vs nolly_reading.json
    gaps = [
        abs(p["frozen_share_B"][n] - p["frozen_share_reading_B"][n])
        for p in pairs
        if p["frozen_share_reading_B"]
        for n in p["frozen_share_B"]
    ]
    out["instrument_frozen_vs_reading"] = {"n": len(gaps), "max_abs_gap": max(gaps)}
    # trap: conditional capacity against its identity baseline q = y_ref
    trap = {}
    for key in ("tone", "gain", "cab", "gain_reversed", "tone_reversed"):
        members = lead_groups[key]
        trap[key] = {
            "n": len(members),
            "median_identity_baseline_RMSd_over_RMSyeval": med(
                [m["eval_rel_B"] for m in members]
            ),
            "median_conditional_yfir": med(
                [m["conditional_capacity_B"]["yfir"] for m in members]
            ),
            "median_conditional_x5_yh5": med(
                [m["conditional_capacity_B"]["x5_yh5"] for m in members]
            ),
            "median_frozen_H5_output_capacity_eval": med(
                [m["frozen_capacity_eval_B"]["hammerstein_5"] for m in members]
            ),
            "median_share_x5_yh5": med([m["share"]["x5_yh5"][h.READ] for m in members]),
        }
    ladder = {}
    for g, row in LEAD["nolly"]["conditional"]["gain_ladder"].items():
        ref, ev = (int(t) for t in row["pair"].split("->"))
        p = lead_pair(ref, ev)
        ladder[g] = {
            "pair": row["pair"],
            "identity_baseline": p["eval_rel_B"],
            "conditional_x5_yh5": p["conditional_capacity_B"]["x5_yh5"],
            "conditional_yfir": p["conditional_capacity_B"]["yfir"],
            "frozen_H5": row["frozen_capacity_eval_B"]["hammerstein_5"],
            "share_x5_yh5": p["share"]["x5_yh5"][h.READ],
            "share_yfir": p["share"]["yfir"][h.READ],
            "below_0p3_any_variant": any(
                v < 0.3 for v in p["conditional_capacity_B"].values()
            ),
            "min_conditional_any_variant": min(p["conditional_capacity_B"].values()),
        }
    out["trap_conditional_capacity"] = {"by_class": trap, "gain_ladder": ladder}
    # lag medians per run from the lead's JSON
    by_run = LEAD["vs100r"]["lag_V"]["by_run"]
    out["lead_lag_V_per_run"] = {
        r: {
            "median": med(v.values()),
            "mean": float(np.mean(list(v.values()))),
            "values": v,
        }
        for r, v in by_run.items()
    }
    return out


# ---------------------------------------------------------------- stage nolly


def feats_yfir(i):
    return [np.asarray(i["y_ref"])]


def feats_x5_yh5(i):
    x, y = np.asarray(i["x"]), np.asarray(i["y_ref"])
    return [x**k for k in range(1, 6)] + [y**k for k in range(1, 6)]


def fit_v_b(features, taps: int, ref: int, ev: int) -> dict:
    t0 = time.time()
    d = {s: h.difference(ref, ev, s) for s in (h.FIT, h.VALID, h.READ)}
    probe = h.FeatureProbe(features, taps=taps, ridge=RIDGE)
    probe.fit(h.section_inputs(h.FIT, ref), d[h.FIT])
    out = {
        s: share(d[s], probe.predict(h.section_inputs(s, ref)))
        for s in (h.VALID, h.READ)
    }
    out["seconds"] = time.time() - t0
    return out


NOLLY_YFIR = {
    "tone": [(1, 3), (20, 22), (39, 33)],
    "gain": [(1, 7), (20, 15)],
    "cab": [(32, 40)],
}


def stage_nolly() -> dict:
    out = {"yfir": [], "other": []}
    for cls, plist in NOLLY_YFIR.items():
        for ref, ev in plist:
            mine = fit_v_b(feats_yfir, 256, ref, ev)
            via_harness = h.difference_share(h.FeatureProbe(feats_yfir), ref, ev)
            lead = lead_pair(ref, ev)["share"]["yfir"]
            row = {
                "pair": f"{ref}->{ev}",
                "class": cls,
                "B_difference_share": via_harness,
                "B_manual": mine[h.READ],
                "V_manual": mine[h.VALID],
                "lead_B": lead[h.READ],
                "lead_V": lead[h.VALID],
                "abs_gap_B": abs(via_harness - lead[h.READ]),
                "abs_gap_V": abs(mine[h.VALID] - lead[h.VALID]),
            }
            out["yfir"].append(row)
            print(row, flush=True)
    for name, feats, taps, (ref, ev) in (
        ("yfir1024", feats_yfir, 1024, (39, 42)),
        ("x5_yh5", feats_x5_yh5, 256, (39, 30)),
    ):
        mine = fit_v_b(feats, taps, ref, ev)
        lead = lead_pair(ref, ev)["share"][name]
        row = {
            "variant": name,
            "pair": f"{ref}->{ev}",
            "B": mine[h.READ],
            "V": mine[h.VALID],
            "lead_B": lead[h.READ],
            "lead_V": lead[h.VALID],
            "abs_gap_B": abs(mine[h.READ] - lead[h.READ]),
            "seconds": mine["seconds"],
        }
        out["other"].append(row)
        print(row, flush=True)
    # frozen H5 share from x, cached frozen predictions, own subtraction
    out["frozen_H5_x"] = []
    for ref, ev in ((1, 3), (39, 30), (1, 7)):
        q = np.asarray(h.frozen_prediction("hammerstein_5", ev)) - np.asarray(
            h.frozen_prediction("hammerstein_5", ref)
        )
        s = share(h.difference(ref, ev, h.READ), q)
        out["frozen_H5_x"].append(
            {
                "pair": f"{ref}->{ev}",
                "share": s,
                "lead": lead_pair(ref, ev)["frozen_share_B"]["hammerstein_5"],
            }
        )
    return out


# ---------------------------------------------------------------- stage identity


def direct_fir_fit(u: np.ndarray, targets: np.ndarray, taps: int, chunk: int = 40_000):
    """Exact ridge least squares of targets (n, k) on [1, u[n], ..., u[n-taps+1]]
    (zeros before the start), lagged design matrix built in chunks; relative ridge on
    the FIR columns as the library (mean of their Gram diagonal)."""
    n = len(u)
    upad = np.concatenate([np.zeros(taps - 1), u])
    gram = np.zeros((taps + 1, taps + 1))
    rhs = np.zeros((taps + 1, targets.shape[1]))
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        lagged = np.ascontiguousarray(sliding_window_view(upad[a : b + taps - 1], taps)[:, ::-1])
        gram[1:, 1:] += lagged.T @ lagged
        sums = lagged.sum(axis=0)
        gram[0, 1:] += sums
        rhs[1:] += lagged.T @ targets[a:b]
        rhs[0] += targets[a:b].sum(axis=0)
    gram[1:, 0] = gram[0, 1:]
    gram[0, 0] = n
    gram[1:, 1:] += RIDGE * np.mean(np.diag(gram)[1:]) * np.eye(taps)
    return np.linalg.solve(gram, rhs)


def direct_predict(theta: np.ndarray, u: np.ndarray) -> np.ndarray:
    return theta[0] + fftconvolve(u, theta[1:])[: len(u)]


def stage_identity() -> dict:
    """FIR 256 on y_ref, ref take 1, evals 4, 5, 7: one Gram, targets y_1, y_e, d."""
    t0 = time.time()
    ref, evals = 1, (4, 5, 7)
    a = {t: np.asarray(h.load(t, (h.FIT,))[h.FIT]) for t in (ref, *evals)}
    b = {t: np.asarray(h.load(t, (h.READ,))[h.READ]) for t in (ref, *evals)}
    scale = rms(a[ref])
    cols = [a[ref]] + [a[e] for e in evals] + [a[e] - a[ref] for e in evals]
    theta = direct_fir_fit(a[ref] / scale, np.stack(cols, axis=1), 256)
    q = [direct_predict(theta[:, j], b[ref] / scale) for j in range(theta.shape[1])]
    keep = slice(h.SKIP, None)
    q_ref = q[0]
    r_ref = b[ref] - q_ref
    out = {"ref": ref, "r_ref_rel": rms(r_ref[keep]) / rms(b[ref][keep]), "pairs": []}
    for k, e in enumerate(evals):
        q_eval, q_d = q[1 + k], q[1 + len(evals) + k]
        d = b[e] - b[ref]
        r_eval = b[e] - q_eval
        unrep = rms((d - q_d)[keep])
        lead = lead_pair(ref, e)
        check = [
            c
            for c in LEAD["checks"]["rows"]
            if c["variant"] == "yfir" and c["pair"] == f"{ref}->{e}"
        ]
        out["pairs"].append(
            {
                "pair": f"{ref}->{e}",
                "share_B_direct_solver": unrep / rms(d[keep]),
                "lead_share_B": lead["share"]["yfir"][h.READ],
                "r_eval_over_unrepresented": rms(r_eval[keep]) / unrep,
                "lead_r_eval_over_unrepresented": check[0][
                    "r_eval_over_unrepresented"
                ]
                if check
                else None,
                "coverage_RMS_reval_minus_rref_over_RMSd": rms((r_eval - r_ref)[keep])
                / rms(d[keep]),
                "linearity_gap_direct": rms((q_eval - q_ref - q_d)[keep])
                / rms(q_d[keep]),
                "output_capacity_eval_with_yref_feature": rms(r_eval[keep])
                / rms(b[e][keep]),
                "identity_baseline_RMSd_over_RMSyeval": rms(d[keep]) / rms(b[e][keep]),
            }
        )
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------- stage linearity


def stage_linearity() -> dict:
    def lead16(i):
        y = np.asarray(i["y_ref"])
        return [np.concatenate([y[16:], np.zeros(16)])]

    def sat(i):
        y = np.asarray(i["y_ref"])
        return [y] + [np.tanh(a * y) for a in (3.0, 10.0, 30.0)]

    out = {}
    for name, feats in (("yfir_lead16", lead16), ("ysat", sat)):
        t0 = time.time()
        gap = h.linearity_check(h.FeatureProbe(feats, ridge=RIDGE), 1, 2)
        out[name] = {"pair": "1->2", "gap": gap, "seconds": time.time() - t0}
        print(name, out[name], flush=True)
    return out


# ---------------------------------------------------------------- stage lags

PAD = 4_800


def shifted(x: np.ndarray, d: float) -> np.ndarray:
    """x delayed by d samples (fractional, phase ramp on the zero-padded signal)."""
    n = len(x) + 2 * PAD
    spec = np.fft.rfft(np.pad(x, PAD))
    return np.fft.irfft(spec * np.exp(-2j * np.pi * np.fft.rfftfreq(n) * d), n)[
        PAD:-PAD
    ]


class EsrCurve:
    """Bench ESR (scored window 0.5 s .. -0.1 s, no pre-emphasis) of the model
    delayed by d against the device."""

    def __init__(self, m: np.ndarray, y: np.ndarray) -> None:
        self.n = len(m) + 2 * PAD
        self.spec = np.fft.rfft(np.pad(m, PAD))
        self.freqs = np.fft.rfftfreq(self.n)
        self.y = y[FS // 2 : -FS // 10]

    def __call__(self, d: float) -> float:
        m = np.fft.irfft(self.spec * np.exp(-2j * np.pi * self.freqs * d), self.n)
        m = m[PAD:-PAD][FS // 2 : -FS // 10]
        return float(np.sum((self.y - m) ** 2) / np.sum(self.y**2))


def xcorr_spectrum(a: np.ndarray, b: np.ndarray):
    size = next_fast_len(2 * len(a))
    s = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    return s, size


def lag_parabolic(c: np.ndarray, span: int = 32) -> tuple[int, float]:
    w = np.concatenate([c[-span:], c[: span + 1]])
    k = int(np.argmax(w))
    lo, mid, hi = w[k - 1], w[k], w[k + 1]
    return k - span, float(k - span + 0.5 * (lo - hi) / (lo - 2 * mid + hi))


def lag_bandlimited(s: np.ndarray, size: int, k: int) -> float:
    f = np.arange(len(s))
    w = np.full(len(s), 2.0)
    w[0] = 1.0
    if size % 2 == 0:
        w[-1] = 1.0

    def neg(t):
        return -float(np.sum(w * np.real(s * np.exp(2j * np.pi * f * t / size))))

    r = minimize_scalar(neg, bounds=(k - 1.0, k + 1.0), method="bounded",
                        options={"xatol": 1e-4})
    return float(r.x)


def phase_slope(m: np.ndarray, y: np.ndarray, band: tuple[float, float]) -> dict:
    """Delay of m against y (positive: m late) from the Welch cross-spectrum phase,
    coherence-weighted least squares, through the origin and with an intercept."""
    f, pxy = csd(y, m, fs=FS, nperseg=8192)  # conj(Y) M
    _, pyy = welch(y, fs=FS, nperseg=8192)
    _, pmm = welch(m, fs=FS, nperseg=8192)
    coh = np.abs(pxy) ** 2 / (pyy * pmm)
    sel = (f >= band[0]) & (f <= band[1])
    phi = np.unwrap(np.angle(pxy[sel]))
    om = 2 * np.pi * f[sel] / FS
    w = np.clip(coh[sel], 0, 0.999) / (1 - np.clip(coh[sel], 0, 0.999))
    origin = -float(np.sum(w * om * phi) / np.sum(w * om**2))
    coef = np.polyfit(om, phi, 1, w=np.sqrt(w))
    return {
        "delay_origin": origin,
        "delay_with_intercept": -float(coef[0]),
        "intercept_rad": float(coef[1]),
        "median_coherence": float(np.median(coh[sel])),
    }


def lag_all(m: np.ndarray, y: np.ndarray) -> dict:
    s, size = xcorr_spectrum(m, y)
    c = np.fft.irfft(s, size)
    k, par = lag_parabolic(c)
    band = lag_bandlimited(s, size, k)
    curve = EsrCurve(m, y)
    r = minimize_scalar(curve, bounds=(-band - 1.5, -band + 1.5), method="bounded",
                        options={"xatol": 1e-3})
    esr0 = curve(0.0)
    return {
        "xcorr_integer": k,
        "xcorr_parabolic": par,
        "xcorr_bandlimited": band,
        "esr_optimal_model_lag": -float(r.x),
        "esr_raw": esr0,
        "esr_aligned": float(r.fun),
        "error_energy_removed_by_delay": 1 - float(r.fun) / esr0,
        "phase_80_2000": phase_slope(m, y, (80.0, 2000.0)),
        "phase_2000_6000": phase_slope(m, y, (2000.0, 6000.0)),
    }


def stage_lags() -> dict:
    t0 = time.time()
    y = np.load(VS / "device" / f"take_03_{h.VALID}_y.npy")
    synthetic = {d: lag_all(shifted(y, d), y) for d in (2.0, -0.95)}
    print("synthetic", json.dumps(synthetic)[:800], flush=True)
    runs = [r for r, v in LEAD["vs100r"]["runs"].items() if not v.get("absent")]
    takes = sorted(int(t) for t in LEAD["vs100r"]["takes"])
    rows = []
    for run in runs:
        arm = LEAD["vs100r"]["runs"][run]["arm"]
        sections = (h.VALID, h.READ) if arm in ("C", "F3") else (h.VALID,)
        for take in takes:
            for s in sections:
                dev = np.load(VS / "device" / f"take_{take:02d}_{s}_y.npy")
                mod = np.load(VS / run / f"take_{take:02d}_{s}.npy")
                row = {"run": run, "arm": arm, "take": take, "section": s}
                row |= lag_all(mod, dev)
                if s == h.VALID:
                    row["lead_parabolic"] = LEAD["vs100r"]["lag_V"]["by_run"][run][
                        str(take)
                    ]
                rows.append(row)
                print(run, take, s, {k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in row.items()
                                     if k.startswith(("xcorr", "esr_opt", "lead"))},
                      round(row["phase_80_2000"]["delay_origin"], 3), flush=True)
    summary = {}
    for run in runs:
        for s in (h.VALID, h.READ):
            mine = [r for r in rows if r["run"] == run and r["section"] == s]
            if not mine:
                continue
            summary[f"{run}|{s}"] = {
                "median_xcorr_parabolic": med([r["xcorr_parabolic"] for r in mine]),
                "median_xcorr_bandlimited": med([r["xcorr_bandlimited"] for r in mine]),
                "median_esr_optimal_model_lag": med(
                    [r["esr_optimal_model_lag"] for r in mine]
                ),
                "median_phase_80_2000": med(
                    [r["phase_80_2000"]["delay_origin"] for r in mine]
                ),
                "median_phase_80_2000_intercept": med(
                    [r["phase_80_2000"]["delay_with_intercept"] for r in mine]
                ),
                "median_phase_2000_6000": med(
                    [r["phase_2000_6000"]["delay_origin"] for r in mine]
                ),
                "median_error_energy_removed": med(
                    [r["error_energy_removed_by_delay"] for r in mine]
                ),
            }
    by_arm = {}
    for arm in ("C", "GREY", "L", "F3"):
        mine = [r for r in rows if r["arm"] == arm and r["section"] == h.VALID]
        by_arm[arm] = {
            "median_xcorr_parabolic_V": med([r["xcorr_parabolic"] for r in mine]),
            "median_xcorr_bandlimited_V": med([r["xcorr_bandlimited"] for r in mine]),
            "median_esr_optimal_model_lag_V": med(
                [r["esr_optimal_model_lag"] for r in mine]
            ),
            "median_phase_80_2000_V": med(
                [r["phase_80_2000"]["delay_origin"] for r in mine]
            ),
            "lead_median_V": LEAD["vs100r"]["lag_V"]["median_by_arm"][arm],
        }
    return {
        "convention": "samples at 48 kHz, positive = model late; ESR-optimal lag = "
        "minus the delay that, applied to the model output, minimises the bench ESR",
        "synthetic_device_shift": synthetic,
        "rows": rows,
        "summary_by_run_section": summary,
        "by_arm_V": by_arm,
        "seconds": time.time() - t0,
    }


# demo/train_length/coreading_delay.py: DELAYS = arange(-2.0, 3.0001, 0.05), the delay
# applied to the model output; a model late by D needs tau = -D.
BENCH_GRID = (-2.0, 3.0)


def stage_common_tau() -> dict:
    """The co-reading rule (tau minimising the median bench ESR over takes on
    di_validation) on the 4 cached takes, F3 only (reduced for CPU time): grid step
    0.05 within +-1 sample of minus the lead's median lag, plus the bench grid edges."""
    takes = sorted(int(t) for t in LEAD["vs100r"]["takes"])
    out = {}
    for run, v in LEAD["vs100r"]["runs"].items():
        if v.get("absent") or v["arm"] != "F3":
            continue
        centre = -round(med(LEAD["vs100r"]["lag_V"]["by_run"][run].values()) * 20) / 20
        grid = np.round(np.arange(centre - 1.0, centre + 1.0001, 0.05), 2)
        grid = np.unique(np.concatenate([grid, BENCH_GRID, [0.0]]))
        curves = []
        for take in takes:
            dev = np.load(VS / "device" / f"take_{take:02d}_{h.VALID}_y.npy")
            mod = np.load(VS / run / f"take_{take:02d}_{h.VALID}.npy")
            curve = EsrCurve(mod, dev)
            curves.append([curve(float(d)) for d in grid])
        medians = np.median(np.array(curves), axis=0)
        best = int(np.argmin(medians))
        inside = (grid >= BENCH_GRID[0]) & (grid <= BENCH_GRID[1])
        best_inside = int(np.flatnonzero(inside)[np.argmin(medians[inside])])
        out[run] = {
            "tau_extended_grid": float(grid[best]),
            "median_esr_at_tau": float(medians[best]),
            "tau_bench_grid": float(grid[best_inside]),
            "median_esr_at_bench_tau": float(medians[best_inside]),
            "median_esr_raw": float(medians[int(np.argmin(np.abs(grid)))]),
            "outside_bench_grid": not BENCH_GRID[0] <= grid[best] <= BENCH_GRID[1],
        }
        print(run, out[run], flush=True)
    return {"bench_grid": BENCH_GRID, "takes": takes, "runs": out}


STAGES = {
    "json": stage_json,
    "nolly": stage_nolly,
    "identity": stage_identity,
    "linearity": stage_linearity,
    "lags": stage_lags,
    "common_tau": stage_common_tau,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    args = parser.parse_args()
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    t0 = time.time()
    result[args.stage] = STAGES[args.stage]()
    result[args.stage]["stage_seconds"] = time.time() - t0
    partial = OUT.with_name(f".{OUT.name}.partial")
    partial.write_text(json.dumps(h.nr.finite(result), indent=1) + "\n")
    partial.replace(OUT)
    print(f"wrote {OUT.relative_to(h.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
