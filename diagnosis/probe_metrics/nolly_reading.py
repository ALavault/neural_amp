#!/usr/bin/env python3
"""Pre-registered reading of the Nolly renders (NOLLY_READING.md), frozen before any
render exists. No neural model. CPU.

    nolly_reading.py RENDERS [--out FILE] [--jobs N]

RENDERS holds the renders named as in demo/capture/v2/settings.csv, plus
loopback_software.wav and center_stereo.wav. The input x is the source file itself
(software render, offline; Reaper compensates the plugin latency), common to every
render. Gates first; the reading stops where NOLLY_READING.md says so. Probes, ESR,
magnitude, phase, envelope and E2 at 48 kHz (each section decimated by 4); harmonic
signature on the synchronized sweeps at 192 kHz (nfft 32768) and with the frozen
48 kHz protocol; multisines at 192 kHz. Writes nolly_reading.json and, next to it,
the per-take best linear approximations of the multisines (.npz).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, csd, fftconvolve, resample_poly, welch

from fssr_nam.metrics.spectral import magnitude_error, phase_error
from fssr_nam.metrics.time import error_to_signal_ratio
from fssr_nam.metrics.transient import envelope_error
from fssr_nam.product.dynamics import dynamic_deviation
from fssr_nam.product.fidelity import (
    HarmonicProtocol,
    Protocol,
    analyze_pair,
    analyze_sweep_pair,
)
from fssr_nam.product.hammerstein import Sweep

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import frozen_probes  # noqa: E402

ROOT = HERE.parents[1]
CAPTURE = ROOT / "demo/capture/v2"
RATE, FS, DOWN = 192_000, 48_000, 4
LSB = 2.0**-23  # one step of the 24-bit source
ESTIMATION, READ = ("di_train",), ("di_test", "pink_noise_-30dBFS_rms")
SKIP = round(0.05 * FS)
# Harmonic band in output (harmonic) frequency; 180 Hz keeps the fundamental of order 9
# inside the 20 Hz sweep start. The 0.1 s window is 19 200 samples at 192 kHz < nfft.
HARMONIC_192 = HarmonicProtocol(order=9, nfft=32_768, band_hz=(180.0, 8_000.0))
HARMONIC_48 = HarmonicProtocol()  # frozen protocol of the VS100R reading
ANCHORS = (1, 20, 39)
CAB = {40: 32, 41: 38, 42: 39}  # cab-on take -> same setting, cab off
CONTAINERS, SUBTYPES = ("WAV", "WAVEX", "RF64", "W64"), ("FLOAT", "DOUBLE")
MARKER_SPAN = 4_096  # lag search, samples at 192 kHz
MARKER_RESPONSE_DB = 20.0  # markers RMS above the silence-section RMS of the output
MAX_LATENCY = 1_024  # median marker lag beyond the probe memory (5.3 ms): stop
MUTED_DB = -100.0  # a quiet step segment below this RMS is muted (gate left on)
WIENER_SEGMENT = 2**17  # cab test: Welch segment at 48 kHz (2.7 s), two-sided FIR
CAB_SCOPE_S = 0.25  # reported: share of the filter's energy beyond 0.25 s
DELAYS = np.round(np.arange(-3.0, 3.0001, 0.05), 2)
LADDER_METRICS = (
    ("esr", "di_test"),
    ("magnitude", "di_test"),
    ("phase", "di_test"),
    ("hammerstein_3", "di_test"),
    ("harmonic_192", "sweep_sync_-10dBFS"),
)
MONOTONE_MIN, MONOTONE_MAX_MISSES = 45, 5  # of 50 signs (5 knobs x 2 sides x 5)
CAPACITY = 0.3  # Protocol.capacity_warning, frozen; "above" means strictly greater
CAPACITY_PROBES = ("hammerstein_5", "volterra_laguerre_3")
E2_NONZERO_DB = 1e-3  # an E2 score above this is a response (bench floor ~6e-5 dB)
CACHE = HERE / ".cache_nolly"
# H3: worst cab residual for "linear" (about 3 x the stand-in floor, NOLLY_READING.md)
CAB_LINEAR = 0.02

LAYOUT = json.loads((CAPTURE / "layout.json").read_text())
SECTIONS = {s["section"]: s for s in LAYOUT["sections"]}
SWEEPS = [n for n in SECTIONS if n.startswith("sweep_sync_")]
STEPS = [n for n in SECTIONS if n.startswith("steps_")]
MULTISINES = [n for n in SECTIONS if n.startswith("multisine_")]
NEEDED = (*ESTIMATION, *READ, *SWEEPS, *STEPS)
SOURCE: dict = {}  # source sections, set in main before the worker processes fork


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(v**2)))


def db(v: float) -> float:
    return float(20 * np.log10(max(v, 1e-300)))


def cut(signal: np.ndarray, name: str) -> np.ndarray:
    s = SECTIONS[name]
    return signal[s["start"] : s["start"] + s["samples"]]


def read_sections(path: Path, names) -> dict:
    out = {}
    with sf.SoundFile(path) as f:
        for name in names:
            s = SECTIONS[name]
            f.seek(s["start"])
            out[name] = f.read(s["samples"], dtype="float64")
    return out


def dec(v: np.ndarray) -> np.ndarray:
    return resample_poly(v, 1, DOWN)


def takes() -> dict:
    with (CAPTURE / "settings.csv").open() as f:
        return {int(r["order"]): r for r in csv.DictReader(f)}


def nearest_anchor(order: int) -> int:
    return min(ANCHORS, key=lambda a: (abs(a - order), a))


def pairs(rows: dict) -> list[tuple[int, int, str]]:
    """(reference take, evaluated take, kind), all fixed before the renders."""
    out = [(1, 20, "repeat"), (20, 39, "repeat"), (1, 39, "repeat")]
    for order, row in rows.items():
        if row["block"] in ("one_knob", "combination_test", "combination_train"):
            out.append((nearest_anchor(order), order, row["block"]))
    out += [(off, on, "cab") for on, off in CAB.items()]
    return out


def wav_format(path: Path, channels: int = 1) -> dict:
    """Render format: WAV container, float, 192 kHz, `channels`, not empty. A file that
    soundfile cannot open fails the gate instead of stopping the reading."""
    try:
        info = sf.info(path)
    except (sf.LibsndfileError, RuntimeError) as error:
        return {"error": str(error), "ok": False}
    ok = (
        info.format in CONTAINERS
        and info.subtype in SUBTYPES
        and info.samplerate == RATE
        and info.channels == channels
        and info.frames > 0
    )
    return {
        "container": info.format,
        "subtype": info.subtype,
        "rate": info.samplerate,
        "channels": info.channels,
        "ok": ok,
    }


def input_gates(renders: Path, x: np.ndarray) -> dict:
    """Software loopback equal to the source within one 24-bit step, nothing after it;
    stereo L == R. Both files in the render format."""
    out = {}
    fmt = wav_format(renders / "loopback_software.wav")
    diff = None
    if fmt["ok"]:
        loop = sf.read(renders / "loopback_software.wav", dtype="float64")[0]
        if len(loop) >= len(x) and np.all(np.isfinite(loop)):
            diff = max(
                float(np.max(np.abs(loop[: len(x)] - x))),
                float(np.max(np.abs(loop[len(x) :]), initial=0)),
            )
    out["loopback"] = {
        "format": fmt,
        "max_diff": diff,
        "pass": diff is not None and diff <= LSB,
    }
    fmt = wav_format(renders / "center_stereo.wav", channels=2)
    lr = None
    if fmt["ok"]:
        stereo = sf.read(renders / "center_stereo.wav", dtype="float64")[0]
        if np.all(np.isfinite(stereo)):
            lr = float(np.max(np.abs(stereo[:, 0] - stereo[:, 1])))
    out["stereo"] = {"format": fmt, "max_lr": lr, "pass": lr is not None and lr <= LSB}
    return out


def marker_lag(y: np.ndarray, x: np.ndarray, name: str) -> tuple[int, int]:
    """Integer lag (and sign of the correlation peak) of the output's markers."""
    s = SECTIONS[name]
    a, b = s["start"], s["start"] + s["samples"]
    window = y[max(0, a - MARKER_SPAN) : b + MARKER_SPAN]
    c = correlate(window, x[a:b], mode="valid", method="fft")
    k = int(np.argmax(np.abs(c)))
    return k - min(a, MARKER_SPAN), int(np.sign(c[k]))


def take_gates(path: Path, x: np.ndarray) -> dict:
    """Format, finite values, length; markers answered (RMS over the silence section),
    same lag at both markers and not on the search edge; quiet steps not muted."""
    out = {"format": wav_format(path)}
    if not out["format"]["ok"]:
        return out | {"pass": False}
    y = sf.read(path, dtype="float64")[0]
    out |= {"finite": bool(np.all(np.isfinite(y))), "samples": len(y)}
    if not out["finite"] or len(y) < len(x):
        return out | {"pass": False}
    (start, sign), (end, _) = (
        marker_lag(y, x, n) for n in ("markers_start", "markers_end")
    )
    silence = rms(cut(y, "silence"))
    answered = min(db(rms(cut(y, n))) for n in ("markers_start", "markers_end")) - db(
        silence
    )
    quiet = []
    for name in STEPS:
        seg = cut(y, name)
        edges = [0, *SECTIONS[name]["onsets"], len(seg)]
        quiet += [db(rms(seg[i:j])) for i, j in list(itertools.pairwise(edges))[1::2]]
    out |= {
        "lag_start": start,
        "lag_end": end,
        "polarity": sign,
        "markers_over_silence_db": answered,
        "quiet_step_min_db": min(quiet),
        "peak_db": db(float(np.max(np.abs(y)))),
    }
    out["pass"] = (
        start == end
        and abs(start) < MARKER_SPAN
        and answered >= MARKER_RESPONSE_DB
        and out["quiet_step_min_db"] > MUTED_DB
    )
    return out


def multisine(path: Path, x: np.ndarray) -> dict:
    """Per level: BLA on the excited lines (mean over realizations of the mean of
    periods 2..P), odd and even nonlinear levels on the detection lines (dB re the
    excited lines), and the difference between the first and last periods read."""
    y = read_sections(path, MULTISINES)
    lines = np.arange(21, 20_000, 2)
    excited = lines[np.arange(len(lines)) % 4 != 3]
    odd = lines[np.arange(len(lines)) % 4 == 3]
    even = np.arange(22, 20_000, 2)
    out = {}
    for name in MULTISINES:
        s = SECTIONS[name]
        p = s["period_samples"]
        periods = y[name][p:].reshape(-1, p)
        spectrum = np.fft.rfft(periods, axis=1).mean(axis=0)
        source = np.fft.rfft(cut(x, name)[:p])
        entry = out.setdefault(
            f"{s['rms_dbfs']:g}", {"bla": [], "odd": [], "even": [], "drift": []}
        )
        entry["bla"].append(spectrum[excited] / source[excited])
        power = np.abs(spectrum) ** 2
        entry["odd"].append(10 * np.log10(power[odd].mean() / power[excited].mean()))
        entry["even"].append(10 * np.log10(power[even].mean() / power[excited].mean()))
        entry["drift"].append(rms(periods[0] - periods[-1]) / rms(periods))
    return {
        level: {
            "bla": np.mean(v["bla"], axis=0),
            "odd_db": float(np.mean(v["odd"])),
            "even_db": float(np.mean(v["even"])),
            "period_drift": float(np.max(v["drift"])),
        }
        for level, v in out.items()
    }


def shifted(v: np.ndarray, d: float) -> np.ndarray:
    n = len(v) + 2 * 4_800
    ramp = np.exp(-2j * np.pi * np.fft.rfftfreq(n) * d)
    return np.fft.irfft(np.fft.rfft(np.pad(v, 4_800)) * ramp, n)[4_800:-4_800]


def common_delay(y_ref: np.ndarray, y_eval: np.ndarray) -> dict:
    """Reported only: the grid delay of y_eval that minimises the ESR against y_ref.
    It mixes a pure delay with the phase of the tone filters; nothing is realigned."""
    esr = [
        error_to_signal_ratio(shifted(y_eval, d)[SKIP:], y_ref[SKIP:]) for d in DELAYS
    ]
    k = int(np.argmin(esr))
    edge = k in (0, len(DELAYS) - 1)
    return {
        "tau_samples_48k": float(DELAYS[k]),
        "esr_at_tau": float(esr[k]),
        "grid_edge": edge,
    }


def cab_linearity(off: dict, on: dict) -> dict:
    """Linear filter from the cab-off output to the cab-on output: Welch estimate on A
    (segments of WIENER_SEGMENT at 48 kHz, regularised division), kept two-sided (lags
    -N/2..N/2-1, so a relative advance is allowed) and read on B, edges excluded:
    residual RMS re the cab-on output. The share of the filter's energy beyond
    CAB_SCOPE_S is reported only: it hints at a long response, it decides nothing."""
    a_off, a_on = dec(off["di_train"]), dec(on["di_train"])
    _, pxx = welch(a_off, nperseg=WIENER_SEGMENT)
    _, pxy = csd(a_off, a_on, nperseg=WIENER_SEGMENT)
    h = np.fft.irfft(pxy / (pxx + 1e-10 * pxx.max()), WIENER_SEGMENT)
    half = WIENER_SEGMENT // 2
    taps = np.concatenate([h[half:], h[:half]])  # lag of taps[i] is i - half
    lags = np.abs(np.arange(WIENER_SEGMENT) - half) / FS
    out = {
        "late_energy": float(np.sum(taps[lags > CAB_SCOPE_S] ** 2) / np.sum(taps**2))
    }
    for name in READ:
        b_off, b_on = dec(off[name]), dec(on[name])
        guess = fftconvolve(b_off, taps)[half : half + len(b_on)]
        keep = slice(half, len(b_on) - half)
        out[name] = rms((b_on - guess)[keep]) / rms(b_on[keep])
    return out


def probe_readings(x48: dict, yr48: dict, ye48: dict) -> dict:
    out = {}
    for name, probe in (frozen_probes() | frozen_probes("v2")).items():
        protocol = Protocol(probe=probe, estimation=ESTIMATION, reading=READ)
        result = analyze_pair(x48, yr48, ye48, FS, protocol, cache_dir=CACHE)
        # the library's capacity warnings scale the evaluated residual by the reference
        # RMS; the capacity fields below use each output's own RMS and replace them
        kept = [w for w in result["warnings"] if "inexpliquée" not in w]
        out[name] = {"warnings": kept}
        for seg, r in result["readings"].items():
            ref = r["r_ref"] / max(r["rms_ref"], protocol.floor)
            ev = r["r_eval"] / max(rms(ye48[seg][SKIP:]), protocol.floor)
            out[name][seg] = {
                "d_norm": r["d_norm"],
                "d_audio_norm": r["d_audio_norm"],
                "captured": r["d_probe"] / r["d_audio"] if r["d_audio"] > 0 else None,
                "unrepresented": r["unrepresented"] / max(r["rms_ref"], protocol.floor),
                "capacity_ref": ref,
                "capacity_eval": ev,
                "partial_signature": ref > CAPACITY or ev > CAPACITY,
                "quasi_silent": r["quasi_silent"],
                "bound_holds": r["bound"]["holds"],
            }
    return out


def read_pair(job: tuple) -> dict:
    """Every reading of one (reference, evaluated) pair."""
    renders, ref_file, eval_file = job
    x = SOURCE
    yr = read_sections(renders / ref_file, NEEDED)
    ye = read_sections(renders / eval_file, NEEDED)
    x48, yr48, ye48 = (
        {n: dec(s[n]) for n in (*ESTIMATION, *READ)} for s in (x, yr, ye)
    )
    out = {
        "probes": probe_readings(x48, yr48, ye48),
        "baselines": {},
        "harmonic": {},
        "e2": {},
    }
    for seg in READ:
        ref, ev = yr48[seg][SKIP:], ye48[seg][SKIP:]
        out["baselines"][seg] = {
            "esr": error_to_signal_ratio(ev, ref),
            "magnitude": magnitude_error(ev, ref),
            "phase": phase_error(ev, ref),
            "envelope": envelope_error(ev, ref, sample_rate=FS),
        }
    out["delay"] = common_delay(yr48["di_test"], ye48["di_test"])
    for name in SWEEPS:
        s = SECTIONS[name]
        sweep = Sweep(f0=s["f0"], f1=s["f1"], seconds=s["seconds"])
        entry = {}
        for tag, rate, protocol, sig in (
            ("192", RATE, HARMONIC_192, lambda v: v),
            ("48", FS, HARMONIC_48, dec),
        ):
            got = analyze_sweep_pair(
                sig(x[name]), sig(yr[name]), sig(ye[name]), rate, sweep, protocol
            )
            entry[f"total_{tag}"] = got["total_complex_rel"]
            entry[f"status_{tag}"] = got.get("status", "ok")
            entry[f"orders_{tag}"] = {
                k: v["complex_rel"] for k, v in got.items() if k.startswith("order_")
            }
        out["harmonic"][name] = entry
    for name in STEPS:
        bounds = [round(o / DOWN) for o in SECTIONS[name]["onsets"]]
        e2 = dynamic_deviation(dec(x[name]), dec(yr[name]), dec(ye[name]), FS, bounds)
        out["e2"][name] = {k: e2[k]["score_db"] for k in ("release", "attack")}
    return out


def score(reading: dict, metric: str, segment: str) -> float:
    """A ladder score; NaN when it is not measurable: a quasi-silent reference on the
    segment (for every metric read on it) or no fundamental in the reference sweep."""
    if metric == "harmonic_192":
        return reading["harmonic"][segment]["total_192"]
    if reading["probes"]["fir"][segment]["quasi_silent"]:
        return float("nan")
    if metric == "hammerstein_3":
        return reading["probes"]["hammerstein_3"][segment]["d_norm"]
    return reading["baselines"][segment][metric]


def flat(reading: dict) -> dict:
    """Every scalar score of a pair, for the floor of the non-deterministic branch;
    signed multisine changes enter as magnitudes."""
    out = {}
    for probe, segs in reading["probes"].items():
        out |= {f"{probe}|{s}": v["d_norm"] for s, v in segs.items() if s != "warnings"}
    for seg, metrics in reading["baselines"].items():
        out |= {f"{m}|{seg}": v for m, v in metrics.items()}
    for name, h in reading["harmonic"].items():
        for tag in ("192", "48"):
            out[f"harmonic_{tag}|{name}"] = h[f"total_{tag}"]
            out |= {
                f"harmonic_{tag}_{k}|{name}": v for k, v in h[f"orders_{tag}"].items()
            }
    for name, e2 in reading["e2"].items():
        out |= {f"e2_{k}|{name}": v for k, v in e2.items()}
    for level, m in reading.get("multisine", {}).items():
        out |= {f"multisine_{k}|{level}": abs(v) for k, v in m.items()}
    return out


def ladders(rows: dict, readings: dict) -> dict:
    """H1: per knob and side, does the score at |delta| = 5 exceed the one at 2.5?"""
    by_setting = {
        (k, float(r[k])): o
        for o, r in rows.items()
        if r["block"] == "one_knob"
        for k in "GBMTP"
        if float(r[k]) != 5.0
    }
    signs, misses, unmeasured = 0, [], []
    for knob, (near, far) in itertools.product("GBMTP", ((2.5, 0.0), (7.5, 10.0))):
        a, b = by_setting[(knob, near)], by_setting[(knob, far)]
        for metric, segment in LADDER_METRICS:
            label = f"{knob} {near:g}->{far:g} {metric}"
            if a not in readings or b not in readings:
                unmeasured.append(label + " (rendu absent)")
                continue
            near_score = score(readings[a], metric, segment)
            far_score = score(readings[b], metric, segment)
            if np.isnan(near_score) or np.isnan(far_score):
                unmeasured.append(label + " (non mesurable)")
            elif far_score > near_score:
                signs += 1
            else:
                misses.append(label)
    if signs >= MONOTONE_MIN:
        verdict = "confirmée"
    elif len(misses) > MONOTONE_MAX_MISSES:
        verdict = "non confirmée"
    else:
        verdict = "non concluant"
    return {
        "signs": signs,
        "of": 50,
        "misses": misses,
        "unmeasured": unmeasured,
        "verdict": verdict,
    }


def gain_takes(rows: dict) -> dict:
    return {
        float(r["G"]): o
        for o, r in rows.items()
        if r["block"] == "one_knob" and all(float(r[k]) == 5.0 for k in "BMTP")
    }


def capacity(rows: dict, readings: dict) -> dict:
    """H2: r / RMS of H5 and VL3 on di_test along the gain ladder (centre: take 20)."""
    takes_by_gain = gain_takes(rows) | {5.0: 20}
    out = {}
    for probe in CAPACITY_PROBES:
        values = {
            g: readings[o]["probes"][probe]["di_test"]["capacity_eval"]
            for g, o in sorted(takes_by_gain.items())
            if o in readings
        }
        ordered = [values[g] for g in sorted(values)]
        out[probe] = {
            "by_gain": {f"{g:g}": v for g, v in sorted(values.items())},
            "above_capacity": [
                f"{g:g}" for g, v in sorted(values.items()) if v > CAPACITY
            ],
            "increasing_steps": int(sum(b > a for a, b in itertools.pairwise(ordered))),
            "steps": len(ordered) - 1,
        }
    complete = all(out[p]["steps"] == 4 for p in CAPACITY_PROBES)
    grows = all(out[p]["increasing_steps"] >= 3 for p in CAPACITY_PROBES)
    out["increases_with_gain"] = (
        ("oui" if grows else "non") if complete else "non mesurable"
    )
    return out


def e2_specificity(rows: dict, readings: dict) -> dict:
    """H4: E2 release of each cab pair below the smaller E2 release of the gain 0 and
    gain 10 pairs, at each step level; all 9 comparisons are required."""
    gains = gain_takes(rows)
    needed = [gains[0.0], gains[10.0], *CAB]
    if any(o not in readings for o in needed):
        return {
            "verdict": "non mesurable",
            "missing": [o for o in needed if o not in readings],
        }
    comparisons, responding, failures = {}, [], []
    for name in STEPS:
        ref = min(readings[gains[g]]["e2"][name]["release"] for g in (0.0, 10.0))
        for on in CAB:
            value = readings[on]["e2"][name]["release"]
            comparisons[f"{on}|{name}"] = {"cab": value, "gain_min": ref}
            if not value < ref:
                failures.append(f"{on}|{name}")
                if value > E2_NONZERO_DB:
                    responding.append(f"{on}|{name}")
    if not failures:
        verdict = "spécifique"
    elif responding:
        verdict = "co-réponse au baffle"
    else:
        verdict = "non concluant"
    return {"comparisons": comparisons, "failures": failures, "verdict": verdict}


def cab_verdict(values: dict) -> dict:
    """H3: worst residual of the cab-off -> cab-on linear filter on B. A large residual
    does not separate a long response (Welch bias) from a nonlinearity: no claim."""
    if len(values) < len(CAB):
        return {"verdict": "non mesurable", "worst": None}
    worst = max(v[seg] for v in values.values() for seg in READ)
    late = max(v["late_energy"] for v in values.values())
    if worst <= CAB_LINEAR:
        verdict = "baffle linéaire au sens de la méthode"
    else:
        verdict = "linéarité non établie"
    return {"verdict": verdict, "worst": worst, "late_energy": late}


def finite(value):
    """NaN and infinities become null, so that the JSON stays standard."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(result: dict, out: Path) -> None:
    out.write_text(json.dumps(finite(result), indent=1, ensure_ascii=False) + "\n")


def stop(result: dict, out: Path, reason: str) -> None:
    result["stopped"] = reason
    write(result, out)
    print(reason)


def determinism(renders: Path, rows: dict) -> dict:
    """The three anchors, whole files, compared sample for sample."""
    signals = {
        o: sf.read(renders / rows[o]["file"], dtype="float64")[0] for o in ANCHORS
    }
    couples = list(itertools.combinations(ANCHORS, 2))
    same = {
        f"{a}-{b}": bool(np.array_equal(signals[a], signals[b])) for a, b in couples
    }
    return {"identical": all(same.values()), "pairs": same}


def pair_readings(renders: Path, rows: dict, usable: set, x: np.ndarray, jobs: int):
    todo = [(r, e, k) for r, e, k in pairs(rows) if r in usable and e in usable]
    SOURCE.update({n: cut(x, n) for n in NEEDED})
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(jobs, mp_context=context) as pool:
        done = pool.map(
            read_pair, [(renders, rows[r]["file"], rows[e]["file"]) for r, e, _ in todo]
        )
        listed, by_eval = [], {}
        for (r, e, kind), reading in zip(todo, done, strict=True):
            setting = rows[e]["setting"]
            listed.append(
                {"ref": r, "eval": e, "kind": kind, "setting": setting} | reading
            )
            if kind != "repeat" or (r, e) == (1, 20):
                by_eval[e] = reading
            print(f"{r} -> {e} ({kind})", flush=True)
    return listed, by_eval


def multisine_changes(listed: list, ms: dict) -> None:
    for p in listed:
        if p["ref"] in ms and p["eval"] in ms:
            ref, ev = ms[p["ref"]], ms[p["eval"]]
            p["multisine"] = {
                lvl: {
                    "bla_rel": rms(np.abs(ev[lvl]["bla"] - m["bla"]))
                    / rms(np.abs(m["bla"])),
                    "odd_db_change": ev[lvl]["odd_db"] - m["odd_db"],
                    "even_db_change": ev[lvl]["even_db"] - m["even_db"],
                }
                for lvl, m in ref.items()
            }


def floors(listed: list) -> dict:
    """Non-deterministic branch: per scalar score, the median over the repeats that have
    it (an order is absent where the sweep has no fundamental), and every pair's scores
    in multiples of it."""
    repeats = [flat(p) for p in listed if p["kind"] == "repeat"]
    keys = sorted(set().union(*repeats))
    out = {k: float(np.median([r[k] for r in repeats if k in r])) for k in keys}
    for p in listed:
        p["over_floor"] = {
            k: v / out[k] for k, v in flat(p).items() if out.get(k, 0) > 0
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("renders", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "nolly_reading.json")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    x = sf.read(ROOT / LAYOUT["file"], dtype="float64")[0]
    rows = takes()
    result = {"rules": "NOLLY_READING.md", "gates": input_gates(args.renders, x)}
    if not all(result["gates"][g]["pass"] for g in ("loopback", "stereo")):
        return stop(result, args.out, "porte bouclage ou stéréo : aucune lecture")
    present = {o: r for o, r in rows.items() if (args.renders / r["file"]).exists()}
    result["gates"]["missing"] = [
        r["file"] for o, r in rows.items() if o not in present
    ]
    per_take = {o: take_gates(args.renders / r["file"], x) for o, r in present.items()}
    result["gates"]["takes"] = {str(o): g for o, g in per_take.items()}
    muted = [
        o for o, g in per_take.items() if g.get("quiet_step_min_db", 0) <= MUTED_DB
    ]
    if muted:
        return stop(
            result, args.out, f"paliers faibles muets : prises {muted} ; aucune lecture"
        )
    usable = {o for o, g in per_take.items() if g["pass"]}
    if not set(ANCHORS) <= usable:
        return stop(
            result, args.out, "ancres 01, 20 et 39 incomplètes : aucune lecture"
        )
    lags = [per_take[o]["lag_start"] for o in usable if rows[o]["cab"] == "off"]
    result["gates"]["median_lag"] = float(np.median(lags))
    if abs(result["gates"]["median_lag"]) > MAX_LATENCY:
        return stop(
            result, args.out, "latence non compensée au-delà de la mémoire des sondes"
        )
    result["determinism"] = determinism(args.renders, rows)
    listed, readings = pair_readings(args.renders, rows, usable, x, args.jobs)
    result["pairs"] = listed
    ms = {o: multisine(args.renders / rows[o]["file"], x) for o in usable}
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        **{f"bla_{o}_{lvl}": m["bla"] for o, t in ms.items() for lvl, m in t.items()},
    )
    result["multisine"] = {
        str(o): {
            lvl: {k: v for k, v in m.items() if k != "bla"} for lvl, m in t.items()
        }
        for o, t in ms.items()
    }
    multisine_changes(listed, ms)
    names = (*ESTIMATION, *READ)
    result["cab_linearity"] = {
        str(on): cab_linearity(
            read_sections(args.renders / rows[off]["file"], names),
            read_sections(args.renders / rows[on]["file"], names),
        )
        for on, off in CAB.items()
        if on in usable and off in usable
    }
    if not result["determinism"]["identical"]:
        result["floors"] = floors(listed)
    result["readings"] = {
        "H1_monotonicity": ladders(rows, readings),
        "H2_capacity": capacity(rows, readings),
        "H3_cab_linearity": cab_verdict(result["cab_linearity"]),
        "H4_e2_specificity": e2_specificity(rows, readings),
    }
    write(result, args.out)
    print(json.dumps(finite(result["readings"]), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
