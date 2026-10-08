#!/usr/bin/env python3
"""Audit of a mic'd VS100R capture delivery (product lane): Loopback/Mic pairs recorded
while the glued stimulus played (probe at 0 s, full at 53 s, overlapping by 3.508 s).

Read-only on the delivery. Measures, for the glued stimulus and each pair:

1. the glued against the probe and full sources, sample for sample (0-53 s = probe,
   from 56.508 s = full shifted by 53 s, the overlap separately); format, durations,
   pairing, recording margins against stimulus truncation;
2. per section (layouts placed on the recording through the measured lags): peak, RMS,
   samples at full scale, DC; noise and events in the margins and silences; runs of
   digital zeros; an estimate of the Mic error floor set by its noise, per section;
3. Loopback against the glued: lag at the four marker groups (sub-sample) and drift;
   lag slips, found by comparing each Loopback with that of a pair whose four marker
   lags agree (the analog loop biases the lag on bass-heavy material the same way in
   every pair, so it cancels), with the disturbed stretch bounded to 32 samples and
   what it holds; one scalar gain and its residual. Mic against Loopback, over
   30 Hz-18 kHz: best linear impulse response per section (delay, polarity,
   coherence) and its zero-lag term (an electrical path into the Mic channel has no
   acoustic delay; a disturbance common to both inputs also lands there);
   Loopback-Mic coherence above the stimulus band (25-90 kHz) over time, a
   disturbance common to both inputs;
4. within-take repeatability: Mic level of the sections the probe and the full share.

Settings are not inferred from the audio. Writes a JSON.
Usage: product_valvestate_mic_audit.py <delivery dir> <output json>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, csd, welch

RATE = 192_000
FULL_AT_S = 53.0
PAIRS = ("1044", "1057", "1105", "1133")
FULL_SCALE = 1.0 - 2.0**-23
MARKERS = {
    "probe_start": 3.0,
    "probe_end": 49.004,
    "full_start": 56.0,
    "full_end": 369.179,
}
# Sections the probe and the full share, for the within-take repeatability.
SHARED = (
    "sweep_-20dBFS",
    "sweep_-10dBFS",
    "sweep_-3dBFS",
    "two_tone_mod_-20dBFS",
    "two_tone_mod_-10dBFS",
    "two_tone_mod_-3dBFS",
    "two_tone_diff_-20dBFS",
    "two_tone_diff_-10dBFS",
    "two_tone_diff_-3dBFS",
)
# Broadband sections for the Mic impulse response (two tones are ill-conditioned).
IR_SECTIONS = (
    "probe.sweep_-20dBFS",
    "full.pink_noise_-30dBFS_rms",
    "full.sweep_-20dBFS",
    "full.di_train",
    "full.di_test",
    "full.level_-12dB",
)
SILENT = ("pad_start", "pad_end", "silence", "markers_start", "markers_end")


def db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-12)))


def read(path: Path, dtype: str = "float32") -> np.ndarray:
    x, rate = sf.read(path, dtype=dtype, always_2d=False)
    assert rate == RATE, (path, rate)
    return x


def paths(delivery: Path, pair: str) -> tuple[Path, Path]:
    media = delivery / "Media"
    return media / f"03-Loopback-261008_{pair}.wav", media / f"04-Mic-261008_{pair}.wav"


def glued_layout(probe: dict, full: dict) -> list[dict]:
    """Sections of both sources on the glued timeline, in samples."""
    shift = int(FULL_AT_S * RATE)
    out = [dict(s, section=f"probe.{s['section']}") for s in probe["sections"]]
    out += [
        dict(s, section=f"full.{s['section']}", start=s["start"] + shift)
        for s in full["sections"]
    ]
    return out


def check_glued(delivery: Path) -> dict:
    media = delivery / "Media"
    g = read(media / "reamp_valvestate_mic_probe_192k-glued.wav", "int32")
    p = read(media / "reamp_valvestate_mic_probe_192k.wav", "int32")
    f = read(media / "reamp_valvestate_mic_192k-imported.wav", "int32")
    shift, probe_end = int(FULL_AT_S * RATE), len(p)
    overlap = g[shift:probe_end].astype(np.int64)
    as_sum = p[shift:].astype(np.int64) + f[: probe_end - shift].astype(np.int64)
    return {
        "glued_samples": len(g),
        "glued_seconds": len(g) / RATE,
        "expected_samples": shift + len(f),
        "probe_head_identical_0_53s": bool(np.array_equal(g[:shift], p[:shift])),
        "full_identical_after_probe_end": bool(
            np.array_equal(g[probe_end:], f[probe_end - shift :])
        ),
        "overlap_s": (probe_end - shift) / RATE,
        "probe_in_overlap_all_zero": bool(not p[shift:].any()),
        "overlap_equals_full": bool(np.array_equal(overlap, f[: probe_end - shift])),
        "overlap_equals_sum": bool(np.array_equal(overlap, as_sum)),
    }


def xcorr_lag(ref: np.ndarray, sig: np.ndarray, max_lag: int) -> tuple[float, float]:
    """Lag (samples, sub-sample) of sig relative to ref, |lag| <= max_lag, and the
    normalised correlation at the peak (signed)."""
    ref, sig = ref.astype(np.float64), sig.astype(np.float64)
    c = correlate(sig, ref, mode="full", method="fft")
    zero = len(ref) - 1
    lo, hi = max(zero - max_lag, 1), min(zero + max_lag, len(c) - 2)
    k = int(np.argmax(np.abs(c[lo : hi + 1]))) + lo
    y0, y1, y2 = c[k - 1], c[k], c[k + 1]
    denom = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    norm = np.sqrt(np.sum(ref**2) * np.sum(sig**2))
    return float(k - zero + frac), float(y1 / norm) if norm > 0 else 0.0


def coarse_offset(glued: np.ndarray, rec: np.ndarray) -> int:
    """Where the glued starts in the recording. Normalised cross-correlation of 3-40 s
    of the glued (markers, sweeps, two tones: a pattern that occurs once) over the
    first 90 s; a short marker template alone also matches loud sections."""
    a, b = int(3.0 * RATE), int(40.0 * RATE)
    ref = glued[a:b].astype(np.float64)
    search = rec[: min(len(rec), 90 * RATE)].astype(np.float64)
    c = correlate(search, ref, mode="valid", method="fft")
    energy = np.cumsum(np.concatenate([[0.0], search**2]))
    local = energy[len(ref) :] - energy[: len(energy) - len(ref)]
    ncc = c / np.sqrt(np.maximum(local, 1e-20) * np.sum(ref**2))
    return int(np.argmax(np.abs(ncc))) - a


def marker_lags(glued: np.ndarray, rec: np.ndarray, offset: int) -> dict:
    """Lag at each marker group (both blips, 0.6 s window), relative to the coarse
    offset, sub-sample; None when the group lies outside the recording."""
    out = {}
    pad = 2000
    for name, t in MARKERS.items():
        a, b = int((t + 0.45) * RATE), int((t + 1.06) * RATE)
        if a + offset - pad < 0 or b + offset + pad > len(rec):
            out[name] = None
            continue
        ref = np.concatenate([np.zeros(pad), glued[a:b], np.zeros(pad)])
        lag, r = xcorr_lag(ref, rec[a + offset - pad : b + offset + pad], pad)
        out[name] = {"time_s": t, "lag_samples": lag, "corr": r}
    return out


def relative_lags(x, off_x, ref, off_ref, span_s: float) -> list[tuple[float, int]]:
    """Integer lag of a Loopback against the reference Loopback on 0.5 s windows every
    1 s of the glued, at the peak of the normalised correlation; kept when the match is
    close (> 0.9) and beats every lag outside +-16 samples by 0.02."""
    out, search, n = [], 1200, int(0.5 * RATE)
    for t in np.arange(1.0, span_s - 1.0, 1.0):
        a = int(t * RATE)
        if a + off_x - search < 0 or a + off_x + n + search > len(x):
            continue
        if a + off_ref < 0 or a + off_ref + n > len(ref):
            continue
        r = ref[a + off_ref : a + off_ref + n].astype(np.float64)
        if np.sqrt(np.mean(r**2)) < 1e-3:
            continue
        s = x[a + off_x - search : a + off_x + n + search].astype(np.float64)
        c = correlate(s, r, mode="valid", method="fft")
        energy = np.cumsum(np.concatenate([[0.0], s**2]))
        ncc = c / np.sqrt((energy[n:] - energy[:-n]) * np.sum(r**2) + 1e-30)
        k = int(np.argmax(ncc))
        mask = np.ones(len(ncc), bool)
        mask[max(k - 16, 0) : k + 17] = False
        if ncc[k] > 0.9 and mask.any() and ncc[k] > ncc[mask].max() + 0.02:
            out.append((float(t), k - search))
    return out


def find_slips(series: list[tuple[float, int]]) -> list[dict]:
    """Lag changes of at least 8 samples that hold for 3 kept windows (one-sample
    flickers of the integer peak are not slips)."""
    slips, current = [], series[0][1] if series else 0
    for i in range(1, len(series)):
        t, lag = series[i]
        if (
            abs(lag - current) >= 8
            and all(abs(v - lag) <= 1 for _, v in series[i : i + 3])
            and len(series[i : i + 3]) == 3
        ):
            slips.append(
                {"after_s": series[i - 1][0], "before_s": t, "samples": lag - current}
            )
            current = lag
    return slips


def rms_dbfs(x: np.ndarray) -> float:
    return db(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))))


def locate_slip(x, mic, off_x: int, ref, off_ref: int, slip: dict) -> dict:
    """Bound the stretch where a Loopback leaves the stimulus at a lag change of n
    samples (off_x, off_ref: recording index of glued sample 0 before the change).

    The switch from the old to the new alignment against the reference Loopback is
    placed by least squares (one break, recording samples); around it, the disturbed
    stretch runs from the end of the last 32-sample window that matches the old
    alignment to the start of the first that matches the new one (match: squared error
    20 dB below the reference, which must be above -80 dBFS). Reported: its level and
    exact zeros on both channels, the reference Loopback level over the same glued
    span, the Mic level just before, and how many glued samples lie between the two
    matching parts (negative: played twice)."""
    n, w = slip["samples"], 32
    a = int(slip["after_s"] * RATE) + off_x
    b = int((slip["before_s"] + 0.5) * RATE) + off_x
    j = np.arange(a, b)
    x64 = x[a:b].astype(np.float64)
    r_old = ref[j - off_x + off_ref].astype(np.float64)
    r_new = ref[j - off_x - n + off_ref].astype(np.float64)
    c_old = np.concatenate([[0.0], np.cumsum((x64 - r_old) ** 2)])
    c_new = np.concatenate([[0.0], np.cumsum((x64 - r_new) ** 2)])
    k = int(np.argmin(c_old + c_new[-1] - c_new))
    starts = np.arange(max(k - RATE // 10, 0), min(k + RATE // 10, len(x64) - w), w)

    def matches(r: np.ndarray) -> np.ndarray:
        err = np.array([np.sum((x64[s : s + w] - r[s : s + w]) ** 2) for s in starts])
        energy = np.array([np.sum(r[s : s + w] ** 2) for s in starts])
        return (err < 0.01 * energy) & (energy > w * 1e-8)

    old_ok, new_ok = matches(r_old), matches(r_new)
    lo = a + int(starts[old_ok & (starts <= k)][-1]) + w
    hi = a + int(starts[new_ok & (starts >= k - w)][0])
    gap = x[lo:hi]
    return {
        "samples": n,
        "glued_time_s": (lo - off_x) / RATE,
        "recording_start": lo,
        "recording_end": hi,
        "recording_start_s": lo / RATE,
        "disturbed_samples": hi - lo,
        "disturbed_ms": (hi - lo) / RATE * 1e3,
        "glued_samples_between_matches": hi - lo - n,
        "window_samples": w,
        "loopback_rms_dbfs": rms_dbfs(gap),
        "reference_loopback_rms_dbfs": rms_dbfs(
            ref[lo - off_x + off_ref : hi - off_x + off_ref]
        ),
        "loopback_exact_zeros": int(np.sum(gap == 0)),
        "mic_rms_dbfs": rms_dbfs(mic[lo:hi]),
        "mic_rms_dbfs_previous_0.1s": rms_dbfs(mic[lo - RATE // 10 : lo]),
        "mic_exact_zeros": int(np.sum(mic[lo:hi] == 0)),
    }


def zero_runs(x: np.ndarray, min_len: int = 64) -> list[tuple[int, int]]:
    """Runs of exact digital zeros of at least min_len samples."""
    z = np.concatenate([[False], x == 0, [False]])
    edges = np.flatnonzero(np.diff(z.astype(np.int8)))
    starts, ends = edges[::2], edges[1::2]
    keep = ends - starts >= min_len
    return list(zip(starts[keep].tolist(), ends[keep].tolist(), strict=True))


def section_stats(x: np.ndarray) -> dict:
    x64 = x.astype(np.float64)
    return {
        "peak_dbfs": db(float(np.max(np.abs(x64)))),
        "rms_dbfs": db(float(np.sqrt(np.mean(x64**2)))),
        "samples_at_full_scale": int(np.count_nonzero(np.abs(x64) >= FULL_SCALE)),
        "dc": float(np.mean(x64)),
    }


def noise_profile(x: np.ndarray) -> dict:
    """RMS, the part below 200 Hz, and the strongest bin, of a stretch without
    stimulus; mains lines against the median bin of 20 Hz-20 kHz."""
    f, p = welch(x.astype(np.float64), fs=RATE, nperseg=min(len(x), 1 << 16))
    df = f[1] - f[0]
    floor = np.median(p[(f >= 20) & (f <= 20_000)])
    lines = {}
    for hz in (50, 100, 150):
        k = int(np.rint(hz / df))
        lines[f"{hz}Hz_over_floor_db"] = float(
            10 * np.log10(max(p[k - 1 : k + 2].max(), 1e-30) / max(floor, 1e-30))
        )
    return {
        "rms_dbfs": db(float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))),
        "below_200Hz_dbfs": float(10 * np.log10(max(np.sum(p[f < 200]) * df, 1e-30))),
        "strongest_bin_hz": float(f[int(np.argmax(p))]),
        **lines,
    }


def events(x: np.ndarray, threshold_dbfs: float = -30.0) -> dict:
    """Loudest 0.1 s of a margin and the time it starts (s from the margin start)."""
    w, hop = int(0.1 * RATE), int(0.025 * RATE)
    if len(x) < w:
        return {}
    energy = np.cumsum(np.concatenate([[0.0], x.astype(np.float64) ** 2]))
    starts = np.arange(0, len(x) - w + 1, hop)
    rms = np.sqrt((energy[starts + w] - energy[starts]) / w)
    k = int(np.argmax(rms))
    return {
        "loudest_0.1s_dbfs": db(float(rms[k])),
        "at_s": float(starts[k] / RATE),
        "above_threshold": bool(rms[k] > 10 ** (threshold_dbfs / 20)),
    }


def passband(f: np.ndarray) -> np.ndarray:
    """1 over 30 Hz-18 kHz, raised-cosine edges down to 0 at 20 Hz and 20 kHz: inside
    the band of the sweeps and the pink noise (a brick wall would ring through the
    impulse response; outside the stimulus band H1 is noise over noise)."""
    w = ((f >= 30) & (f <= 18_000)).astype(np.float64)
    up = (f > 20) & (f < 30)
    w[up] = 0.5 - 0.5 * np.cos(np.pi * (f[up] - 20) / 10)
    down = (f > 18_000) & (f < 20_000)
    w[down] = 0.5 + 0.5 * np.cos(np.pi * (f[down] - 18_000) / 2_000)
    return w


def mic_ir(lo: np.ndarray, mi: np.ndarray) -> dict:
    """Best linear impulse response Mic/Loopback (H1, Welch) weighted by passband():
    main peak (50-400 samples), its sign, onset (first tap above 20 % of it), the
    largest tap within the first 8 samples (no acoustic path is that short) and the
    share of the response energy (first 4096 taps) in those 8; mean coherence over
    200 Hz-5 kHz and over 25-90 kHz (above the stimulus band: a common disturbance)."""
    nfft = 1 << 15
    lo, mi = lo.astype(np.float64), mi.astype(np.float64)
    f, pxy = csd(lo, mi, fs=RATE, nperseg=nfft)
    _, pxx = welch(lo, fs=RATE, nperseg=nfft)
    _, pyy = welch(mi, fs=RATE, nperseg=nfft)
    h = np.fft.irfft(pxy / np.maximum(pxx, 1e-300) * passband(f), n=nfft)
    main = int(np.argmax(np.abs(h[50:400]))) + 50
    onset = int(np.argmax(np.abs(h[: main + 1]) > 0.2 * abs(h[main])))
    z = int(np.argmax(np.abs(h[:8])))
    coh = np.abs(pxy) ** 2 / np.maximum(pxx * pyy, 1e-300)
    return {
        "main_lag_samples": main,
        "main_lag_ms": main / RATE * 1e3,
        "main_tap": float(h[main]),
        "polarity": int(np.sign(h[main])),
        "onset_samples": onset,
        "zero_lag_tap": float(h[z]),
        "zero_lag_over_main_db": db(abs(h[z])) - db(abs(h[main])),
        "zero_lag_energy_share_db": float(
            10 * np.log10(np.sum(h[:8] ** 2) / np.sum(h[:4096] ** 2))
        ),
        "coherence_200Hz_5kHz": float(np.mean(coh[(f >= 200) & (f <= 5000)])),
        "coherence_25_90kHz": float(np.mean(coh[(f >= 25_000) & (f <= 90_000)])),
    }


def common_disturbance(loop: np.ndarray, mic: np.ndarray) -> dict:
    """Loopback-Mic coherence over 25-90 kHz on 1 s windows of the recording, with the
    level of each channel in that band; stretches of windows above 0.3."""
    rows = []
    for a in range(0, len(loop) - RATE + 1, RATE):
        lo = loop[a : a + RATE].astype(np.float64)
        mi = mic[a : a + RATE].astype(np.float64)
        f, pxy = csd(lo, mi, fs=RATE, nperseg=4096)
        _, pxx = welch(lo, fs=RATE, nperseg=4096)
        _, pyy = welch(mi, fs=RATE, nperseg=4096)
        band = (f >= 25_000) & (f <= 90_000)
        df = f[1] - f[0]
        coh = np.abs(pxy[band]) ** 2 / np.maximum(pxx[band] * pyy[band], 1e-300)
        rows.append(
            (
                a / RATE,
                float(np.mean(coh)),
                float(10 * np.log10(max(np.sum(pxx[band]) * df, 1e-30))),
                float(10 * np.log10(max(np.sum(pyy[band]) * df, 1e-30))),
            )
        )
    stretches: list[dict] = []
    for t, c, lev_l, lev_m in rows:
        if c <= 0.3:
            continue
        if stretches and stretches[-1]["end_s"] == t:
            s = stretches[-1]
            s["end_s"] = t + 1.0
            s["max_coherence"] = max(s["max_coherence"], c)
            s["max_loopback_dbfs"] = max(s["max_loopback_dbfs"], lev_l)
            s["max_mic_dbfs"] = max(s["max_mic_dbfs"], lev_m)
        else:
            stretches.append(
                {
                    "start_s": t,
                    "end_s": t + 1.0,
                    "max_coherence": c,
                    "max_loopback_dbfs": lev_l,
                    "max_mic_dbfs": lev_m,
                }
            )
    coh = np.array([r[1] for r in rows])
    return {
        "median_coherence": float(np.median(coh)),
        "max_coherence": float(np.max(coh)),
        "seconds_above_0.3": int(np.sum(coh > 0.3)),
        "median_mic_dbfs": float(np.median([r[3] for r in rows])),
        "stretches_recording_s": stretches,
    }


def formats(*files: Path) -> dict:
    out = {}
    for name, path in zip(("loopback", "mic"), files, strict=True):
        i = sf.info(path)
        out[name] = {
            "rate": i.samplerate,
            "channels": i.channels,
            "subtype": i.subtype,
            "frames": i.frames,
            "seconds": i.frames / i.samplerate,
        }
    return out


def alignment(glued, loop, mic, ref, off_ref, is_ref: bool) -> dict:
    """Offset, marker lags, slips (located) and the constant-lag segments."""
    offset = coarse_offset(glued, loop)
    markers = marker_lags(glued, loop, offset)
    base = int(np.rint(markers["probe_start"]["lag_samples"]))
    span = len(glued) / RATE
    series = [] if is_ref else relative_lags(loop, offset, ref, off_ref, span)
    slips = [] if is_ref else find_slips(series)
    located, lag = [], base
    for s in slips:
        located.append(locate_slip(loop, mic, offset + lag, ref, off_ref, s))
        lag += s["samples"]
    found = [m for m in markers.values() if m]
    shift = sum(s["samples"] for s in slips)
    drift = found[-1]["lag_samples"] - found[0]["lag_samples"] - shift
    lags = [base]
    for s in located:
        lags.append(lags[-1] + s["samples"])
    return {
        "offset": offset,
        "markers": markers,
        "base": base,
        "windows": len(series),
        "slips": located,
        "drift_ppm": drift / ((found[-1]["time_s"] - found[0]["time_s"]) * RATE) * 1e6,
        "total": offset + base + shift + len(glued),
        "bounds": [0.0] + [s["glued_time_s"] for s in located] + [span],
        "lags": lags,
    }


def rec_index(al: dict, sample: int) -> int:
    """Recording index of a glued sample, through the lag of its segment."""
    t = sample / RATE
    for k, lag in enumerate(al["lags"]):
        if al["bounds"][k] <= t < al["bounds"][k + 1]:
            return sample + al["offset"] + lag
    return sample + al["offset"] + al["lags"][-1]


def loopback_fit(glued, loop, al: dict) -> dict:
    """One scalar gain of the Loopback against the glued, fitted over all the
    constant-lag segments together, and the residual of that single fit."""
    num = den = 0.0
    parts = []
    offset = al["offset"]
    for k, lag in enumerate(al["lags"]):
        a = max(int(al["bounds"][k] * RATE), -(offset + lag))
        b = min(int(al["bounds"][k + 1] * RATE), len(loop) - offset - lag)
        if b > a:
            g = glued[a:b].astype(np.float64)
            r = loop[a + offset + lag : b + offset + lag].astype(np.float64)
            num += float(np.dot(g, r))
            den += float(np.dot(g, g))
            parts.append((g, r))
    gain = num / den
    res = sum(float(np.sum((r - gain * g) ** 2)) for g, r in parts)
    sig = sum(float(np.sum((gain * g) ** 2)) for g, _ in parts)
    return {
        "loopback_gain_db": db(abs(gain)),
        "loopback_gain_sign": int(np.sign(gain)),
        "loopback_residual_db": float(10 * np.log10(max(res, 1e-30) / max(sig, 1e-30))),
    }


def section_table(layout, loop, mic, al: dict) -> dict:
    sections = {}
    for s in layout:
        a = rec_index(al, s["start"])
        b = a + s["samples"]
        if a < 0 or b > len(loop):
            sections[s["section"]] = None
            continue
        sections[s["section"]] = {
            "loopback": section_stats(loop[a:b]),
            "mic": section_stats(mic[a:b]),
        }
    return sections


def quiet_table(layout, loop, mic, al: dict) -> dict:
    """Stretches without stimulus: margins and silence sections."""
    quiet = {}
    first, last = al["offset"] + al["base"], al["total"]
    if first > RATE:
        quiet["margin_before"] = (0, first - RATE // 2)
    if len(loop) - last > RATE:
        quiet["margin_after"] = (last + RATE // 2, len(loop))
    for s in layout:
        if s["section"].endswith(".silence"):
            a = rec_index(al, s["start"])
            quiet[s["section"]] = (a, a + s["samples"])
    return {
        name: {
            "start_s": a / RATE,
            "seconds": (b - a) / RATE,
            "loopback": noise_profile(loop[a:b]),
            "mic": noise_profile(mic[a:b]),
            "mic_event": events(mic[a:b]),
        }
        for name, (a, b) in quiet.items()
    }


def noise_floor_errors(quiet: dict, sections: dict) -> dict:
    """Noise-limited error floor of the Mic per sounding section: noise energy (mean
    of the two silence sections) over the section's energy, in dB."""
    keys = [k for k in ("probe.silence", "full.silence") if k in quiet]
    noise = np.mean([10 ** (quiet[k]["mic"]["rms_dbfs"] / 10) for k in keys])
    return {
        name: float(10 * np.log10(noise) - st["mic"]["rms_dbfs"])
        for name, st in sections.items()
        if st and not name.endswith(SILENT)
    }


def zero_run_table(layout, loop, mic, al: dict) -> dict:
    """Digital zero runs inside sounding sections (dropouts)."""
    drops = {}
    for name, x in (("loopback", loop), ("mic", mic)):
        runs = []
        for s in layout:
            if s["section"].endswith(SILENT):
                continue
            a = rec_index(al, s["start"])
            for z0, z1 in zero_runs(x[max(a, 0) : a + s["samples"]]):
                runs.append(
                    {
                        "section": s["section"],
                        "at_s": (a + z0) / RATE,
                        "samples": z1 - z0,
                    }
                )
        drops[name] = runs
    return drops


def ir_table(layout, loop, mic, al: dict) -> dict:
    """Mic against Loopback, per broadband section (up to 30 s each)."""
    ir = {}
    for name in IR_SECTIONS:
        s = next(x for x in layout if x["section"] == name)
        a = rec_index(al, s["start"])
        n = min(s["samples"], 30 * RATE)
        ir[name] = mic_ir(loop[a : a + n], mic[a : a + n])
    return ir


def repeatability(sections: dict) -> dict:
    """Mic and Loopback level, full minus probe, on the sections both contain."""
    out = {}
    for name in SHARED:
        p, f = sections.get(f"probe.{name}"), sections.get(f"full.{name}")
        if p and f:
            out[name] = {
                "mic_rms_full_minus_probe_db": f["mic"]["rms_dbfs"]
                - p["mic"]["rms_dbfs"],
                "loopback_rms_full_minus_probe_db": f["loopback"]["rms_dbfs"]
                - p["loopback"]["rms_dbfs"],
            }
    return out


def audit_pair(delivery, pair, glued, layout, ref, off_ref, is_ref) -> dict:
    p_loop, p_mic = paths(delivery, pair)
    loop, mic = read(p_loop), read(p_mic)
    al = alignment(glued, loop, mic, ref, off_ref, is_ref)
    first = al["offset"] + al["base"]
    out: dict = {
        "format": formats(p_loop, p_mic),
        "same_length": len(loop) == len(mic),
        "reference": is_ref,
        "offset_samples": al["offset"],
        "offset_s": al["offset"] / RATE,
        "markers": al["markers"],
        "relative_lag_windows": al["windows"],
        "slips": al["slips"],
        "drift_ppm_excluding_slips": al["drift_ppm"],
        "margin_before_s": first / RATE,
        "margin_after_s": (len(loop) - al["total"]) / RATE,
        "stimulus_truncated_start": first < 0,
        "stimulus_truncated_end": al["total"] > len(loop),
    }
    if first < 0:  # what a negative margin cuts off: leading digital silence?
        out["cut_at_start_is_silence"] = bool(not np.any(glued[:-first]))
    out |= loopback_fit(glued, loop, al)
    sections = section_table(layout, loop, mic, al)
    out["sections"] = sections
    out["whole"] = {"loopback": section_stats(loop), "mic": section_stats(mic)}
    out["quiet"] = quiet_table(layout, loop, mic, al)
    out["mic_noise_floor_error_db"] = noise_floor_errors(out["quiet"], sections)
    out["zero_runs_in_sounding_sections"] = zero_run_table(layout, loop, mic, al)
    out["mic_ir"] = ir_table(layout, loop, mic, al)
    out["common_disturbance_25_90kHz"] = common_disturbance(loop, mic)
    out["repeatability"] = repeatability(sections)
    return out


def main() -> None:
    delivery, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    probe = json.loads((delivery / "layout_probe.json").read_text())
    full = json.loads((delivery / "layout.json").read_text())
    layout = glued_layout(probe, full)
    result: dict = {"glued": check_glued(delivery), "pairs": {}}
    print(json.dumps(result["glued"], indent=1), flush=True)
    glued = read(delivery / "Media" / "reamp_valvestate_mic_probe_192k-glued.wav")

    # Reference Loopback: the first pair whose four marker lags agree within 0.5 sample.
    ref = off_ref = ref_pair = None
    for pair in PAIRS:
        loop = read(paths(delivery, pair)[0])
        offset = coarse_offset(glued, loop)
        m = [v["lag_samples"] for v in marker_lags(glued, loop, offset).values() if v]
        if len(m) == len(MARKERS) and max(m) - min(m) < 0.5:
            base = int(np.rint(m[0]))
            ref, off_ref, ref_pair = loop, offset + base, pair
            break
    assert ref is not None, (
        "no pair with constant marker lags to compare the others with"
    )
    result["reference_pair"] = ref_pair
    print(f"reference pair {ref_pair}", flush=True)

    for pair in PAIRS:
        start = time.time()
        result["pairs"][pair] = r = audit_pair(
            delivery, pair, glued, layout, ref, off_ref, pair == ref_pair
        )
        slips = [
            (x["samples"], round(x["glued_time_s"], 4), x["disturbed_samples"])
            for x in r["slips"]
        ]
        print(
            f"{pair}: margins {r['margin_before_s']:.3f}/{r['margin_after_s']:.3f} s, "
            f"drift {r['drift_ppm_excluding_slips']:.4f} ppm, slips {slips}, "
            f"{time.time() - start:.0f} s",
            flush=True,
        )
    out_path.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
