#!/usr/bin/env python3
"""Quality control of the VS100R FX loop takes. (Product lane, CPU, read-only on audio.)

Each FX out file pairs with the loopback of the same Reaper stamp, recorded in the same
stream, so the pass start found on the loopback (passes.json) holds for the FX track.
Per take, following demo/capture/valvestate/ANALYSIS_RULES.md:
- peak of the FX track over the pass (clipping at >= -0.1 dBFS);
- RMS of the DI sections and of the 5 s silence section (the take's noise floor);
- SNR = DI RMS / silence RMS; below 20 dB the take is "faible signal";
- isolated spikes, as seconds from the pass start (source time) and WAV sample index;
- FX-vs-loopback delay on the opening and closing blips: equal values mean the two
  tracks stayed in step over the pass (not that they never slipped in between).

Writes demo/capture/valvestate/qc.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "demo/capture/valvestate"
MEDIA = ROOT / "datasets/amp_capture/Media"
LAYOUT = json.loads((SHEET / "layout.json").read_text())
SECTIONS = {s["section"]: s for s in LAYOUT["sections"]}
RATE = LAYOUT["sample_rate"]
MARGIN = RATE // 10  # 0.1 s trimmed at each section edge
LOW_SNR_DB = 20.0
SPIKE_RATIO = 4.0  # a sample this many times above its 16 neighbours is a spike


def dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def section(x: np.ndarray, start: int, name: str) -> np.ndarray:
    s = SECTIONS[name]
    return x[start + s["start"] + MARGIN : start + s["start"] + s["samples"] - MARGIN]


def spikes(x: np.ndarray) -> list[int]:
    """Isolated samples far above every one of their 8 neighbours on each side."""
    a = np.abs(x.astype(np.float64))
    width = 8
    peaks = (a == maximum_filter1d(a, 2 * width + 1)) & (a > 10 ** (-40 / 20))
    index = np.nonzero(peaks)[0]
    index = index[(index >= width) & (index < len(a) - width)]
    offsets = np.concatenate([np.arange(-width, 0), np.arange(1, width + 1)])
    neighbours = a[index[:, None] + offsets].max(axis=1)
    return index[a[index] > SPIKE_RATIO * neighbours].tolist()


def blip_lag(fx: np.ndarray, loop: np.ndarray, start: int, name: str) -> int:
    s = SECTIONS[name]
    a, n, reach = start + s["start"], s["samples"], 4_096
    reference = loop[a : a + n].astype(np.float64)
    window = fx[a - reach : a + n + reach].astype(np.float64)
    corr = fftconvolve(window, reference[::-1], mode="valid")
    return int(np.argmax(np.abs(corr))) - reach


def main() -> None:
    passes = json.loads((SHEET / "passes.json").read_text())
    rows = []
    for p in passes:
        stamp = p["file"].removeprefix("02-Looopbck-")
        fx_path = MEDIA / f"03-FX Out-{stamp}"
        if not fx_path.exists():
            continue
        fx, rate = sf.read(fx_path, dtype="float32")
        info = sf.info(fx_path)
        assert rate == RATE and fx.ndim == 1, fx_path
        start = p["start"]
        end = start + SECTIONS["pad_end"]["start"]
        di = np.concatenate(
            [section(fx, start, n) for n in ("di_train", "di_validation", "di_test")]
        )
        silence = section(fx, start, "silence")
        row = {
            "order": p.get("order"),
            "setting": p.get("setting"),
            "fx_file": fx_path.name,
            "subtype": info.subtype,
            "same_length_as_loopback": info.frames == sf.info(MEDIA / p["file"]).frames,
            "peak_dbfs": round(float(20 * np.log10(np.abs(fx[start:end]).max())), 2),
            "di_rms_dbfs": round(dbfs(di), 2),
            "silence_rms_dbfs": round(dbfs(silence), 2),
        }
        found = spikes(fx[start:end])
        row["spikes_s"] = [round(i / RATE, 4) for i in found]  # from the pass start
        row["spikes_wav_sample"] = [start + i for i in found]
        loop = sf.read(MEDIA / p["file"], dtype="float32")[0]
        row["blip_lag_start"] = blip_lag(fx, loop, start, "markers_start")
        row["blip_lag_end"] = blip_lag(fx, loop, start, "markers_end")
        row["snr_db"] = round(row["di_rms_dbfs"] - row["silence_rms_dbfs"], 1)
        row["clipped"] = row["peak_dbfs"] >= -0.1
        row["low_signal"] = row["snr_db"] < LOW_SNR_DB
        rows.append(row)
        print(
            f"{row['order']!s:>3} {row['setting']!s:32s} peak {row['peak_dbfs']:6.1f}"
            f"  DI {row['di_rms_dbfs']:6.1f}  silence {row['silence_rms_dbfs']:6.1f}"
            f"  SNR {row['snr_db']:5.1f}"
            + ("  ECRETE" if row["clipped"] else "")
            + ("  FAIBLE" if row["low_signal"] else "")
            + ("" if row["same_length_as_loopback"] else "  LONGUEUR")
            + (f"  PICS {row['spikes_s']}" if found else "")
            + f"  decalage bips {row['blip_lag_start']}/{row['blip_lag_end']}",
            flush=True,
        )
    (SHEET / "qc.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
