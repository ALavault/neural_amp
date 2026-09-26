#!/usr/bin/env python3
"""Where does the VS100R output noise come from? (Product lane, CPU, descriptive.)

In the 5 s silence section the source is digital zero, so the amp receives only the
noise of the playback chain (converter, reamp box), which the loopback also records,
plus the loopback input's own noise, which the amp never receives. Per take:
- FX noise level, full band and in 100 Hz - 10 kHz;
- C = sum(|S_lf|^2 / S_ll) / sum S_ff: share of the FX noise linearly explained by the
  loopback noise (estimator floor about 1/windows when nothing is shared).
High C: the amp amplifies the noise it is sent. C at the floor: its own noise (or noise
the loopback measures but the amp never received). Reported against the gain knob.
The FX noise is also split, in dBFS: broadband (median bin level over the band), mains
harmonics (50 Hz multiples, +-3 bins) and the 8 kHz line, both above that floor.

Writes demo/capture/valvestate/noise.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "demo/capture/valvestate"
MEDIA = ROOT / "datasets/amp_capture/Media"
LAYOUT = json.loads((SHEET / "layout.json").read_text())
SILENCE = next(s for s in LAYOUT["sections"] if s["section"] == "silence")
RATE = LAYOUT["sample_rate"]
WIN, HOP = 65536, 32768  # 2.9 Hz bins: mains harmonics resolved
BAND = (100.0, 10_000.0)


def spectra(a: np.ndarray, b: np.ndarray):
    taper = np.hanning(WIN)
    starts = range(0, len(a) - WIN + 1, HOP)
    fa = np.fft.rfft(np.stack([a[s : s + WIN] for s in starts]) * taper)
    fb = np.fft.rfft(np.stack([b[s : s + WIN] for s in starts]) * taper)
    return (
        (np.abs(fa) ** 2).mean(0),
        (np.abs(fb) ** 2).mean(0),
        (np.conj(fa) * fb).mean(0),
        len(starts),
    )


def share(coherent: np.ndarray, total: np.ndarray, mask: np.ndarray) -> float:
    return round(float(coherent[mask].sum() / total[mask].sum()), 4)


def dbfs(power: float) -> float:
    return round(float(10 * np.log10(max(power, 1e-30))), 2)


def main() -> None:
    passes = json.loads((SHEET / "passes.json").read_text())
    with (SHEET / "settings.csv").open() as stream:
        settings = {int(r["order"]): r for r in csv.DictReader(stream)}
    freqs = np.fft.rfftfreq(WIN, 1 / RATE)
    band = (freqs >= BAND[0]) & (freqs < BAND[1])
    width = 3 * RATE / WIN
    hum = np.zeros(freqs.shape, bool)
    for k in range(1, int(BAND[1] // 50) + 1):
        hum |= np.abs(freqs - 50 * k) <= width
    line = np.abs(freqs - 8_000) <= 4 * RATE / WIN
    hum &= ~line  # 8 kHz is also a multiple of 50 Hz: disjoint masks
    plain = band & ~hum & ~line
    rows = []
    for p in sorted(passes, key=lambda p: p["order"]):
        fx_path = MEDIA / f"03-FX Out-{p['file'].removeprefix('02-Looopbck-')}"
        if not fx_path.exists():
            continue
        lo = p["start"] + SILENCE["start"] + RATE // 10
        hi = p["start"] + SILENCE["start"] + SILENCE["samples"] - RATE // 10
        fx = sf.read(fx_path, dtype="float64", start=lo, stop=hi)[0]
        loop = sf.read(MEDIA / p["file"], dtype="float64", start=lo, stop=hi)[0]
        s_ll, s_ff_all, s_lf, count = spectra(loop, fx)
        scale = np.mean(fx**2) / s_ff_all.sum()  # bins in mean-square units
        floor = np.median(s_ff_all[plain])
        coherent = np.abs(s_lf) ** 2 / s_ll  # FX power linearly explained, per bin
        s_ll, s_ff, s_lf = s_ll[band], s_ff_all[band], np.abs(s_lf[band])
        setting = settings[p["order"]]
        row = {
            "order": p["order"],
            "setting": p["setting"],
            "channel": setting["channel"],
            "gain": setting["G"],
            "power_dimension": setting["power_dimension"],
            "fx_noise_fullband_dbfs": round(float(10 * np.log10(np.mean(fx**2))), 2),
            "fx_noise_dbfs": dbfs(s_ff_all[band].sum() * scale),
            "coherent_share": round(float((s_lf**2 / s_ll).sum() / s_ff.sum()), 4),
            "estimator_floor": round(1 / count, 4),
            "coherent_share_broadband": share(coherent, s_ff_all, plain),
            "coherent_share_mains": share(coherent, s_ff_all, band & hum),
            "coherent_share_8k": share(coherent, s_ff_all, line),
            "broadband_dbfs": dbfs(floor * band.sum() * scale),
            "mains_dbfs": dbfs((s_ff_all[band & hum] - floor).sum() * scale),
            "line_8k_dbfs": dbfs((s_ff_all[line] - floor).sum() * scale),
        }
        rows.append(row)
        print(
            f"{row['order']:>3} {row['channel']:5s} G {row['gain']:>4s}"
            f" PD {row['power_dimension']:4s} bruit FX {row['fx_noise_dbfs']:6.1f} dBFS"
            f"  part coherente avec le bouclage {row['coherent_share']:.3f}"
            f" (plancher {row['estimator_floor']:.3f})"
            f"  large bande {row['broadband_dbfs']:6.1f}"
            f"  secteur {row['mains_dbfs']:6.1f}  8 kHz {row['line_8k_dbfs']:6.1f}",
            flush=True,
        )
    (SHEET / "noise.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
