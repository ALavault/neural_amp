#!/usr/bin/env python3
"""Group delay of the VS100R and of its model, band by band. (Product lane. POST HOC.)

Follows timing_shift.py: the model at 48 kHz is about one sample early on the device.
Readings fixed before this ran:
- model-to-device delay about +1 sample in every octave band (within +-0.3): a global
  offset (the model misses one sample of pure delay, or an alignment step shifts);
- a delay that changes across bands: the model misrenders the device's own group delay
  (dispersion), not a pure latency.

Per OD2 take that is not low-signal, on the test section at 48 kHz (as
product_vs100r_eval.py): Welch cross-spectra (Hann, 4096, half overlap) between the
loopback x, the device output y and the model output m. In each octave band, the delay
in samples is minus the slope of the unwrapped phase against angular frequency, fitted
with |cross-spectrum| weights, over bins whose coherence is at least 0.5:
- device: x -> y; model: x -> m; model to device: m -> y (positive: the device lags the
  model, i.e. the model is early).

Writes demo/capture/valvestate/group_delay.json.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import csd, resample_poly, welch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import product_vs100r_eval as ev  # noqa: E402

RUN = "vs100r_gc_film_joint_s925_r2"
RATE = 48_000
NPERSEG = 4096
CENTRES = (63, 125, 250, 500, 1_000, 2_000, 4_000, 8_000)


def band_delays(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    """Delay of b relative to a, in samples, per octave band."""
    freqs, s_ab = csd(a, b, RATE, nperseg=NPERSEG)
    _, s_aa = welch(a, RATE, nperseg=NPERSEG)
    _, s_bb = welch(b, RATE, nperseg=NPERSEG)
    coherence = np.abs(s_ab) ** 2 / (s_aa * s_bb + 1e-30)
    out = {}
    for centre in CENTRES:
        band = (freqs >= centre / 2**0.5) & (freqs < centre * 2**0.5)
        keep = band & (coherence >= 0.5)
        if keep.sum() < 4:
            out[str(centre)] = None
            continue
        omega = 2 * np.pi * freqs[keep] / RATE
        # scipy's csd is conj(A) * B: its phase is that of B / A, -omega * delay.
        phase = np.unwrap(np.angle(s_ab[keep]))
        weight = np.abs(s_ab[keep])
        slope = np.polyfit(omega, phase, 1, w=weight)[0]
        out[str(centre)] = float(-slope)
    return out


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    model = ev.load(RUN)
    rows = []
    for take in ev.takes():
        if take["low_signal"]:
            continue
        loop, fx = ev.audio(take)
        x, y = resample_poly(loop, 1, 4), resample_poly(fx, 1, 4)
        controls = torch.tensor([v / 10 for v in take["gain_contour"]])
        m = ev.run(model, x, controls)
        x, y, m = ev.scored(x), ev.scored(y), ev.scored(m)
        rows.append(
            {
                "order": take["order"],
                "gain_contour": take["gain_contour"],
                "device": band_delays(x, y),
                "model": band_delays(x, m),
                "model_to_device": band_delays(m, y),
            }
        )
        print(
            f"prise {take['order']:>2} {take['gain_contour']}: modele -> appareil "
            + " ".join(
                f"{c}:{v:+.2f}" if v is not None else f"{c}:--"
                for c, v in rows[-1]["model_to_device"].items()
            ),
            flush=True,
        )
    summary = {}
    for kind in ("device", "model", "model_to_device"):
        summary[kind] = {}
        for centre in map(str, CENTRES):
            values = [r[kind][centre] for r in rows if r[kind][centre] is not None]
            summary[kind][centre] = (
                statistics.median(values) if values else None,
                len(values),
            )
    print(json.dumps(summary, indent=1))
    Path(__file__).with_name("group_delay.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
