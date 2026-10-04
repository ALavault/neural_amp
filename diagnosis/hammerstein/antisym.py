#!/usr/bin/env python3
"""Candidate 1 of NOTE.md: odd part of existing models at inference,
y_odd = (f(x) - f(-x)) / 2. ESR on the 8 VS100R test takes (scored region, 48 kHz)
of f(x), y_odd and the even part alone, for the A48 and B192 runs; control: D2 of y_odd
on take 17's -10 dBFS sweep. Read-only, CPU. Writes antisym.json.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vs100r_hammerstein as vh  # noqa: E402

from fssr_nam.product.hammerstein import deconvolve, harmonic_spectra  # noqa: E402

ev = vh.ev


def outputs(
    model, arm: str, loop: np.ndarray, controls
) -> tuple[np.ndarray, np.ndarray]:
    plus = vh.model_output(model, arm, loop, controls)
    minus = vh.model_output(model, arm, -loop, controls)
    return plus, minus


def d2_rel_db(y: np.ndarray, ref: np.ndarray) -> float:
    d = harmonic_spectra(
        deconvolve(y, ref, vh.RATE, vh.SWEEP), vh.RATE, vh.SWEEP, 2, vh.NFFT
    )
    f = np.fft.rfftfreq(vh.NFFT, 1 / vh.RATE)
    band = (f >= 200) & (f <= 8000)
    return float(
        10 * np.log10((np.abs(d[1][band]) ** 2).sum() / (np.abs(d[0][band]) ** 2).sum())
    )


def main() -> None:
    torch.set_num_threads(16)
    takes = [t for t in ev.takes() if t["split"] == "test" and not t["low_signal"]]
    data = {t["order"]: ev.audio(t) for t in takes}
    t17 = next(t for t in ev.takes() if t["order"] == 17)
    sweep_loop, sweep_fx = vh.section(t17, "sweep_-10dBFS")
    result = {
        "device_d2_take17_m10": d2_rel_db(
            vh.decimate(sweep_fx), vh.decimate(sweep_loop)
        ),
        "runs": {},
    }
    for arm in ("A48", "B192"):
        for run in vh.RUNS[arm]:
            model = ev.load(run)
            rows = {}
            for take in takes:
                loop, fx = data[take["order"]]
                real = resample_poly(fx, 1, 4)
                controls = [v / 10 for v in take["gain_contour"]]
                plus, minus = outputs(model, arm, loop, controls)
                odd, even = (plus - minus) / 2, (plus + minus) / 2
                rows[take["order"]] = {
                    "esr": ev.esr(plus, real),
                    "esr_odd": ev.esr(odd, real),
                    "even_over_odd_db": float(
                        10
                        * np.log10(
                            (ev.scored(even - even.mean()) ** 2).sum()
                            / (ev.scored(odd) ** 2).sum()
                        )
                    ),
                }
            c17 = [v / 10 for v in t17["gain_contour"]]
            p, m = outputs(model, arm, sweep_loop, c17)
            ref = vh.decimate(sweep_loop)
            s = statistics.mean(math.log(r["esr"]) for r in rows.values())
            s_odd = statistics.mean(math.log(r["esr_odd"]) for r in rows.values())
            result["runs"][run] = {
                "arm": arm,
                "takes": rows,
                "s": s,
                "s_odd": s_odd,
                "delta": s_odd - s,
                "d2_model": d2_rel_db(p, ref),
                "d2_odd": d2_rel_db((p - m) / 2, ref),
            }
            print(
                run,
                "Δ ln ESR %.3f" % (s_odd - s),
                "D2 modèle %.1f, impair %.1f dB"
                % (result["runs"][run]["d2_model"], result["runs"][run]["d2_odd"]),
                flush=True,
            )
    deltas = [r["delta"] for r in result["runs"].values()]
    result["reading"] = {
        "median_delta": statistics.median(deltas),
        "n_lower": sum(d < 0 for d in deltas),
        "verdict": "l'antisymétrisation aide"
        if statistics.median(deltas) <= -0.05 and sum(d < 0 for d in deltas) >= 5
        else "n'aide pas",
    }
    print(json.dumps(result["reading"], ensure_ascii=False))
    (HERE / "antisym.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
