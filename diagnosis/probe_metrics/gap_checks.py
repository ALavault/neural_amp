#!/usr/bin/env python3
"""Post hoc checks of two real gaps, before any ad hoc metric (NOTE.md).

jvm: why does Hammerstein 5 leave ~0.75 of the JVM410H differences unrepresented?
  - capacity on the reference itself: r_ref / RMS(y_ref) of H5 (256 taps) at the centre;
  - on three ladder pairs: H5 with 1024 taps (memory), Volterra-Laguerre 3 (frozen v2;
    interactions), and the coherence between x and the unrepresented difference
    (near 0: noise-like or not linearly tied to x).
sweep: why does the harmonic signature stop separating at -3 dBFS on the VS100R?
  orders 1-9 on one repeat pair and one gain pair, at -10 and -3 dBFS.
Writes gap_checks_<which>.json. CPU only. Post hoc: diagnosis, not a protocol change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import coherence

from fssr_nam.product.fidelity import (
    HarmonicProtocol,
    Protocol,
    analyze_pair,
    analyze_sweep_pair,
)
from fssr_nam.product.probes import Probe, fit, render

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import frozen_probes  # noqa: E402
from real_pairs import SWEEPS, common_section, jvm_audio  # noqa: E402

PAIRS = ("B050_M050_T050_G040", "B080_M050_T050_G050", "B050_M050_T020_G050")


def capacity(probe: Probe, rate: int, x: dict, y: dict) -> float:
    q = render(fit(probe, x["A"], y["A"]), x["B"])
    skip = round(0.05 * rate)
    return float(
        np.sqrt(np.mean((y["B"] - q)[skip:] ** 2) / np.mean(y["B"][skip:] ** 2))
    )


def pair_check(rate, x, y_ref, y_eval, probes) -> dict:
    out = {}
    for name, probe in probes.items():
        protocol = Protocol(probe=probe, estimation=("A",), reading=("B",))
        reading = analyze_pair(x, y_ref, y_eval, rate, protocol)["readings"]["B"]
        out[name] = {
            "captured": reading["d_probe"] / reading["d_audio"],
            "unrepresented": reading["unrepresented"] / reading["d_audio"],
        }
    skip = round(0.05 * rate)
    diff = (y_eval["B"] - y_ref["B"])[skip:]
    freqs, coh = coherence(x["B"][skip:], diff, fs=rate, nperseg=8192)
    band = (freqs > 80) & (freqs < 8000)
    out["coherence_x_diff_median_80_8k"] = float(np.median(coh[band]))
    return out


def jvm() -> dict:
    v2 = frozen_probes("v2")
    probes = {
        "hammerstein_5": frozen_probes("v1")["hammerstein_5"],
        "hammerstein_5_1024": Probe(5, 1024, 1e-4),
        "volterra_laguerre_3": v2["volterra_laguerre_3"],
    }
    rate, x, y_ref = jvm_audio("PreampOut", "B050_M050_T050_G050")
    out = {
        "capacity_centre": {
            "hammerstein_5": capacity(probes["hammerstein_5"], rate, x, y_ref),
            "hammerstein_5_1024": capacity(
                probes["hammerstein_5_1024"], rate, x, y_ref
            ),
        }
    }
    for setting in PAIRS:
        _, _, y_eval = jvm_audio("PreampOut", setting)
        out[setting] = pair_check(rate, x, y_ref, y_eval, probes)
        print(setting, flush=True)
    return out


def sweep() -> dict:
    from fssr_nam.product import vs100r
    from fssr_nam.product.hammerstein import Sweep

    sheet = {t["order"]: t for t in vs100r.sheet()}
    sweep_def = Sweep(f0=20.0, f1=20_000.0, seconds=5.0)
    protocol = HarmonicProtocol(order=9)
    out = {}
    for ref, ev, label in ((20, 39, "repeat"), (39, 38, "gain_7.5")):
        for name in ("sweep_-10dBFS", "sweep_-3dBFS"):
            s = common_section(sheet[ref], sheet[ev], name)
            result = analyze_sweep_pair(
                s["x"],
                s["y_ref"],
                s["y_eval"],
                48_000.0,
                sweep_def,
                protocol,
                s["x_eval"],
            )
            out[f"{label}|{name}"] = {
                k: {"complex_rel": v["complex_rel"], "level_ref_db": v["level_ref_db"]}
                for k, v in result.items()
                if k.startswith("order_")
            }
    assert set(SWEEPS) >= {"sweep_-10dBFS", "sweep_-3dBFS"}
    return out


def main() -> None:
    which = sys.argv[1]
    result = jvm() if which == "jvm" else sweep()
    (HERE / f"gap_checks_{which}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
