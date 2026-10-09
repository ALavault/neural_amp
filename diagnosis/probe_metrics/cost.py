#!/usr/bin/env python3
"""Computation cost of each score on one fixed condition (NOTE.md).

Coloured noise at -20 dBFS RMS, 4 s estimation and 4 s reading at 48 kHz (WH: 1 s),
gain defect 1.02, seed 300; harmonic signature on a 4 s synchronized sweep at
-10 dBFS; IMD on a two-tone at -10 dBFS. Wall-clock seconds, median of 3 runs, with
the machine load recorded (the machine is shared). Writes cost.json. CPU only.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from fssr_nam.metrics.spectral import multi_resolution_stft_error
from fssr_nam.metrics.time import error_to_signal_ratio
from fssr_nam.product import defect_bench as db
from fssr_nam.product.fidelity import (
    HarmonicProtocol,
    Protocol,
    analyze_pair,
    analyze_sweep_pair,
)
from fssr_nam.product.probes import VolterraLaguerre, WienerHammerstein

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import FS, frozen_probes, imd  # noqa: E402


def timed(function) -> float:
    times = []
    for _ in range(3):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def probes() -> dict:
    out = frozen_probes()
    chosen = json.loads((HERE / "dev_select_v2.json").read_text())
    for order in (2, 3):
        c = chosen[f"volterra_laguerre_{order}"]
        out[f"volterra_laguerre_{order}"] = VolterraLaguerre(
            c["size"], c["pole"], order, c["ridge"]
        )
    out["wiener_hammerstein"] = WienerHammerstein(starts=3, iterations=20)
    return out


def main() -> None:
    base, changed = db.variant("gain", 1.02)
    seconds = {"A": 4.0, "B": 4.0}
    costs = {"load_before": os.getloadavg()[0]}
    for name, probe in probes().items():
        span = 1.0 if name == "wiener_hammerstein" else 4.0
        x = {
            k: db.excitation("noise", -20.0, span, 300 + 50 * i, FS)
            for i, k in enumerate(seconds)
        }
        y_ref = {k: db.apply(base, v, FS) for k, v in x.items()}
        y_eval = {k: db.apply(changed, v, FS) for k, v in x.items()}
        protocol = Protocol(probe=probe, estimation=("A",), reading=("B",))
        costs[name] = timed(
            lambda p=protocol, a=x, r=y_ref, e=y_eval: analyze_pair(a, r, e, FS, p)
        )
    xs, sweep = db.synchronized_sweep(4.0, FS)
    xs = db.at_level(xs, -10.0, "peak")
    ys = db.apply(base, xs, FS), db.apply(changed, xs, FS)
    costs["harmonic_signature"] = timed(
        lambda: analyze_sweep_pair(xs, *ys, FS, sweep, HarmonicProtocol())
    )
    xb = db.excitation("noise", -20.0, 4.0, 350, FS)
    yb = db.apply(base, xb, FS), db.apply(changed, xb, FS)
    costs["esr"] = timed(lambda: error_to_signal_ratio(yb[1], yb[0]))
    costs["mrstft"] = timed(lambda: multi_resolution_stft_error(yb[1], yb[0]))
    xt = db.excitation("two_tone", -10.0, 4.0, 350, FS)
    yt = db.apply(base, xt, FS), db.apply(changed, xt, FS)
    costs["imd"] = timed(lambda: imd(yt[1], yt[0], 350))
    costs["load_after"] = os.getloadavg()[0]
    (HERE / "cost.json").write_text(json.dumps(costs, indent=1) + "\n")
    print(json.dumps(costs, indent=1))


if __name__ == "__main__":
    main()
