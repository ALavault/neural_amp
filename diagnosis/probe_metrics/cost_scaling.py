#!/usr/bin/env python3
"""How the cost of each score grows with the data length (NOTE.md, usages vs cost).

Coloured noise at -20 dBFS RMS, gain defect 1.02, estimation = reading = T seconds,
T in 2, 8, 32 s (Wiener-Hammerstein: 1, 2, 4 s). Wall-clock seconds of one
analyze_pair (or of the plain metric), median of 2 runs, and the fitted exponent of
time ~ T^k. The machine is shared: the load is recorded. Writes cost_scaling.json.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from fssr_nam.metrics.spectral import multi_resolution_stft_error
from fssr_nam.metrics.time import error_to_signal_ratio
from fssr_nam.product import defect_bench as db
from fssr_nam.product.fidelity import Protocol, analyze_pair

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import FS  # noqa: E402
from cost import probes  # noqa: E402


def timed(function) -> float:
    times = []
    for _ in range(2):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def case(seconds: float):
    base, changed = db.variant("gain", 1.02)
    x = {
        k: db.excitation("noise", -20.0, seconds, 300 + 50 * i, FS)
        for i, k in enumerate(("A", "B"))
    }
    return (
        x,
        {k: db.apply(base, v, FS) for k, v in x.items()},
        {k: db.apply(changed, v, FS) for k, v in x.items()},
    )


def exponent(points: dict) -> float:
    (t0, c0), (t1, c1) = min(points.items()), max(points.items())
    return math.log(c1 / c0) / math.log(t1 / t0)


def main() -> None:
    out = {"load_before": os.getloadavg()[0], "metrics": {}}
    for name, probe in probes().items():
        durations = (
            (1.0, 2.0, 4.0) if name == "wiener_hammerstein" else (2.0, 8.0, 32.0)
        )
        points = {}
        for seconds in durations:
            x, y_ref, y_eval = case(seconds)
            protocol = Protocol(probe=probe, estimation=("A",), reading=("B",))
            points[seconds] = timed(
                lambda a=x, r=y_ref, e=y_eval, p=protocol: analyze_pair(a, r, e, FS, p)
            )
        out["metrics"][name] = {"seconds": points, "exponent": exponent(points)}
        print(name, points, flush=True)
    for name, metric in (
        ("esr", error_to_signal_ratio),
        ("mrstft", multi_resolution_stft_error),
    ):
        points = {}
        for seconds in (2.0, 8.0, 32.0):
            _, y_ref, y_eval = case(seconds)
            points[seconds] = timed(
                lambda m=metric, e=y_eval["B"], r=y_ref["B"]: m(e, r)
            )
        out["metrics"][name] = {"seconds": points, "exponent": exponent(points)}
    out["load_after"] = os.getloadavg()[0]
    (HERE / "cost_scaling.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
