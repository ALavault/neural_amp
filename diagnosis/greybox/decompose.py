#!/usr/bin/env python3
"""Measurements of DIAGNOSIS_greybox.md (section 4) for H2(b), H5 and the two
instrument checks of section 2, plus the probe check that made the planned H1-H3
decomposition unusable. Read-only; renders and the delay co-reading on the GPU like
demo/train_length/coreading_delay.py (fp32, no TF32), so P is reproduced to rounding,
not to the bit (the P reading ran on the CPU).

Instrument: (a) P recomputed from the renders must equal demo/greybox/evaluation.json;
(b) the delay co-reading of coreading_delay.py (rule unchanged) applied to GREY.
Probe check (abandoned H1-H3 measure): nested probes fitted from x to the DEVICE output
of take 1 on the first half of the scored window (A), read on A and on the second half
(B): FIR, Hammerstein 3 and 5, 512 taps, constant term, relative ridge 1e-4 and 1e-2.
H2(b): slope of H2/H1 from -20 to -10 dBFS in
diagnosis/hammerstein/vs100r_hammerstein.json.
H5: excess ln ESR GREY - C per take. Writes diagnosis/greybox/decompose.json.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve
from scipy.signal import fftconvolve, resample_poly
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "demo/train_length"))
import coreading_delay as cd  # noqa: E402
import eval_train_length as tl  # noqa: E402

from fssr_nam.product import vs100r  # noqa: E402
from fssr_nam.product.probes import Probe, normal_equations, regularized  # noqa: E402

ev = tl.ev
FS = 48_000
RUNS = {"C": "vs100r48fast_film_s{}", "GREY": "vs100r48grey_s{}"}
TAPS = 512
CACHE = Path("/fastdata/lavaulta/tmp/greybox_diagnosis")


def windows(n: int) -> tuple[slice, slice]:
    """Halves A and B of the scored window (ev.scored: 0.5 s head, 0.1 s tail)."""
    a, b = FS // 2, n - FS // 10
    middle = (a + b) // 2
    return slice(a, middle), slice(middle, b)


def probe_check(x: np.ndarray, y: np.ndarray) -> dict:
    """Shares of y captured by FIR / Hammerstein 3 / 5 fitted on A, read on A and B."""
    part_a, part_b = windows(len(x))
    powers = [x**k for k in range(1, 6)]
    u = [s / np.sqrt(np.mean(s**2)) for s in powers]
    gram, rhs = normal_equations(Probe(taps=TAPS), [s[part_a] for s in u], y[part_a])
    out = {
        "rms_x": {"A": float(x[part_a].std()), "B": float(x[part_b].std())},
        "peak_x": {"A": float(abs(x[part_a]).max()), "B": float(abs(x[part_b]).max())},
    }
    for ridge in (1e-4, 1e-2):
        for degree in (1, 3, 5):
            width = 1 + degree * TAPS
            probe = Probe(taps=TAPS, ridge=ridge)
            theta = solve(
                regularized(probe, gram[:width, :width]), rhs[:width], assume_a="pos"
            )
            fitted = np.full(len(y), theta[0])
            for k in range(degree):
                fitted += fftconvolve(u[k], theta[1 + k * TAPS : 1 + (k + 1) * TAPS])[
                    : len(y)
                ]
            for name, part in (("A", part_a), ("B", part_b)):
                r = y[part] - fitted[part]
                share = 1 - float(r @ r) / float(y[part] @ y[part])
                out[f"ridge{ridge:g}_degree{degree}_{name}"] = share
    return out


def renders(run_id: str, takes: list[dict], data: dict) -> dict:
    """GPU renders of the P reading inputs, cached outside the repository."""
    path = CACHE / f"{run_id}.npz"
    if path.exists():
        stored = np.load(path)
        return {t["order"]: stored[str(t["order"])] for t in takes}
    model = tl.load(run_id, "last").to(cd.DEVICE)
    out = {}
    for t in takes:
        out[t["order"]] = cd.run(model, data[t["order"]][0], cd.controls(t))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{str(k): v for k, v in out.items()})
    return out


def p_reading(ln_esr: dict, takes: list[dict]) -> float:
    pairs: dict[tuple, list[float]] = {}
    for t in takes:
        if tuple(t["gain_contour"]) != tl.UNSEEN:
            pairs.setdefault(tuple(t["gain_contour"]), []).append(ln_esr[t["order"]])
    return statistics.mean(statistics.mean(v) for v in pairs.values())


def harmonic_slopes() -> dict:
    """Median over bands 200 Hz - 8 kHz, takes (and seeds) of H2/H1 (dB) at -10
    minus at -20 dBFS, for the device and each arm."""
    data = json.loads(
        (ROOT / "diagnosis/hammerstein/vs100r_hammerstein.json").read_text()
    )
    centres = np.array(data["centres"])
    band = (centres >= 200) & (centres <= 8000)

    def ratio(d: list) -> np.ndarray:
        d = np.array(d, dtype=float)
        return (d[1] - d[0])[band]

    device, arms = [], {"C": [], "GREY": []}
    for take in data["takes"].values():
        high, low = take["-10"]["device_d"], take["-20"]["device_d"]
        device.append(np.nanmedian(ratio(high) - ratio(low)))
        for name, model in take["-10"]["models"].items():
            if model["arm"] in arms:
                low = take["-20"]["models"][name]["d"]
                arms[model["arm"]].append(np.nanmedian(ratio(model["d"]) - ratio(low)))
    return {
        "device": float(np.median(device)),
        **{k: float(np.median(v)) for k, v in arms.items()},
        "cases": {"device": len(device), **{k: len(v) for k, v in arms.items()}},
    }


def main() -> None:
    torch.set_num_threads(16)
    takes = [t for t in ev.takes() if not t["low_signal"]]
    data = {
        t["order"]: tuple(resample_poly(s, 1, 4) for s in ev.audio(t)) for t in takes
    }
    reference = json.loads((ROOT / "demo/greybox/evaluation.json").read_text())
    result = {"takes": [t["order"] for t in takes], "runs": {}}
    first = takes[0]["order"]
    result["probe_check"] = {"take": first, **probe_check(*data[first])}
    print(f"probe check: {result['probe_check']}", flush=True)

    for arm, pattern in RUNS.items():
        for seed in tl.SEEDS:
            run_id = pattern.format(seed)
            pred = renders(run_id, takes, data)
            ln_esr = {
                t["order"]: math.log(ev.esr(pred[t["order"]], data[t["order"]][1]))
                for t in takes
            }
            entry = {
                "arm": arm,
                "seed": seed,
                "ln_esr_by_take": ln_esr,
                "s_pairs": p_reading(ln_esr, takes),
                "s_reference": reference["esr"]["s"][run_id],
            }
            result["runs"][run_id] = entry
            print(
                f"{run_id}: s {entry['s_pairs']:.6f} (ref {entry['s_reference']:.6f})",
                flush=True,
            )

    # Instrument (b): delay co-reading for GREY, rule of coreading_delay.py.
    validation = {
        t["order"]: tuple(
            resample_poly(s, 1, 4) for s in vs100r.audio(t, name="di_validation")
        )
        for t in takes
    }
    raw = {t["order"]: ev.audio(t) for t in takes}
    result["delay"] = {}
    for seed in tl.SEEDS:
        run_id = RUNS["GREY"].format(seed)
        model = tl.load(run_id, "last").to(cd.DEVICE)
        tau, _ = cd.common_delay(model, takes, validation)
        scored = cd.score(model, takes, raw, tau)
        result["delay"][run_id] = {
            "tau": tau,
            "grid_edge": tau in (cd.DELAYS[0], cd.DELAYS[-1]),
            "s_pairs": scored["s_pairs"],
        }
        print(f"{run_id}: tau {tau:+.2f} s {scored['s_pairs']:.3f}", flush=True)

    result["harmonic_slopes_db"] = harmonic_slopes()
    excess = {
        t["order"]: statistics.median(
            result["runs"][RUNS["GREY"].format(s)]["ln_esr_by_take"][t["order"]]
            - result["runs"][RUNS["C"].format(s)]["ln_esr_by_take"][t["order"]]
            for s in tl.SEEDS
        )
        for t in takes
    }
    gains = [t["gain_contour"][0] for t in takes]
    contours = [t["gain_contour"][1] for t in takes]
    values = [excess[t["order"]] for t in takes]
    result["excess_by_take"] = {
        "excess": excess,
        "range": max(values) - min(values),
        "spearman_gain": float(spearmanr(gains, values).statistic),
        "spearman_contour": float(spearmanr(contours, values).statistic),
    }
    (HERE / "decompose.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n"
    )
    print("written decompose.json", flush=True)


if __name__ == "__main__":
    main()
