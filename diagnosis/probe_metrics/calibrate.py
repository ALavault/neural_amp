#!/usr/bin/env python3
"""Known-defect calibration of the deviation scores (diagnosis/probe_metrics/NOTE.md).

For each excitation family, level and seed: the static (or dynamic) reference chain
and its variant with one defect at one severity, both driven by the same x. Scores on
the reading segment B: probe scores (FIR, Hammerstein 3 and 5, frozen protocol from
dev_select.json), harmonic-response signature on sweeps, and the existing metrics
(ESR, MR-STFT of metrics/spectral.py, intermodulation on two tones). Also the noisy
repeat of the reference (same system, two noise draws at -80 dB) as a floor.

Usage: calibrate.py --phase test [--families noise sweep ...] [--seeds 200 201 202]
Writes calibration_<phase>_<families>.csv (one row per metric x defect x severity x
condition).
CPU only.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

from fssr_nam.metrics.nonlinear import complex_intermodulation_error
from fssr_nam.metrics.spectral import multi_resolution_stft_error
from fssr_nam.metrics.time import error_to_signal_ratio
from fssr_nam.product import defect_bench as db
from fssr_nam.product.fidelity import (
    HarmonicProtocol,
    Protocol,
    analyze_pair,
    analyze_sweep_pair,
)
from fssr_nam.product.probes import Probe, VolterraLaguerre, WienerHammerstein

HERE = Path(__file__).resolve().parent
FS = 48_000.0
SECONDS = 4.0
REPEAT_DB = -80.0  # noise of the "same system, measured twice" floor


def frozen_probes(which: str = "v1") -> dict:
    """v1: FIR, Hammerstein 3 and 5 (dev_select.json); v2: Volterra-Laguerre orders
    2 and 3 (dev_select_v2.json); wh: the fixed Wiener-Hammerstein probe."""
    if which == "wh":
        return {"wiener_hammerstein": WienerHammerstein(starts=3, iterations=20)}
    if which == "v2":
        chosen = json.loads((HERE / "dev_select_v2.json").read_text())
        return {
            f"volterra_laguerre_{o}": VolterraLaguerre(
                chosen[f"volterra_laguerre_{o}"]["size"],
                chosen[f"volterra_laguerre_{o}"]["pole"],
                o,
                chosen[f"volterra_laguerre_{o}"]["ridge"],
            )
            for o in (2, 3)
        }
    chosen = json.loads((HERE / "dev_select.json").read_text())
    degrees = {"fir": 1, "hammerstein_3": 3, "hammerstein_5": 5}
    return {
        name: Probe(degrees[name], chosen[name]["taps"], chosen[name]["ridge"])
        for name in degrees
    }


def pairs():
    """(defect, severity, reference chain, variant chain), the repeat floor first."""
    noisy = replace(db.STATIC, noise_db=REPEAT_DB)
    yield "repeat_floor", REPEAT_DB, noisy, noisy
    for defect, (_, _, severities) in db.DEFECTS.items():
        for severity in severities:
            yield (defect, severity, *db.variant(defect, severity))


def probe_rows(probes, x, y_ref, y_eval) -> list[dict]:
    rows = []
    for name, probe in probes.items():
        protocol = Protocol(probe=probe, estimation=("A",), reading=("B",))
        out = analyze_pair(x, y_ref, y_eval, FS, protocol)["readings"]["B"]
        rows.append(
            {
                "metric": name,
                "value": out["d_probe"],
                "normalized": out["d_norm"],
                "r_ref": out["r_ref"] / max(out["rms_ref"], protocol.floor),
                "r_eval": out["r_eval"] / max(out["rms_ref"], protocol.floor),
                "unrepresented": out["unrepresented"]
                / max(out["rms_ref"], protocol.floor),
                "bound_holds": out["bound"]["holds"],
                "d_audio_norm": out["d_audio_norm"],
                "start_spread": max(out["start_spread_ref"], out["start_spread_eval"])
                / max(out["rms_ref"], protocol.floor),
            }
        )
    return rows


def baseline_rows(family: str, xb, ref, ev, seed: int) -> list[dict]:
    skip = round(0.05 * FS)
    ref, ev = ref[skip:], ev[skip:]
    rows = [
        {"metric": "esr", "value": error_to_signal_ratio(ev, ref)},
        {"metric": "mrstft", "value": multi_resolution_stft_error(ev, ref)},
    ]
    if family == "two_tone":
        rows.append({"metric": "imd", "value": imd(ev, ref, seed + 50)})
    return rows


def imd(ev, ref, seed: int) -> float:
    """Complex error on the two tones and their products up to third order."""
    f1, f2 = db.two_tone_frequencies(seed)
    lines = [f1, f2, f2 - f1, 2 * f1 - f2, 2 * f2 - f1, f1 + f2, 2 * f1, 2 * f2]
    lines = sorted({round(f, 6) for f in lines if 0 < f < FS / 2})
    return complex_intermodulation_error(
        ev, ref, frequencies_hz=lines, sample_rate=int(FS)
    )


def sweep_rows(level: float, seed: int, base, changed) -> list[dict]:
    x, sweep = db.synchronized_sweep(SECONDS * (1 + 0.1 * ((seed + 50) % 3)), FS)
    x = db.at_level(x, level, "peak")
    out = analyze_sweep_pair(
        x,
        db.apply(base, x, FS, 1),
        db.apply(changed, x, FS, 2),
        FS,
        sweep,
        HarmonicProtocol(),
    )
    rows = [{"metric": "harmonic_total", "value": out["total_complex_rel"]}]
    for k in range(1, 6):
        entry = out[f"order_{k}"]
        rows.append(
            {
                "metric": f"harmonic_order_{k}",
                "value": entry["complex_rel"],
                "bins_only_eval": entry["bins_only_eval"],
            }
        )
    return rows


def condition(
    family: str, level: float, seed: int, probes, seconds=SECONDS, baselines=True
) -> list[dict]:
    x = {
        "A": db.excitation(family, level, seconds, 0 if family == "di" else seed, FS),
        "B": db.excitation(
            family, level, seconds, 1 if family == "di" else seed + 50, FS
        ),
    }
    rows = []
    for defect, severity, base, changed in pairs():
        y_ref = {k: db.apply(base, v, FS, seed=1) for k, v in x.items()}
        y_eval = {k: db.apply(changed, v, FS, seed=2) for k, v in x.items()}
        found = probe_rows(probes, x, y_ref, y_eval)
        if baselines:
            found += baseline_rows(family, x["B"], y_ref["B"], y_eval["B"], seed)
        if baselines and family == "sweep":
            found += sweep_rows(level, seed, base, changed)
        for row in found:
            row.update(
                family=family,
                level_db=level,
                seed=seed,
                defect=defect,
                severity=severity,
            )
        rows += found
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--families", nargs="+", default=list(db.LEVEL_KIND))
    parser.add_argument("--seeds", nargs="+", type=int, default=[200, 201, 202])
    parser.add_argument("--levels", nargs="+", type=float, default=None)
    parser.add_argument("--probes", default="v1", choices=["v1", "v2", "wh"])
    parser.add_argument("--seconds", type=float, default=SECONDS)
    args = parser.parse_args()
    probes = frozen_probes(args.probes)
    rows = []
    for family in args.families:
        seeds = [0] if family == "di" else args.seeds
        for level in args.levels or db.LEVELS_DB[db.LEVEL_KIND[family]]:
            for seed in seeds:
                start = time.time()
                rows += condition(
                    family, level, seed, probes, args.seconds, args.probes == "v1"
                )
                print(
                    f"{family} {level} {seed}: {time.time() - start:.0f}s", flush=True
                )
    fields = sorted({k for row in rows for k in row})
    tag = "" if args.probes == "v1" else f"_{args.probes}"
    name = f"calibration_{args.phase}{tag}_{'_'.join(args.families)}.csv"
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
