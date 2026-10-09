#!/usr/bin/env python3
"""Existing repository metrics added to the known-defect bench (post hoc, NOTE.md).

magnitude_error and phase_error (metrics/spectral.py: STFT magnitude and target-weighted
wrapped phase, 4096 points) separate what MR-STFT mixes; envelope_error
(metrics/transient.py, causal envelope 2 ms / 50 ms) targets envelope dynamics. Same
conditions, seeds, defects, severities and repeat floor as calibrate.py, same reading
rules in summarize.py. No probe is fitted. Writes calibration_test_extra_<families>.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fssr_nam.metrics.spectral import magnitude_error, phase_error
from fssr_nam.metrics.transient import envelope_error
from fssr_nam.product import defect_bench as db

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from calibrate import FS, SECONDS, pairs  # noqa: E402

SKIP = round(0.05 * FS)


def rows_for(family: str, level: float, seed: int) -> list[dict]:
    xb = db.excitation(family, level, SECONDS, 1 if family == "di" else seed + 50, FS)
    out = []
    for defect, severity, base, changed in pairs():
        ref = db.apply(base, xb, FS, seed=1)[SKIP:]
        ev = db.apply(changed, xb, FS, seed=2)[SKIP:]
        for name, value in (
            ("magnitude", magnitude_error(ev, ref)),
            ("phase", phase_error(ev, ref)),
            ("envelope", envelope_error(ev, ref, sample_rate=int(FS))),
        ):
            out.append(
                {
                    "metric": name,
                    "value": value,
                    "family": family,
                    "level_db": level,
                    "seed": seed,
                    "defect": defect,
                    "severity": severity,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=list(db.LEVEL_KIND))
    args = parser.parse_args()
    rows = []
    for family in args.families:
        seeds = [0] if family == "di" else [200, 201, 202]
        for level in db.LEVELS_DB[db.LEVEL_KIND[family]]:
            for seed in seeds:
                rows += rows_for(family, level, seed)
            print(family, level, flush=True)
    name = f"calibration_test_extra_{'_'.join(args.families)}.csv"
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
