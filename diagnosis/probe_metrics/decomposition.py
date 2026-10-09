#!/usr/bin/env python3
"""Decomposition of the probe deviation by branch and by octave (NOTE.md).

One fixed condition per excitation family given on the command line (default: coloured
noise at -20 dBFS RMS, seed 200 / reading 250), every defect at its strongest
severity. For Hammerstein 5 (frozen v1): RMS of the deviation of each branch x^k (and
of the constant), relative to d_probe; for every probe: the octave bands of the
deviation and of the raw output difference. Branches are a decomposition in the
monomial basis, conditioned by that basis and by the excitation: not a physical
decomposition of the distortion. Writes decomposition_<family>.json. CPU.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from fssr_nam.product import defect_bench as db
from fssr_nam.product.fidelity import Protocol, analyze_pair

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import FS, SECONDS, frozen_probes  # noqa: E402

LEVEL = {"noise": -20.0, "plucks": -10.0, "sweep": -10.0, "multisine": -20.0}


def main() -> None:
    family = sys.argv[1] if len(sys.argv) > 1 else "noise"
    x = {
        "A": db.excitation(family, LEVEL[family], SECONDS, 200, FS),
        "B": db.excitation(family, LEVEL[family], SECONDS, 250, FS),
    }
    probes = frozen_probes("v1")
    out = {"family": family, "level_db": LEVEL[family], "defects": {}}
    for defect, (_, _, severities) in db.DEFECTS.items():
        severity = max(severities)
        base, changed = db.variant(defect, severity)
        y_ref = {k: db.apply(base, v, FS, seed=1) for k, v in x.items()}
        y_eval = {k: db.apply(changed, v, FS, seed=2) for k, v in x.items()}
        entry = {"severity": severity}
        for name, probe in probes.items():
            protocol = Protocol(probe=probe, estimation=("A",), reading=("B",))
            reading = analyze_pair(x, y_ref, y_eval, FS, protocol)["readings"]["B"]
            total = reading["d_probe"] or math.inf
            entry[name] = {
                "d_probe_norm": reading["d_norm"],
                "d_audio_norm": reading["d_audio_norm"],
                "branch_share": {k: v / total for k, v in reading["by_branch"].items()},
                "bands_probe": reading["by_band"]["probe"],
                "bands_audio": reading["by_band"]["audio"],
            }
        out["defects"][defect] = entry
        print(defect, flush=True)
    out["edges_hz"] = list(
        Protocol(probe=probes["fir"], estimation=(), reading=()).edges_hz
    )
    (HERE / f"decomposition_{family}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
