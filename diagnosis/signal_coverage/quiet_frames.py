#!/usr/bin/env python3
"""Quiet target frames: arm C against A48 (Q1), share of a constant offset (Q2), and
each model's output on the near-silent input (descriptive). NOTE.md, predictions
written first. Read-only, CPU. Writes quiet_frames.json.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "diagnosis/hammerstein"))
import error_structure as es  # noqa: E402
import vs100r_hammerstein as vh  # noqa: E402

QUIET_DBFS = -40
NOISE = {
    r["order"]: r
    for r in json.loads((es.ROOT / "demo/capture/valvestate/noise.json").read_text())
}


@torch.no_grad()
def model_reading(model, takes: list[dict]) -> dict:
    err_quiet = tgt_quiet = dc_quiet = 0.0
    silence_db, device_db = [], []
    for take in takes:
        loop, fx = es.ev.audio(take)
        target = es.ev.scored(resample_poly(fx, 1, 4))
        controls = torch.tensor([v / 10 for v in take["gain_contour"]])
        y = es.ev.scored(es.ev.run(model, resample_poly(loop, 1, 4), controls))
        tgt = es.frame_energy(target)
        quiet = 10 * np.log10(tgt / es.FRAME + 1e-20) < QUIET_DBFS
        offset = float(y.mean() - target.mean())
        err_quiet += float(es.frame_energy(y - target)[quiet].sum())
        tgt_quiet += float(tgt[quiet].sum())
        dc_quiet += es.FRAME * offset**2 * int(quiet.sum())
        s_loop, _ = vh.section(take, "silence")
        idle = es.ev.run(model, resample_poly(s_loop, 1, 4), controls)[48_000:]
        silence_db.append(10 * np.log10(np.mean(idle**2) + 1e-30))
        device_db.append(NOISE[take["order"]]["fx_noise_fullband_dbfs"])
    return {
        "quiet_esr": err_quiet / tgt_quiet,
        "dc_share_of_quiet_error": dc_quiet / err_quiet,
        "silence_output_dbfs_median": statistics.median(silence_db),
        "device_noise_dbfs_median": statistics.median(device_db),
    }


def main() -> None:
    torch.set_num_threads(16)
    takes = [t for t in es.ev.takes() if not t["low_signal"]]
    runs = {}
    for arm, ids in es.ARMS.items():
        for run in ids:
            runs[run] = {"arm": arm, **model_reading(es.ev.load(run), takes)}
            print(run, runs[run], flush=True)
    pairs = list(zip(es.ARMS["A48"], es.ARMS["C"], strict=True))
    ratios = [runs[c]["quiet_esr"] / runs[a]["quiet_esr"] for a, c in pairs]
    shares = [r["dc_share_of_quiet_error"] for r in runs.values()]
    q2 = statistics.median(shares)
    reading = {
        "q1_ratios_c_over_a48": ratios,
        "Q1": "soutenue"
        if all(r >= 1.5 for r in ratios)
        else "contraire"
        if sum(0.67 <= r <= 1.5 for r in ratios) >= 2
        else "intermédiaire",
        "q2_median_dc_share": q2,
        "Q2": "soutenue" if q2 >= 0.5 else "contraire" if q2 < 0.1 else "intermédiaire",
    }
    print(json.dumps(reading, indent=1, ensure_ascii=False))
    (HERE / "quiet_frames.json").write_text(
        json.dumps({"runs": runs, "reading": reading}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
