#!/usr/bin/env python3
"""Does the device's own noise explain where the models' error is high? (NOTE.md, test
of the device noise, prediction written first.) Read-only, CPU.

Frames of 10 ms of the target, classed by target level (5 dB). Per class: observed ESR
and the noise floor (1 - coherent share) x noise power / mean target power, from
demo/capture/valvestate/noise.json per take. Writes noise_share.json.
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
import error_structure as es  # noqa: E402

RUNS = ["vs100r_gc_film_joint_s925_r2", "vs100r48fast_film_s42"]
NOISE = {
    r["order"]: r
    for r in json.loads((es.ROOT / "demo/capture/valvestate/noise.json").read_text())
}
EDGES = np.arange(-70, 0, 5)


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    takes = [t for t in es.ev.takes() if not t["low_signal"]]
    result = {"edges": EDGES.tolist(), "runs": {}}
    for run in RUNS:
        model = es.ev.load(run)
        tgt_all, err_all, noise_all, level_all, since_all, mid_all = (
            [] for _ in range(6)
        )
        for take in takes:
            loop, fx = es.ev.audio(take)
            x = resample_poly(loop, 1, 4)
            target = es.ev.scored(resample_poly(fx, 1, 4))
            controls = torch.tensor([v / 10 for v in take["gain_contour"]])
            y = es.ev.scored(es.ev.run(model, x, controls))
            tgt = es.frame_energy(target)
            noise = NOISE[take["order"]]
            power = 10 ** (noise["fx_noise_fullband_dbfs"] / 10)
            unpredictable = (1 - noise["coherent_share"]) * power * es.FRAME
            lv = es.level_db(es.ev.scored(x))
            lo, hi = np.percentile(lv, [30, 70])
            tgt_all.append(tgt)
            err_all.append(es.frame_energy(y - target))
            noise_all.append(np.full(len(tgt), unpredictable))
            level_all.append(10 * np.log10(tgt / es.FRAME + 1e-20))
            since_all.append(es.since_last_loud(lv, np.percentile(lv, 90)))
            mid_all.append((lv >= lo) & (lv <= hi))
        tgt, err, noise, level, since, mid = (
            np.concatenate(a)
            for a in (tgt_all, err_all, noise_all, level_all, since_all, mid_all)
        )
        cls = np.digitize(level, EDGES) - 1
        classes = []
        for k in range(len(EDGES) - 1):
            sel = cls == k
            if sel.sum() < 200:
                continue
            classes.append(
                {
                    "from_dbfs": int(EDGES[k]),
                    "frames": int(sel.sum()),
                    "esr": float(err[sel].sum() / tgt[sel].sum()),
                    "esr_noise": float(noise[sel].sum() / tgt[sel].sum()),
                }
            )
        high = [c for c in classes if c["esr"] > 0.02]
        share = (
            statistics.median(c["esr_noise"] / c["esr"] for c in high) if high else None
        )
        near = mid & (since >= 50) & (since < 200)
        far = mid & (since >= 1000)
        result["runs"][run] = {
            "classes": classes,
            "median_noise_share_where_esr_above_0.02": share,
            "o2_target_level_near_minus_far_db": float(
                10 * np.log10(tgt[near].mean() / tgt[far].mean())
            ),
        }
        print(run, share, flush=True)
    shares = [
        r["median_noise_share_where_esr_above_0.02"]
        for r in result["runs"].values()
        if r["median_noise_share_where_esr_above_0.02"] is not None
    ]
    median = statistics.median(shares)
    result["reading"] = {
        "median_share": median,
        "H_bruit": "soutenue"
        if median >= 0.5
        else "contraire"
        if median < 0.1
        else "intermédiaire",
    }
    print(json.dumps(result["reading"], ensure_ascii=False))
    (HERE / "noise_share.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
