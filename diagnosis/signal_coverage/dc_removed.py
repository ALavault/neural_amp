#!/usr/bin/env python3
"""Readings without a constant component (NOTE.md, predictions written first): the same
causal one-pole high-pass (R = 0.9995) on output and target, then the quiet-frame ESR
(D1) and reading P (D2), raw and filtered. Read-only, CPU. Writes dc_removed.json.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import lfilter, resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import error_structure as es  # noqa: E402

R = 0.9995
UNSEEN = (2.5, 7.5)


def dc_block(x: np.ndarray) -> np.ndarray:
    """y[n] = x[n] - x[n-1] + R y[n-1], from a zero state."""
    return lfilter([1.0, -1.0], [1.0, -R], x)


@torch.no_grad()
def model_reading(model, takes: list[dict]) -> dict:
    out = {}
    for name in ("raw", "filtered"):
        out[name] = {"err_quiet": 0.0, "tgt_quiet": 0.0, "ln_esr": {}}
    for take in takes:
        loop, fx = es.ev.audio(take)
        target = resample_poly(fx, 1, 4)
        controls = torch.tensor([v / 10 for v in take["gain_contour"]])
        y = es.ev.run(model, resample_poly(loop, 1, 4), controls)
        for name, filt in (("raw", lambda v: v), ("filtered", dc_block)):
            yt, tt = es.ev.scored(filt(y)), es.ev.scored(filt(target))
            tgt = es.frame_energy(tt)
            quiet = 10 * np.log10(tgt / es.FRAME + 1e-20) < -40
            o = out[name]
            o["err_quiet"] += float(es.frame_energy(yt - tt)[quiet].sum())
            o["tgt_quiet"] += float(tgt[quiet].sum())
            o["ln_esr"][take["order"]] = math.log(
                float(((tt - yt) ** 2).sum() / (tt**2).sum())
            )
    pair = {t["order"]: tuple(t["gain_contour"]) for t in takes}
    for o in out.values():
        o["quiet_esr"] = o.pop("err_quiet") / o.pop("tgt_quiet")
        groups: dict[tuple, list[float]] = {}
        for order, value in o["ln_esr"].items():
            if pair[order] != UNSEEN:
                groups.setdefault(pair[order], []).append(value)
        o["s_pairs"] = statistics.mean(statistics.mean(v) for v in groups.values())
    return out


def main() -> None:
    torch.set_num_threads(16)
    takes = [t for t in es.ev.takes() if not t["low_signal"]]
    runs = {}
    for ids in es.ARMS.values():
        for run in ids:
            runs[run] = model_reading(es.ev.load(run), takes)
            print(
                run,
                {k: (v["quiet_esr"], v["s_pairs"]) for k, v in runs[run].items()},
                flush=True,
            )
    pairs = list(zip(es.ARMS["A48"], es.ARMS["C"], strict=True))
    reading = {}
    for name in ("raw", "filtered"):
        reading[name] = {
            "quiet_ratio_c_over_a48": [
                runs[c][name]["quiet_esr"] / runs[a][name]["quiet_esr"]
                for a, c in pairs
            ],
            "p_gap_c_minus_a48": statistics.mean(
                runs[c][name]["s_pairs"] - runs[a][name]["s_pairs"] for a, c in pairs
            ),
        }
    ratios = reading["filtered"]["quiet_ratio_c_over_a48"]
    change = (
        reading["filtered"]["p_gap_c_minus_a48"] - reading["raw"]["p_gap_c_minus_a48"]
    )
    reading["D1"] = (
        "soutenue"
        if sum(0.67 <= r <= 1.5 for r in ratios) >= 2
        else "contraire"
        if sum(r > 1.5 for r in ratios) >= 2
        else "intermédiaire"
    )
    reading["D2_change"] = change
    reading["D2"] = (
        "soutenue"
        if abs(change) < 0.02
        else "contraire"
        if abs(change) > 0.05
        else "intermédiaire"
    )
    print(json.dumps(reading, indent=1, ensure_ascii=False))
    (HERE / "dc_removed.json").write_text(
        json.dumps({"runs": runs, "reading": reading}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
