#!/usr/bin/env python3
"""Delay-compensated co-reading of demo/train_length/predictions.md. Read-only, CPU.

Rule frozen in 6f62341 before the P reading, run after it. Per run, checkpoint last:
one common delay tau on the DELAYS grid, the one that minimises the median ESR over the
takes on their di_validation section; applied unchanged to the model output on di_test;
same aggregates, labels and acceptance bound as P, descriptive. Instrument checks: edge
effect of the shift in the scored region, delays on a grid edge. eval_train_length.py
and its JSON are not modified. Writes eval_train_length_aligned.json.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_train_length as tl  # noqa: E402

from fssr_nam.product import vs100r  # noqa: E402

ev = tl.ev
DELAYS = np.round(np.arange(-2.0, 3.0001, 0.05), 2)
PAD = 4_800  # 0.1 s of zeros on each side at 48 kHz, against circular wrap-around
EDGE_DELAY = 0.8
MARGIN = vs100r.RATE  # 1 s of context at the capture rate
# Inference on the GPU (fp32, no TF32), unlike P (CPU): every run of the co-reading
# goes through the same device; parity with the CPU in predictions.md (Résultats).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Shifter:
    """Fractional delays of one signal: timing_shift.delayed() on the zero-padded
    signal, cropped back, with the spectrum computed once."""

    def __init__(self, x: np.ndarray) -> None:
        self.n = len(x) + 2 * PAD
        self.spectrum = np.fft.rfft(np.pad(x, PAD))
        self.freqs = np.fft.rfftfreq(self.n)

    def __call__(self, d: float) -> np.ndarray:
        ramp = np.exp(-2j * np.pi * self.freqs * d)
        return np.fft.irfft(self.spectrum * ramp, self.n)[PAD:-PAD]


def controls(take: dict) -> torch.Tensor:
    return torch.tensor([v / 10 for v in take["gain_contour"]])


@torch.no_grad()
def run(model, x: np.ndarray, knobs: torch.Tensor) -> np.ndarray:
    """ev.run on DEVICE."""
    model.reset_states()
    signal = torch.from_numpy(x.astype(np.float32))[None, None].to(DEVICE)
    out = model(signal, knobs[None].to(DEVICE))
    return out.cpu().numpy()[0, 0].astype(np.float64)


def common_delay(model, takes: list[dict], validation: dict) -> tuple[float, list]:
    """Grid delay minimising the median ESR over the takes on di_validation."""
    curves = []
    for take in takes:
        x, y = validation[take["order"]]
        shift = Shifter(run(model, x, controls(take)))
        curves.append([ev.esr(shift(d), y) for d in DELAYS])
    medians = np.median(np.array(curves), axis=0)
    return float(DELAYS[int(np.argmin(medians))]), medians.tolist()


def score(model, takes: list[dict], data: dict, tau: float) -> dict:
    """tl.score with the model output delayed by tau before scoring."""
    ln_esr = {}
    for take in takes:
        loop, fx = data[take["order"]]
        y = Shifter(run(model, resample_poly(loop, 1, 4), controls(take)))(tau)
        ln_esr[take["order"]] = math.log(ev.esr(y, resample_poly(fx, 1, 4)))
    pairs: dict[tuple, list[float]] = {}
    for take in takes:
        if tuple(take["gain_contour"]) != tl.UNSEEN:
            pairs.setdefault(tuple(take["gain_contour"]), []).append(
                ln_esr[take["order"]]
            )
    return {
        "ln_esr_by_take": ln_esr,
        "s_pairs": statistics.mean(statistics.mean(v) for v in pairs.values()),
        "n_pairs": len(pairs),
        "s_test8": statistics.mean(
            ln_esr[t["order"]] for t in takes if t["split"] == "test"
        ),
        "take16": ln_esr.get(16),
    }


def edge_effect(take: dict) -> float:
    """Energy of the difference, in the scored region, between the FX output of di_test
    delayed alone and delayed with 1 s of context on each side, over the signal's."""
    part = vs100r.section("di_test")
    p = take["pass"]
    a = p["start"] + part["start"] - MARGIN
    stamp = p["file"].removeprefix("02-Looopbck-")
    ext = sf.read(
        vs100r.MEDIA / f"03-FX Out-{stamp}",
        dtype="float64",
        start=a,
        stop=a + part["samples"] + 2 * MARGIN,
    )[0]
    # alone follows the evaluation path: cut at the capture rate, then decimate.
    alone = Shifter(resample_poly(ext[MARGIN:-MARGIN], 1, 4))(EDGE_DELAY)
    m = MARGIN // 4
    with_context = Shifter(resample_poly(ext, 1, 4))(EDGE_DELAY)[m:-m]
    return ev.esr(alone, with_context)


def readings(runs: dict, p_readings: dict) -> dict:
    """Paired differences arm - C per seed on the aligned scores; seeds whose run is on
    a grid edge are dropped, and a reading with a dropped seed gets no label. Against P:
    same label and acceptability -> robust to the alignment; otherwise divergence, and a
    divergence on acceptability goes to the user (predictions.md)."""
    out = {}
    for arm in ("L10", "L20"):
        row = {}
        for key in ("s_pairs", "s_test8", "take16"):
            d_i = {}
            for seed in tl.SEEDS:
                a, c = (runs[tl.ARMS[k].format(seed)] for k in (arm, "C"))
                if not (a["grid_edge"] or c["grid_edge"]):
                    d_i[seed] = a[key] - c[key]
            d = statistics.mean(d_i.values()) if d_i else None
            row[key] = {"d_i": d_i, "d": d}
            if key == "s_pairs" and len(d_i) == len(tl.SEEDS):
                p = p_readings[f"{arm}_last"]["s_pairs"]
                verdict = tl.verdict(list(d_i.values()))
                accept = tl.acceptable(list(d_i.values()))
                same = (verdict, accept) == (p["verdict"], p["acceptable"])
                row[key] |= {
                    "verdict": verdict,
                    "acceptable": accept,
                    "p_verdict": p["verdict"],
                    "p_acceptable": p["acceptable"],
                    "against_p": "robuste au calage" if same else "divergence",
                    "acceptability_to_user": accept != p["acceptable"],
                }
        out[arm] = row
    return out


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    p_result = json.loads((HERE / "eval_train_length.json").read_text())
    takes = [t for t in ev.takes() if not t["low_signal"]]
    data = {t["order"]: ev.audio(t) for t in takes}
    validation = {
        t["order"]: tuple(
            resample_poly(s, 1, 4) for s in vs100r.audio(t, name="di_validation")
        )
        for t in takes
    }
    result = {
        "rule": "predictions.md (6f62341) : règle écrite avant P, exécutée après P",
        "takes": [t["order"] for t in takes],
        "edge_effect": {"take": takes[0]["order"], "energy": edge_effect(takes[0])},
        "runs": {},
    }
    print(f"edge effect {result['edge_effect']['energy']:.3e}", flush=True)
    for arm, pattern in tl.ARMS.items():
        for seed in tl.SEEDS:
            run_id = pattern.format(seed)
            model = tl.load(run_id, "last").to(DEVICE)
            tau, curve = common_delay(model, takes, validation)
            entry = {"arm": arm, "seed": seed, "tau": tau, "median_esr_curve": curve}
            entry["grid_edge"] = tau in (DELAYS[0], DELAYS[-1])
            entry |= score(model, takes, data, tau)
            result["runs"][run_id] = entry
            print(f"{run_id}: tau {tau:+.2f} s {entry['s_pairs']:.3f}", flush=True)
    smallest = min(
        math.exp(v)
        for run in result["runs"].values()
        if not run["grid_edge"]
        for v in run["ln_esr_by_take"].values()
    )
    result["edge_effect"] |= {
        "bound": 0.01 * smallest,
        "reliable": result["edge_effect"]["energy"] < 0.01 * smallest,
    }
    if result["edge_effect"]["reliable"]:
        result["readings"] = readings(result["runs"], p_result["readings"])
        print(json.dumps(result["readings"], indent=1, ensure_ascii=False))
    else:
        print("edge effect above the bound: co-reading not reliable, not read")
    (HERE / "eval_train_length_aligned.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
