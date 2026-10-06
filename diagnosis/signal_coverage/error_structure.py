#!/usr/bin/env python3
"""Where the VS100R models' error lies: rare input levels (O1), after a loud event (O2),
where the seeds disagree (O3). Predictions in NOTE.md, written first. Read-only, CPU.

Models A48 and C, section di_test of the 35 non-low-signal takes at 48 kHz (scored
region of product_vs100r_eval.py), 10 ms frames. Writes error_structure.json.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import product_vs100r_eval as ev  # noqa: E402

FRAME = 480  # 10 ms at 48 kHz
ARMS = {
    "A48": ["vs100r_gc_film_joint_s925_r2", "vs100r48_film_s43", "vs100r48_film_s44"],
    "C": ["vs100r48fast_film_s42", "vs100r48fast_film_s43", "vs100r48fast_film_s44"],
}
EDGES = np.arange(-60, -5, 5)  # level classes [-60,-55) ... [-15,-10)
TRAINVAL = ROOT / "datasets/raw/internal/valvestate_vs100r/explore_48k/dry/trainval"
SINCE_MS = ((0, 50), (50, 200), (200, 1000), (1000, np.inf))


def frame_energy(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    return (x[: n * FRAME].reshape(n, FRAME) ** 2).sum(1)


def level_db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(frame_energy(x) / FRAME + 1e-20)


def training_density() -> np.ndarray:
    levels = np.concatenate(
        [
            level_db(sf.read(p, dtype="float64")[0])
            for p in sorted(TRAINVAL.glob("*.wav"))
        ]
    )
    counts = np.histogram(levels, bins=EDGES)[0]
    return counts / counts.sum()


def since_last_loud(levels: np.ndarray, loud: float) -> np.ndarray:
    """Milliseconds since the last frame at or above `loud` (inf before any)."""
    out = np.full(len(levels), np.inf)
    last = None
    for i, value in enumerate(levels):
        if value >= loud:
            last = i
        if last is not None:
            out[i] = (i - last) * FRAME / 48.0
    return out


def esr(err: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> float:
    return float(err[mask].sum() / tgt[mask].sum()) if mask.any() else float("nan")


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    density = training_density()
    takes = [t for t in ev.takes() if not t["low_signal"]]
    frames: dict[str, dict] = {}
    o3: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for arm, runs in ARMS.items():
        models = [ev.load(r) for r in runs]
        for take in takes:
            loop, fx = ev.audio(take)
            x = resample_poly(loop, 1, 4)
            target = ev.scored(resample_poly(fx, 1, 4))
            controls = torch.tensor([v / 10 for v in take["gain_contour"]])
            outs = [ev.scored(ev.run(m, x, controls)) for m in models]
            lv = level_db(ev.scored(x))
            tgt = frame_energy(target)
            for run, y in zip(runs, outs, strict=True):
                f = frames.setdefault(
                    run, {"level": [], "tgt": [], "err": [], "since": [], "mid": []}
                )
                f["level"].append(lv)
                f["tgt"].append(tgt)
                f["err"].append(frame_energy(y - target))
                f["since"].append(since_last_loud(lv, np.percentile(lv, 90)))
                lo, hi = np.percentile(lv, [30, 70])
                f["mid"].append((lv >= lo) & (lv <= hi))
            mean = np.mean(outs, axis=0)
            disagree = np.mean([frame_energy(y - mean) for y in outs], axis=0) / tgt
            error = np.mean([frame_energy(y - target) for y in outs], axis=0) / tgt
            keep = 10 * np.log10(tgt / FRAME + 1e-20) > -50
            if keep.sum() > 50:
                o3[arm].append(float(spearmanr(disagree[keep], error[keep])[0]))
            print(arm, take["order"], flush=True)
    result = {"density": density.tolist(), "edges": EDGES.tolist(), "runs": {}}
    for run, f in frames.items():
        lv, tgt, err, since, mid = (
            np.concatenate(f[k]) for k in ("level", "tgt", "err", "since", "mid")
        )
        cls = np.digitize(lv, EDGES) - 1
        by_class = [
            (k, esr(err, tgt, cls == k), int((cls == k).sum()))
            for k in range(len(EDGES) - 1)
        ]
        usable = [(k, e) for k, e, n in by_class if n >= 200 and density[k] > 0]
        rho = spearmanr(
            [np.log(density[k]) for k, _ in usable], [np.log(e) for _, e in usable]
        )[0]
        by_since = {
            f"{a}-{b}": esr(err, tgt, mid & (since >= a) & (since < b))
            for a, b in SINCE_MS
        }
        result["runs"][run] = {
            "esr_by_level_class": by_class,
            "o1_spearman": float(rho),
            "o2_esr_by_since_ms": by_since,
            "o2_ratio": by_since["50-200"] / by_since["1000-inf"],
        }
    rho1 = statistics.median(r["o1_spearman"] for r in result["runs"].values())
    ratio2 = statistics.median(r["o2_ratio"] for r in result["runs"].values())
    rho3 = {arm: statistics.median(v) for arm, v in o3.items()}
    result["readings"] = {
        "O1_median_spearman": rho1,
        "O1": "soutenue"
        if rho1 <= -0.5
        else "contraire"
        if abs(rho1) < 0.3
        else "intermédiaire",
        "O2_median_ratio": ratio2,
        "O2": "soutenue"
        if ratio2 >= 1.5
        else "contraire"
        if 0.8 <= ratio2 <= 1.25
        else "intermédiaire",
        "O3_median_spearman": rho3,
        "O3": "soutenue"
        if min(rho3.values()) >= 0.5
        else "contraire"
        if max(rho3.values()) < 0.2
        else "intermédiaire",
    }
    print(json.dumps(result["readings"], indent=1, ensure_ascii=False))
    (HERE / "error_structure.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
