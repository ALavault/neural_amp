#!/usr/bin/env python3
"""Do the models imitate the noise floor in quiet passages? (Read-only, CPU, no GPU.)

Pre-registered in diagnosis/noise/predictions_quiet.md, written before this script ran.
Quiet and loud passages are chosen on the INPUT (the DI), never on the target, so the
selection does not pick favourable realisations of the target noise. In each, the auto
and cross spectra of model output m and target y give:

  beta = sum|S_my| / sum S_mm   1 for a pure regressor whatever the target noise,
                                below 1 for an output too loud or incoherent;
  L    = sum S_mm / sum S_yy    output level against target level;
  C    = sum(|S_my|^2 / S_mm) / sum S_yy
                                share of target power linearly explained by the output.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnosis/butterfly"))

import product_nablafx_bench as bench  # noqa: E402
from mode_connectivity import processor, weights  # noqa: E402

RATE = 48_000
FRAME = int(0.020 * RATE)
SKIP = RATE // 2  # state warm-up after each reset
GUARD = 10  # quiet frames (200 ms) that must precede a quiet frame
WIN, HOP = 2048, 1024
BAND = (100.0, 10_000.0)
OCTAVES = (125, 250, 500, 1_000, 2_000, 4_000, 8_000)
RUNS = [f"pilotC_{arm}_seed{s}" for arm in ("decide", "fixe") for s in range(42, 58)]


def test_set() -> tuple[np.ndarray, np.ndarray]:
    data = bench.data_module("test")
    data.setup("test")
    items = [data.test_dataset[i] for i in range(len(data.test_dataset))]
    x = torch.stack([a for a, _ in items])
    y = torch.stack([b for _, b in items])
    return x.numpy()[:, 0], y.numpy()[:, 0]


def frame_masks(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Quiet (with 200 ms of quiet before) and loud 20 ms frames of the DI."""
    kept = x[:, SKIP:]
    count = kept.shape[1] // FRAME
    power = (kept[:, : count * FRAME].reshape(len(x), count, FRAME) ** 2).mean(-1)
    floor = float(np.percentile(power, 1))
    quiet = power < 10 * floor
    run = np.zeros(quiet.shape, dtype=int)
    for k in range(count):
        run[:, k] = np.where(quiet[:, k], (run[:, k - 1] if k else 0) + 1, 0)
    return floor, run > GUARD, power >= np.median(power)


def windows(mask: np.ndarray, length: int) -> list[tuple[int, int]]:
    """(segment, start) of every window whose frames all satisfy the mask."""
    out = []
    for seg in range(mask.shape[0]):
        for start in range(0, length - WIN + 1, HOP):
            first, last = start // FRAME, (start + WIN - 1) // FRAME
            if last < mask.shape[1] and mask[seg, first : last + 1].all():
                out.append((seg, SKIP + start))
    return out


def spectra(m: np.ndarray, y: np.ndarray, where: list[tuple[int, int]]):
    taper = np.hanning(WIN)
    ms = np.stack([m[s, t : t + WIN] for s, t in where]) * taper
    ys = np.stack([y[s, t : t + WIN] for s, t in where]) * taper
    fm, fy = np.fft.rfft(ms, axis=1), np.fft.rfft(ys, axis=1)
    return (
        (np.abs(fm) ** 2).mean(0),
        (np.abs(fy) ** 2).mean(0),
        (np.conj(fm) * fy).mean(0),
    )


def stats(smm, syy, smy, freqs, low, high) -> dict[str, float]:
    band = (freqs >= low) & (freqs < high)
    smm, syy, smy = smm[band], syy[band], np.abs(smy[band])
    return {
        "beta": float(smy.sum() / smm.sum()),
        "L": float(smm.sum() / syy.sum()),
        "C": float((smy**2 / smm).sum() / syy.sum()),
    }


def outcome(beta_quiet: float, beta_loud: float) -> str:
    if beta_quiet >= 0.8:
        return "refutee : regression"
    if beta_quiet <= 0.5:
        return (
            "soutenue : exces propre aux passages calmes"
            if beta_loud >= 0.8
            else ("exces general")
        )
    return "indecidable"


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(8)
    x, y = test_set()
    floor, quiet, loud = frame_masks(x)
    length = x.shape[1] - SKIP
    where = {"quiet": windows(quiet, length), "loud": windows(loud, length)}
    freqs = np.fft.rfftfreq(WIN, 1 / RATE)
    print(
        f"{len(x)} segments ; plancher DI {10 * np.log10(floor):.1f} dBFS ; fenetres"
        f" calmes {len(where['quiet'])}, fortes {len(where['loud'])}"
    )
    model = processor()
    rows = {}
    for run in RUNS:
        model.load_state_dict(weights(run))
        model.eval()
        outs = []
        for start in range(0, len(x), 8):
            model.reset_states()
            batch = torch.from_numpy(x[start : start + 8, None])
            outs.append(model(batch)[:, 0].numpy())
        m = np.concatenate(outs)
        esr = float(
            np.mean(((y - m)[:, SKIP:] ** 2).sum(1) / (y[:, SKIP:] ** 2).sum(1))
        )
        row = {"esr_after_skip": esr}
        for kind, spots in where.items():
            smm, syy, smy = spectra(m, y, spots)
            row[kind] = stats(smm, syy, smy, freqs, *BAND)
            row[kind]["octaves"] = {
                str(c): stats(smm, syy, smy, freqs, c / 2**0.5, c * 2**0.5)
                for c in OCTAVES
            }
            row[f"{kind}_target_dbfs"] = float(
                10
                * np.log10(np.mean([(y[s, t : t + WIN] ** 2).mean() for s, t in spots]))
            )
        row["outcome"] = outcome(row["quiet"]["beta"], row["loud"]["beta"])
        rows[run] = row
        print(
            f"{run:22s} ESR {esr:.4f}  beta calme {row['quiet']['beta']:.2f}"
            f" (L {row['quiet']['L']:.2f}, C {row['quiet']['C']:.2f})"
            f"  beta fort {row['loud']['beta']:.2f}  -> {row['outcome']}",
            flush=True,
        )

    median = {
        kind: {
            k: statistics.median(r[kind][k] for r in rows.values())
            for k in ("beta", "L", "C")
        }
        for kind in where
    }
    verdict = outcome(median["quiet"]["beta"], median["loud"]["beta"])
    counts: dict[str, int] = {}
    for r in rows.values():
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    print(f"\nmedianes : {json.dumps(median)}")
    print(f"classes par modele : {counts}")
    print(f"verdict (medianes) : {verdict}")
    (Path(__file__).parent / "quiet_frames.json").write_text(
        json.dumps(
            {
                "di_floor_dbfs": 10 * np.log10(floor),
                "windows": {k: len(v) for k, v in where.items()},
                "rows": rows,
                "median": median,
                "counts": counts,
                "verdict": verdict,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
