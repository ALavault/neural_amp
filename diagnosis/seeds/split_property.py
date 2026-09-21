#!/usr/bin/env python3
"""Properties of the validation split, seed by seed. (Read-only, CPU, no GPU.)

Pre-registered in diagnosis/seeds/pilot_G_split_property.md. The properties of a split
depend on the seed alone, not on the run, so they are measured and committed before the
test ESR of seeds 47 to 49 exists: the predictor is frozen before the outcome for three
seeds of eight, and the commit order attests it.

Each split is reproduced by repeating the pilot's construction order - seed_everything,
then the processor, then the data module - and verified per seed, where a finished run
allows it, by recomputing that run's validation loss on the reproduced split and
comparing it with the logged value. A wrong split would differ by about 30 %, the
standard deviation of the segment effect being 1.0 in log.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnosis/butterfly"))

import lightning as pl  # noqa: E402
import product_nablafx_bench as bench  # noqa: E402
import torch  # noqa: E402
from mode_connectivity import processor, weights  # noqa: E402
from nablafx.evaluation.flexible_loss import FlexibleLoss  # noqa: E402

SEEDS = range(42, 50)
SAMPLE_RATE = 48_000


def features(segments: np.ndarray) -> dict[str, float]:
    """Aggregates of a set of target segments, shape (n, samples)."""
    rms = np.sqrt((segments**2).mean(1))
    peak = np.abs(segments).max(1)
    spectrum = np.abs(np.fft.rfft(segments, axis=1)) ** 2
    freqs = np.fft.rfftfreq(segments.shape[1], 1 / SAMPLE_RATE)
    centroid = (freqs[None, :] * spectrum).sum(1) / spectrum.sum(1)
    frame = int(0.020 * SAMPLE_RATE)
    usable = segments[:, : segments.shape[1] // frame * frame]
    envelope = np.sqrt((usable.reshape(len(segments), -1, frame) ** 2).mean(2))
    quiet = (envelope < (peak[:, None] / 100)).mean(1)
    return {
        "rms_min": float(rms.min()),
        "rms_mean": float(rms.mean()),
        "energy": float((segments**2).sum()),
        "crest_mean": float((peak / rms).mean()),
        "centroid_mean": float(centroid.mean()),
        "quiet_share_mean": float(quiet.mean()),
    }


def split(
    seed: int,
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    pl.seed_everything(seed, workers=True)
    bench.build_processor(
        SimpleNamespace(
            model="ssm-wavenet",
            num_blocks=8,
            channels=16,
            state_dim=4,
            output_act="tanh",
            discretization="zoh",
        )
    )
    data = bench.data_module("trainval")
    data.setup("fit")
    indices = list(getattr(data.val_dataset, "indices", []))
    val_x = torch.stack([a for a, _ in data.val_dataset])
    val_y = torch.stack([b for _, b in data.val_dataset])
    train_y = torch.stack([b for _, b in data.train_dataset])
    return indices, val_x, val_y, train_y


def logged(run: str) -> float | None:
    path = bench.RUNS_DIR / f"nablafx_{run}/logs/metrics.csv"
    if not path.exists():
        return None
    with path.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("loss/val/tot")]
    return float(rows[-1]["loss/val/tot"]) if rows else None


def main() -> None:
    torch.set_num_threads(4)
    loss = FlexibleLoss(
        losses=[
            {"name": "l1_loss", "weight": 1.0, "alias": "l1"},
            {"name": "mrstft_loss", "weight": 0.1, "alias": "mrstft"},
        ]
    )
    model = processor()
    out = {}
    for seed in SEEDS:
        indices, val_x, val_y, train_y = split(seed)
        row = {"val_indices": indices}
        row.update({f"val_{k}": v for k, v in features(val_y.numpy()[:, 0]).items()})
        row["train_rms_min"] = features(train_y.numpy()[:, 0])["rms_min"]
        # Control: recompute a finished run's validation loss on the reproduced split.
        run = f"pilotC_decide_seed{seed}"
        reference = logged(run)
        if reference is not None:
            model.load_state_dict(weights(run))
            model.eval()
            with torch.no_grad():
                model.reset_states()
                total = float(loss(model(val_x), val_y)[-1])
            row["logged_val"] = reference
            row["recomputed_val"] = total
            row["relative_gap"] = abs(total - reference) / reference
        out[str(seed)] = row
        controle = (
            f"  controle {row['recomputed_val']:.5f} contre {row['logged_val']:.5f}"
            f" ({row['relative_gap']:.1%})"
            if reference is not None
            else "  (pas encore de run pour cette graine)"
        )
        print(
            f"graine {seed} : rms_min {row['val_rms_min']:.5f}"
            f"  rms_moyen {row['val_rms_mean']:.5f}"
            f"  centroide {row['val_centroid_mean']:.0f} Hz" + controle
        )

    (Path(__file__).parent / "split_property.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
