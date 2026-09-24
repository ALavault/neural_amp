#!/usr/bin/env python3
"""Level sweep of the -37 dB floor (predictions_floor.md, complementary measurement).
V2 settings (high-quality resampling, float64, FOH reading, 96 kHz against 48 kHz inside
the 48 kHz band) at 0, -20, -40 and -60 dB input, three seeds. (Read-only, CPU.)"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
import product_nablafx_bench as bench
import product_sr_transfer as transfer
from floor import UP, hq, interior_esr
from nablafx.data.datasets import PluginDataset

from fssr_nam.models.ssm_wavenet import zoh_to_foh

LEVELS_DB = (0, -20, -40, -60)


@torch.no_grad()
def sweep(seed: int, x44: torch.Tensor) -> dict:
    model, _ = transfer.load_model(f"paired_ssm_wavenet_s{seed}", "last")
    zoh_to_foh(model)
    model = model.double()
    base = {
        id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
        for m in transfer.ssm_modules(model)
    }
    out = {}
    for db in LEVELS_DB:
        x = x44.double() * 10 ** (db / 20)
        transfer.set_rate(model, 48_000, base)
        ref = transfer.run(model, hq(x, 44_100, 48_000))
        transfer.set_rate(model, UP, base)
        up = hq(transfer.run(model, hq(x, 44_100, UP)), UP, 48_000)
        out[str(db)] = interior_esr(up, ref)
        print(seed, db, out[str(db)], flush=True)
    return out


def main() -> None:
    torch.set_num_threads(16)
    native = PluginDataset(
        root_dir_dry=str(bench.DATA / "DRY/test"),
        root_dir_wet=str(bench.WET / "test" / bench.SETTING),
        sample_length=44_100 * 5,
        sample_rate=44_100,
        preload=True,
    )
    x44 = torch.stack([s[0] for s in native])
    out = {seed: sweep(seed, x44) for seed in (925, 736, 688)}
    Path(__file__).with_name("floor_levels.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
