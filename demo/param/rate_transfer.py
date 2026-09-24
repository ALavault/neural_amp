#!/usr/bin/env python3
"""Measurements of demo/param/predictions_rate.md. (Read-only, CPU.)

The parametric SSM-WaveNet-FiLM of the Marshall at other sample rates: dt rescaled
(`delta`), then read as first-order hold (`foh`), against the native 44.1 kHz capture
and, above 48 kHz, against the model at 48 kHz inside the common band. Every pass
receives the knob values of its segment; the mislabelled setting gets treble 0.1.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import product_nablafx_bench as bench
import product_sr_transfer as transfer
from nablafx.core import BlackBoxModel
from nablafx.data.datasets import ParametricPluginDataset

from fssr_nam.models.ssm_wavenet import zoh_to_foh

RUN = "param_ssm_wavenet_marshall"
UP = (96_000, 192_000)
# label_check.md: this ToneTwist setting is treble 1, not 10 (bass, mid, treble, gain).
MISLABELLED_FIX = {"B030_M070_T100_G050": torch.tensor([0.3, 0.7, 0.1, 0.5])}


def load_marshall(run_id: str) -> BlackBoxModel:
    record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
    args = argparse.Namespace(model=record["model"], device="marshall", **record["ssm"])
    processor = bench.build_processor(args)
    path = bench.RUNS_DIR / f"nablafx_{run_id}/checkpoints/last.ckpt"
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.processor."
    processor.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    )
    return BlackBoxModel(processor).eval()


def tensors(
    dataset, keep: list[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    keep = list(range(len(dataset))) if keep is None else keep
    items = [dataset[i] for i in keep]
    x, y, c = (torch.stack(p) for p in zip(*items, strict=True))
    settings = [Path(dataset.samples[i]["target_file"]).parent.name for i in keep]
    c = c.clone()
    for k, setting in enumerate(settings):
        if setting in MISLABELLED_FIX:
            c[k] = MISLABELLED_FIX[setting]
    return x, y, c, settings


@torch.no_grad()
def run(model, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    out = []
    for start in range(0, len(x), 8):
        model.reset_states()
        out.append(model(x[start : start + 8], c[start : start + 8]))
    return torch.cat(out)


def esr_by_setting(out, ref, rate, settings) -> dict:
    o, r = transfer.interior(out, rate), transfer.interior(ref, rate)
    per = (((r - o) ** 2).sum(-1) / ((r**2).sum(-1) + 1e-8)).flatten()
    groups: dict[str, list[float]] = {}
    for value, setting in zip(per.tolist(), settings, strict=True):
        groups.setdefault(setting, []).append(value)
    return {s: statistics.mean(v) for s, v in sorted(groups.items())}


def arms(model, base, x44, y44, c44, x48, c48, settings) -> dict:
    result = {}
    transfer.set_rate(model, 44_100, base)
    result["44k1"] = esr_by_setting(run(model, x44, c44), y44, 44_100, settings)
    transfer.set_rate(model, 48_000, base)
    reference48 = run(model, x48, c48)
    for rate in UP:
        transfer.set_rate(model, rate, base)
        out = transfer.whole(
            run(model, transfer.whole(x44, 44_100, rate), c44), rate, 48_000
        )
        result[f"{rate // 1000}k_band48_vs_48k"] = esr_by_setting(
            out, reference48, 48_000, settings
        )
        print(
            rate,
            statistics.median(result[f"{rate // 1000}k_band48_vs_48k"].values()),
            flush=True,
        )
    transfer.set_rate(model, 48_000, base)
    return result


def main() -> None:
    torch.set_num_threads(16)
    model = load_marshall(RUN)
    data48 = bench.data_module("test", device="marshall")
    data48.setup("test")
    x48, _, c48, settings48 = tensors(data48.test_dataset)
    spec = bench.DEVICES["marshall"]
    native = ParametricPluginDataset(
        root_dir_dry=str(spec["dry"] / "test"),
        root_dir_wet=str(spec["wet"] / "test"),
        params_idxs_to_use=spec["params"],
        sample_length=44_100 * 5,
        sample_rate=44_100,
        preload=True,
    )
    # nablafx counts 48 kHz segments from the 44.1 kHz file length: 17 per setting
    # there, 19 natively. Keep the native segments that have a 48 kHz twin.
    twins = {
        (Path(s["target_file"]).parent.name, s["offset"] // 240_000)
        for s in data48.test_dataset.samples
    }
    keep = [
        i
        for i, s in enumerate(native.samples)
        if (Path(s["target_file"]).parent.name, s["offset"] // 220_500) in twins
    ]
    x44, y44, c44, settings = tensors(native, keep)
    assert settings == settings48, "44.1 and 48 kHz segments must correspond"

    base = {
        id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
        for m in transfer.ssm_modules(model)
    }
    out = {"delta": arms(model, base, x44, y44, c44, x48, c48, settings)}
    transfer.set_rate(model, 48_000, base)
    around = transfer.resample(
        run(model, transfer.resample(x44, 44_100, 48_000), c44), 48_000, 44_100
    )
    out["around"] = {"44k1": esr_by_setting(around, y44, 44_100, settings)}
    zoh_to_foh(model)
    base = {
        id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
        for m in transfer.ssm_modules(model)
    }
    out["foh"] = arms(model, base, x44, y44, c44, x48, c48, settings)

    medians = {
        arm: {k: statistics.median(v.values()) for k, v in results.items()}
        for arm, results in out.items()
    }
    print(json.dumps(medians, indent=1))
    path = Path(__file__).with_name("rate_transfer.json")
    path.write_text(
        json.dumps({"by_setting": out, "medians": medians}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
