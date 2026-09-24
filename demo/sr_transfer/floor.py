#!/usr/bin/env python3
"""Measurements of demo/sr_transfer/predictions_floor.md. (Read-only, CPU.)

The FOH reading of a ZOH checkpoint (identical to the two-coefficient correction) at
96 kHz against itself at 48 kHz, inside the 48 kHz band and on segment interiors, as
product_sr_transfer.two_tap_test scores it, under five variants V0-V4.
Usage: floor.py [--seeds 925 736 688]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import product_nablafx_bench as bench
import product_sr_transfer as transfer
from nablafx.data.datasets import PluginDataset

from fssr_nam.models.ssm_wavenet import zoh_to_foh

UP = 96_000
CUTOFF_HZ = 10_000


def hq(x: torch.Tensor, source: int, target: int) -> torch.Tensor:
    """Whole-file resampling with a long Kaiser-windowed sinc."""
    count = len(x)
    y = torchaudio.functional.resample(
        x.reshape(1, -1),
        source,
        target,
        lowpass_filter_width=64,
        rolloff=0.99,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    return y.reshape(count, 1, -1)


def esr(out: torch.Tensor, ref: torch.Tensor) -> float:
    """auraloss's ESR per segment (eps 1e-8), averaged: what transfer.score gives,
    without its MR-STFT, whose float32 window refuses float64 inputs."""
    per = ((ref - out) ** 2).sum(-1) / ((ref**2).sum(-1) + 1e-8)
    return float(per.mean())


def interior_esr(out: torch.Tensor, ref: torch.Tensor) -> float:
    return esr(transfer.interior(out, 48_000), transfer.interior(ref, 48_000))


def spectral(out: torch.Tensor, ref: torch.Tensor) -> dict:
    o, r = transfer.interior(out, 48_000), transfer.interior(ref, 48_000)
    error = torch.fft.rfft(o - r)
    freqs = torch.fft.rfftfreq(o.shape[-1], 1 / 48_000)
    energy = error.abs() ** 2
    low = torch.fft.rfft(o), torch.fft.rfft(r)
    for spectrum in low:
        spectrum[..., freqs > CUTOFF_HZ] = 0
    o_low, r_low = (torch.fft.irfft(s, o.shape[-1]) for s in low)
    return {
        "hf_error_share": float(energy[..., freqs > CUTOFF_HZ].sum() / energy.sum()),
        "esr_lowpass": esr(o_low, r_low),
    }


@torch.no_grad()
def measure(seed: int, x48: torch.Tensor, x44: torch.Tensor) -> dict:
    model, _ = transfer.load_model(f"paired_ssm_wavenet_s{seed}", "last")
    zoh_to_foh(model)
    base = {
        id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
        for m in transfer.ssm_modules(model)
    }

    def pair(net, x44, reference_input, resample):
        transfer.set_rate(net, 48_000, base)
        ref = transfer.run(net, reference_input)
        transfer.set_rate(net, UP, base)
        out = resample(transfer.run(net, resample(x44, 44_100, UP)), UP, 48_000)
        transfer.set_rate(net, 48_000, base)
        return out, ref

    result = {}
    out, ref = pair(model, x44, x48, transfer.whole)
    result["V0_default"] = interior_esr(out, ref)
    out, ref = pair(model, x44, hq(x44, 44_100, 48_000), hq)
    result["V1_hq_resampler"] = interior_esr(out, ref)
    model64 = model.double()
    base.update(
        {id(m): (v[0].double(), v[1].double()) for m, v in _pairs(model64, base)}
    )
    x64 = x44.double()
    out, ref = pair(model64, x64, hq(x64, 44_100, 48_000), hq)
    result["V2_hq_float64"] = interior_esr(out, ref)
    result["V4"] = spectral(out, ref)
    quiet = x64 * 0.1
    out, ref = pair(model64, quiet, hq(quiet, 44_100, 48_000), hq)
    result["V3_hq_float64_m20db"] = interior_esr(out, ref)
    print(seed, json.dumps(result), flush=True)
    return result


def _pairs(model, base):
    return [(m, base[id(m)]) for m in transfer.ssm_modules(model)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[925])
    args = parser.parse_args()
    torch.set_num_threads(16)
    bench48 = bench.data_module("test")
    bench48.setup("test")
    x48 = torch.stack([s[0] for s in bench48.test_dataset])
    native = PluginDataset(
        root_dir_dry=str(bench.DATA / "DRY/test"),
        root_dir_wet=str(bench.WET / "test" / bench.SETTING),
        sample_length=44_100 * 5,
        sample_rate=44_100,
        preload=True,
    )
    x44 = torch.stack([s[0] for s in native])
    out = {seed: measure(seed, x48, x44) for seed in args.seeds}
    path = Path(__file__).with_name("floor_" + "_".join(map(str, args.seeds)) + ".json")
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"written {path}")


if __name__ == "__main__":
    main()
