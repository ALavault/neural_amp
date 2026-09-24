#!/usr/bin/env python3
"""Measurements of demo/param/label_check.md. (Read-only, CPU.)

1. Is the ToneTwist target of a setting the same recording as the original speaker
   output of the setting it should match? Offset by FFT cross-correlation, then ESR of
   the ToneTwist target against the least-squares-scaled original over the overlap.
2. Which treble value does the trained parametric SSM-WaveNet recognise in the suspect
   recording? Test ESR with the ToneTwist value (1.0) and the original one (0.1). The
   same for S4-param, added after the SSM result as a descriptive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
from nablafx.core import BlackBoxModel  # noqa: E402
from nablafx.evaluation.registry import EvaluationRegistry  # noqa: E402

ORIGINAL = ROOT / "datasets/raw/external/marshall_jvm410h/extracted/final/test"
TONETWIST = bench.MARSHALL / "SpeakerOut/test"
PAIRS = {  # ToneTwist setting -> original setting
    "B030_M070_T100_G050": "B3_M7_T1_G5",  # suspect
    "B010_M030_T070_G050": "B1_M3_T7_G5",  # control, labels agree
}
TREBLE = 2  # control index: bass, mid, treble, gain


def same_recording(tonetwist: str, original: str) -> dict:
    t, rate_t = sf.read(TONETWIST / tonetwist / f"{tonetwist}.test.target.wav")
    o, rate_o = sf.read(ORIGINAL / original / f"{original}-speakerout.wav")
    assert rate_t == rate_o
    t, o = torch.tensor(t), torch.tensor(o)
    size = 1 << (len(t) + len(o)).bit_length()
    corr = torch.fft.irfft(
        torch.fft.rfft(t, size) * torch.fft.rfft(o, size).conj(), size
    )
    lag = int(corr.argmax())
    lag = lag if lag < size // 2 else lag - size  # t[n] ~ o[n - lag]
    start, stop = max(0, lag), min(len(t), len(o) + lag)
    tt, oo = t[start:stop], o[start - lag : stop - lag]
    gain = float((tt * oo).sum() / (oo * oo).sum())
    return {
        "lag_samples": lag,
        "overlap_s": (stop - start) / rate_t,
        "gain": gain,
        "correlation": float((tt * oo).sum() / (tt.norm() * oo.norm())),
        "esr": float(((tt - gain * oo) ** 2).sum() / (tt**2).sum()),
    }


@torch.no_grad()
def treble_test(run_id: str, setting: str, values: tuple[float, ...]) -> dict:
    record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
    args = argparse.Namespace(
        model=record["model"], device="marshall", **(record["ssm"] or {})
    )
    processor = bench.build_processor(args)
    run_dir = bench.RUNS_DIR / f"nablafx_{run_id}/checkpoints"
    state = torch.load(run_dir / "last.ckpt", map_location="cpu", weights_only=False)
    prefix = "model.processor."
    processor.load_state_dict(
        {
            k[len(prefix) :]: v
            for k, v in state["state_dict"].items()
            if k.startswith(prefix)
        }
    )
    model = BlackBoxModel(processor).eval()
    data = bench.data_module("test", device="marshall")
    data.setup("test")
    dataset = data.test_dataset
    members = [
        i
        for i in range(len(dataset))
        if Path(dataset.samples[i]["target_file"]).parent.name == setting
    ]
    esr = EvaluationRegistry.get_function("esr_loss")
    results = {}
    for value in values:
        total = 0.0
        for start in range(0, len(members), 8):
            items = [dataset[i] for i in members[start : start + 8]]
            x, y, controls = (torch.stack(parts) for parts in zip(*items, strict=True))
            controls = controls.clone()
            controls[:, TREBLE] = value
            model.reset_states()
            total += float(esr(model(x, controls), y)) * len(items)
        results[str(value)] = total / len(members)
        print(f"treble {value}: ESR {results[str(value)]:.4f}", flush=True)
    return {"segments": len(members), "esr_by_treble": results}


def main() -> None:
    torch.set_num_threads(8)
    out = {"same_recording": {}}
    for tonetwist, original in PAIRS.items():
        out["same_recording"][tonetwist] = {"original": original} | same_recording(
            tonetwist, original
        )
        print(tonetwist, original, out["same_recording"][tonetwist], flush=True)
    # SSM first (the measurement of label_check.md); S4-param added afterwards, as a
    # descriptive, to recompute both models' Marshall summaries with the right label.
    out["treble_test"] = {
        run: treble_test(run, "B030_M070_T100_G050", (1.0, 0.1))
        for run in ("param_ssm_wavenet_marshall", "param_s4_marshall")
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"written {path}")


if __name__ == "__main__":
    main()
