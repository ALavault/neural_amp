#!/usr/bin/env python3
"""Run a Big Muff bench checkpoint, trained at 48 kHz, at other sample rates.

Every memory in SSM-WaveNet, and in nablafx's S4, is a diagonal SSM whose pole is
exp(lambda * dt), with dt = exp(log_dt); everything else acts sample by sample. Under
zero-order hold the input vector follows dt as well, so running the model at fs with
dt * 48000 / fs samples the same continuous-time system at the new rate. This script
adds log(48000 / fs) to every log_dt and compares arms on the ToneTwist Big Muff test
file (60 s, twelve 5 s segments, each starting from a zero state as in the bench):

- 48k: the bench's own test input (44.1 -> 48 kHz by torchaudio), unscaled model. Gate:
  reproduces the ESR the bench recorded.
- at 44.1 kHz, scored against the device's native 44.1 kHz capture:
  naive (dt unchanged), delta (dt rescaled), around (resample to 48 kHz, model,
  resample back: offline and non-causal, an upper bound for real-time resampling);
  up2_delta / up4_delta: resample to 88.2 / 176.4 kHz, delta-rescaled model, back to
  44.1 kHz - oversampling without retraining, still scored against the device.
- *_lead: delta plus the SSM-branch lead restored (see set_rate), exploratory.
- at 88.2, 96 and 192 kHz there is no device reference: naive and delta are scored
  against `around` resampled to that rate, i.e. consistency with the 48 kHz model.

Metrics are nablafx's esr_loss and mrstft_loss, averaged over batches of eight weighted
by batch size, as Lightning's test loop does. MR-STFT FFT sizes are fixed in samples,
so MR-STFT is compared only within one rate. CPU only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import product_nablafx_bench as bench
from nablafx.core import BlackBoxModel
from nablafx.data.datasets import PluginDataset
from nablafx.evaluation.registry import EvaluationRegistry

TRAIN_RATE = 48_000
DEVICE_RATE = 44_100
SEGMENT_S = 5
OUT_DIR = bench.ROOT / "demo/sr_transfer"
ESR = EvaluationRegistry.get_function("esr_loss")
MRSTFT = EvaluationRegistry.get_function("mrstft_loss")


def load_model(run_id: str, checkpoint: str) -> tuple[torch.nn.Module, dict]:
    record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
    ssm = record["ssm"] or {}
    args = argparse.Namespace(model=record["model"], **ssm)
    processor = bench.build_processor(args)
    run_dir = bench.RUNS_DIR / f"nablafx_{run_id}/checkpoints"
    path = run_dir / (
        "last.ckpt" if checkpoint == "last" else record["best_checkpoint"]
    )
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.processor."
    processor.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    )
    model = BlackBoxModel(processor).eval()
    return model, record


def ssm_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [m for m in model.modules() if hasattr(m, "log_dt") and hasattr(m, "C")]


def set_rate(
    model: torch.nn.Module, rate: float, base: dict, lead: bool = False
) -> None:
    """Rescale every log_dt from its trained value for a model running at `rate`.

    Both SSMs output C h[n] with h[n] = a h[n-1] + b x[n]: the state has integrated
    the held input up to t_n + dt, so each SSM branch leads its D feedthrough by one
    sample, 1/fs seconds. Training fixed that lead at 1/48000 s. With lead=True, C is
    multiplied by exp(lambda * (fs/48000 - 1) dt') to restore it; for fs above 48 kHz
    this drops the input the lead would need from the future (exploratory arm).
    """
    for module in ssm_modules(model):
        log_dt, c = base[id(module)]
        module.log_dt.data = log_dt + math.log(TRAIN_RATE / rate)
        factor = torch.ones(())
        if lead:
            lam = torch.complex(-torch.exp(module.log_A_real), module.A_imag).detach()
            dt = torch.exp(module.log_dt.detach()).unsqueeze(-1)
            factor = torch.exp(lam * dt * (rate / TRAIN_RATE - 1))
        module.C.data = torch.view_as_real(torch.view_as_complex(c) * factor)


def pole_audit(model: torch.nn.Module, rate: float) -> dict:
    """Modes whose discrete angle |Im(lambda)| * dt exceeds pi at `rate`."""
    folded = total = 0
    for module in model.modules():
        if hasattr(module, "log_dt") and hasattr(module, "A_imag"):
            dt = torch.exp(module.log_dt.detach()) * TRAIN_RATE / rate
            angle = module.A_imag.detach().abs() * dt.unsqueeze(-1)
            folded += int((angle > math.pi).sum())
            total += angle.numel()
    return {"folded": folded, "total": total}


@torch.no_grad()
def run(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    outputs = []
    for batch in x.split(8):
        model.reset_states()
        outputs.append(model(batch, None))
    return torch.cat(outputs)


@torch.no_grad()
def score(pred: torch.Tensor, target: torch.Tensor) -> dict:
    esr = mrstft = 0.0
    for p, t in zip(pred.split(8), target.split(8), strict=True):
        esr += float(ESR(p, t)) * len(p)
        mrstft += float(MRSTFT(p, t)) * len(p)
    return {"esr": esr / len(pred), "mrstft": mrstft / len(pred)}


def resample(x: torch.Tensor, source: int, target: int) -> torch.Tensor:
    return torchaudio.functional.resample(x, source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="a demo/nablafx_bench record")
    parser.add_argument("--checkpoint", default="last", choices=["last", "best"])
    parser.add_argument(
        "--gates-only", action="store_true", help="48 kHz gate and pole audit only"
    )
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)

    model, record = load_model(args.run_id, args.checkpoint)
    base = {
        id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
        for m in ssm_modules(model)
    }
    # Gate (d): zoh has no learned input vector, so no parameter ends in ".B".
    has_b = any(name.endswith(".B") for name, _ in model.named_parameters())
    result = {
        "run_id": f"sr_transfer_{args.run_id}_{args.checkpoint}",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": bench.git_head(bench.ROOT),
        "source_run": args.run_id,
        "checkpoint": args.checkpoint,
        "model": record["model"],
        "discretization": (record["ssm"] or {}).get("discretization", "nablafx-dssm"),
        "learned_input_vector": has_b,
        "poles": {
            str(rate): pole_audit(model, rate) for rate in (44_100, 48_000, 96_000)
        },
        "arms": {},
    }

    # 48 kHz gate: the bench's own test data, through the same data class.
    bench48 = bench.data_module("test")
    bench48.setup("test")
    x48 = torch.stack([s[0] for s in bench48.test_dataset])
    y48 = torch.stack([s[1] for s in bench48.test_dataset])
    set_rate(model, TRAIN_RATE, base)
    out48 = run(model, x48)
    recorded = record[f"test_{args.checkpoint}"]["metric/test/esr"]
    result["arms"]["48k"] = {**score(out48, y48), "recorded_esr": recorded}
    print(json.dumps(result, indent=2), flush=True)
    if args.gates_only:
        return

    native = PluginDataset(
        root_dir_dry=str(bench.DATA / "DRY/test"),
        root_dir_wet=str(bench.WET / "test" / bench.SETTING),
        sample_length=DEVICE_RATE * SEGMENT_S,
        sample_rate=DEVICE_RATE,
        preload=True,
    )
    x44 = torch.stack([s[0] for s in native])
    y44 = torch.stack([s[1] for s in native])
    assert len(x44) == len(x48)
    arms = result["arms"]

    set_rate(model, DEVICE_RATE, base)
    arms["44k1_delta"] = score(run(model, x44), y44)
    set_rate(model, DEVICE_RATE, base, lead=True)
    arms["44k1_delta_lead"] = score(run(model, x44), y44)
    set_rate(model, TRAIN_RATE, base)
    arms["44k1_naive"] = score(run(model, x44), y44)
    # Input resampled segment by segment, so each 5 s span is the same audio.
    around44 = resample(
        run(model, resample(x44, DEVICE_RATE, TRAIN_RATE)), TRAIN_RATE, DEVICE_RATE
    )
    arms["44k1_around"] = score(around44, y44)
    for factor in (2, 4):
        rate = DEVICE_RATE * factor
        set_rate(model, rate, base)
        out = run(model, resample(x44, DEVICE_RATE, rate))
        arms[f"44k1_up{factor}_delta"] = score(resample(out, rate, DEVICE_RATE), y44)
        set_rate(model, rate, base, lead=True)
        out = run(model, resample(x44, DEVICE_RATE, rate))
        arms[f"44k1_up{factor}_delta_lead"] = score(
            resample(out, rate, DEVICE_RATE), y44
        )

    for rate in (88_200, 96_000, 192_000):
        x = resample(x44, DEVICE_RATE, rate)
        set_rate(model, TRAIN_RATE, base)
        reference = resample(
            run(model, resample(x, rate, TRAIN_RATE)), TRAIN_RATE, rate
        )
        arms[f"{rate // 1000}k_naive_vs_around"] = score(run(model, x), reference)
        set_rate(model, rate, base)
        arms[f"{rate // 1000}k_delta_vs_around"] = score(run(model, x), reference)
        set_rate(model, rate, base, lead=True)
        arms[f"{rate // 1000}k_delta_lead_vs_around"] = score(run(model, x), reference)
        print(rate, arms[f"{rate // 1000}k_delta_vs_around"], flush=True)

    print(json.dumps(result, indent=2))
    if args.no_record:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{result['run_id']}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with bench.RUNS_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
