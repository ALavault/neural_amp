#!/usr/bin/env python3
"""Run a Big Muff bench checkpoint, trained at 48 kHz, at other sample rates.

Every memory in SSM-WaveNet is a diagonal SSM whose pole is exp(lambda * dt), with
dt = exp(log_dt); everything else acts sample by sample. nablafx's S4-TF-L-16 also keeps
a TFiLM (pooling over 128-sample blocks and an LSTM), whose physical rate follows fs:
rescaling its SSMs only transfers it partially. Under
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
- *_lead: delta with C multiplied by exp(lambda (dt0 - dt')), exploratory; this is not
  a full time advance (see set_rate), so its failure does not test a timing hypothesis.
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

# two_tap_kernel lived here when it measured H6; moved unchanged to the model module.
from fssr_nam.models.ssm_wavenet import two_tap_kernel, zoh_to_foh

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
    model: torch.nn.Module,
    rate: float,
    base: dict,
    lead: bool = False,
    hold: bool = False,
) -> None:
    """Rescale every log_dt from its trained value for a model running at `rate`.

    Both SSMs output C h[n] with h[n] = a h[n-1] + b x[n], read after x[n] enters.
    With lead=True, C is multiplied by exp(lambda * (fs/48000 - 1) dt'), meant to
    shift each mode by 1/48000 - 1/fs seconds. Codex's review (2026-09-23) showed it
    is not a time shift: a true advance also needs the input term
    (exp(lambda eps) - 1) / lambda * x, so this rescales the modal residues (a DC gain
    of exp(lambda eps) on real modes). Kept as the exploratory arm that was
    pre-registered, not as a test of the timing argument.

    With hold=True, each mode keeps the 48 kHz zero-order-hold gate: its input
    coefficient (exp(lambda dt') - 1) / lambda becomes (exp(lambda dt0) - 1) / lambda *
    dt' / dt0, dt0 the trained step (demo/sr_transfer/predictions_hold.md). Causal, and
    the identity at 48 kHz.
    """
    for module in ssm_modules(model):
        log_dt, c = base[id(module)]
        module.log_dt.data = log_dt + math.log(TRAIN_RATE / rate)
        factor = torch.ones(())
        if lead:
            lam = torch.complex(-torch.exp(module.log_A_real), module.A_imag).detach()
            dt = torch.exp(module.log_dt.detach()).unsqueeze(-1)
            factor = torch.exp(lam * dt * (rate / TRAIN_RATE - 1))
        if hold:
            lam = torch.complex(-torch.exp(module.log_A_real), module.A_imag).detach()
            dt0 = torch.exp(log_dt).unsqueeze(-1)
            dt = torch.exp(module.log_dt.detach()).unsqueeze(-1)
            factor = torch.expm1(lam * dt0) * (dt / dt0) / torch.expm1(lam * dt)
        module.C.data = torch.view_as_real(torch.view_as_complex(c) * factor)


def set_two_tap(model: torch.nn.Module, base: dict, on: bool) -> None:
    """Install (or remove) the two-coefficient kernel on every SSM, after set_rate."""
    for module in ssm_modules(model):
        name = "get_kernel" if hasattr(type(module), "get_kernel") else "_kernel"
        if on:
            dt0 = torch.exp(base[id(module)][0])
            setattr(module, name, lambda n, m=module, d=dt0: two_tap_kernel(m, d, n))
        elif name in vars(module):
            delattr(module, name)


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


TRIM_S = 0.1


def whole(segments: torch.Tensor, source: int, target: int) -> torch.Tensor:
    """Resample the concatenated segments as one file, then cut them again."""
    count = len(segments)
    signal = resample(segments.reshape(1, -1), source, target)
    return signal.reshape(count, 1, -1)


def interior(x: torch.Tensor, rate: int) -> torch.Tensor:
    trim = int(rate * TRIM_S)
    return x[..., trim:-trim]


def post_hoc(model, base, x44, y44, out48) -> dict:
    """Arms rescored without resampling edges (Codex's review, point 5) and, above
    48 kHz, inside the 48 kHz model's band (point 6). Post hoc: no prediction."""
    arms = {}
    set_rate(model, DEVICE_RATE, base)
    arms["44k1_delta_interior"] = score(
        interior(run(model, x44), DEVICE_RATE), interior(y44, DEVICE_RATE)
    )
    # out48 ran on the bench's own input, resampled as a whole file.
    around = whole(out48, TRAIN_RATE, DEVICE_RATE)
    arms["44k1_around_wholefile_interior"] = score(
        interior(around, DEVICE_RATE), interior(y44, DEVICE_RATE)
    )
    for rate in (88_200, 96_000, 192_000):
        x = whole(x44, DEVICE_RATE, rate)
        reference = whole(out48, TRAIN_RATE, rate)
        for name, scale in (("naive", TRAIN_RATE), ("delta", rate)):
            set_rate(model, scale, base)
            out = run(model, x)
            key = f"{rate // 1000}k_{name}"
            arms[f"{key}_vs_around_interior"] = score(
                interior(out, rate), interior(reference, rate)
            )
            arms[f"{key}_band48_vs_48k_interior"] = score(
                interior(whole(out, rate, TRAIN_RATE), TRAIN_RATE),
                interior(out48, TRAIN_RATE),
            )
        print(rate, arms[f"{rate // 1000}k_delta_band48_vs_48k_interior"], flush=True)
    return arms


def hold_test(model, base, x44, y44, x48) -> dict:
    """Arms of demo/sr_transfer/predictions_hold.md, scored on segment interiors."""
    arms = {}
    for name, hold in (("delta", False), ("delta_hold", True)):
        set_rate(model, DEVICE_RATE, base, hold=hold)
        arms[f"44k1_{name}_interior"] = score(
            interior(run(model, x44), DEVICE_RATE), interior(y44, DEVICE_RATE)
        )
    for gain, tag in ((1.0, ""), (0.1, "_m20db")):
        set_rate(model, TRAIN_RATE, base)
        reference = run(model, x48 * gain)
        rates = (88_200, 96_000, 192_000) if gain == 1.0 else (96_000, 192_000)
        for rate in rates:
            x = whole(x44 * gain, DEVICE_RATE, rate)
            for name, hold in (("delta", False), ("delta_hold", True)):
                set_rate(model, rate, base, hold=hold)
                out = whole(run(model, x), rate, TRAIN_RATE)
                key = f"{rate // 1000}k_{name}{tag}_band48_vs_48k_interior"
                arms[key] = score(
                    interior(out, TRAIN_RATE), interior(reference, TRAIN_RATE)
                )
                print(key, arms[key], flush=True)
    return arms


def two_tap_test(model, base, x44, y44, x48) -> dict:
    """The two-coefficient arm, scored like hold_test."""
    arms = {}
    set_rate(model, TRAIN_RATE, base)
    set_two_tap(model, base, True)
    arms["48k_two_tap_identity_esr"] = score(
        run(model, x48), run_plain(model, base, x48)
    )
    set_rate(model, DEVICE_RATE, base)
    arms["44k1_delta_2tap_interior"] = score(
        interior(run(model, x44), DEVICE_RATE), interior(y44, DEVICE_RATE)
    )
    set_two_tap(model, base, False)
    set_rate(model, TRAIN_RATE, base)
    reference = run(model, x48)
    for rate in (88_200, 96_000, 192_000):
        x = whole(x44, DEVICE_RATE, rate)
        set_rate(model, rate, base)
        set_two_tap(model, base, True)
        out = whole(run(model, x), rate, TRAIN_RATE)
        set_two_tap(model, base, False)
        key = f"{rate // 1000}k_delta_2tap_band48_vs_48k_interior"
        arms[key] = score(interior(out, TRAIN_RATE), interior(reference, TRAIN_RATE))
        print(key, arms[key], flush=True)
    return arms


def plain_test(model, base, x44, y44, x48) -> dict:
    """dt rescaling alone, scored like two_tap_test: for a first-order-hold model,
    trained as such or read from a ZOH checkpoint by zoh_to_foh (FIX_foh.md)."""
    arms = {}
    set_rate(model, DEVICE_RATE, base)
    arms["44k1_delta_interior"] = score(
        interior(run(model, x44), DEVICE_RATE), interior(y44, DEVICE_RATE)
    )
    set_rate(model, TRAIN_RATE, base)
    reference = run(model, x48)
    for rate in (88_200, 96_000, 192_000):
        set_rate(model, rate, base)
        out = whole(run(model, whole(x44, DEVICE_RATE, rate)), rate, TRAIN_RATE)
        key = f"{rate // 1000}k_delta_band48_vs_48k_interior"
        arms[key] = score(interior(out, TRAIN_RATE), interior(reference, TRAIN_RATE))
        print(key, arms[key], flush=True)
    set_rate(model, TRAIN_RATE, base)
    return arms


def run_plain(model, base, x):
    set_two_tap(model, base, False)
    out = run(model, x)
    set_two_tap(model, base, True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="a demo/nablafx_bench record")
    parser.add_argument("--checkpoint", default="last", choices=["last", "best"])
    parser.add_argument(
        "--gates-only", action="store_true", help="48 kHz gate and pole audit only"
    )
    parser.add_argument(
        "--post-hoc",
        action="store_true",
        help="post-hoc checks after Codex's review, in a separate file: whole-file"
        " resampling, scores on segment interiors, and up-rate outputs compared with"
        " the 48 kHz model inside its band",
    )
    parser.add_argument(
        "--hold-test",
        action="store_true",
        help="arms of demo/sr_transfer/predictions_hold.md, in a separate file",
    )
    parser.add_argument(
        "--two-tap-test",
        action="store_true",
        help="two-coefficient arm of demo/sr_transfer/predictions_hold.md",
    )
    parser.add_argument(
        "--as-foh",
        action="store_true",
        help="read the ZOH checkpoint as a first-order hold (zoh_to_foh), then plain"
        " dt-rescaling test (demo/sr_transfer/FIX_foh.md)",
    )
    parser.add_argument(
        "--plain-test",
        action="store_true",
        help="dt rescaling alone, scored like --two-tap-test (for FOH checkpoints)",
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
    if args.as_foh:
        zoh_to_foh(model)
        # set_rate restores C from base: take it again, after the conversion.
        base = {
            id(m): (m.log_dt.detach().clone(), m.C.detach().clone())
            for m in ssm_modules(model)
        }
        set_rate(model, TRAIN_RATE, base)
        result["run_id"] += "_asfoh"
        result["arms"] = {
            "48k": arms["48k"],
            "48k_asfoh_identity_esr": score(run(model, x48), out48),
            **plain_test(model, base, x44, y44, x48),
        }
        record_result(result, args.no_record)
        return
    if args.plain_test:
        result["run_id"] += "_plain"
        result["arms"] = {"48k": arms["48k"], **plain_test(model, base, x44, y44, x48)}
        record_result(result, args.no_record)
        return
    if args.two_tap_test:
        result["run_id"] += "_twotap"
        result["arms"] = {
            "48k": arms["48k"],
            **two_tap_test(model, base, x44, y44, x48),
        }
        record_result(result, args.no_record)
        return
    if args.hold_test:
        result["run_id"] += "_holdtest"
        result["arms"] = {"48k": arms["48k"], **hold_test(model, base, x44, y44, x48)}
        record_result(result, args.no_record)
        return
    if args.post_hoc:
        result["run_id"] += "_posthoc"
        result["arms"] = {"48k": arms["48k"], **post_hoc(model, base, x44, y44, out48)}
        record_result(result, args.no_record)
        return

    set_rate(model, DEVICE_RATE, base)
    arms["44k1_delta"] = score(run(model, x44), y44)
    set_rate(model, DEVICE_RATE, base, lead=True)
    arms["44k1_delta_lead"] = score(run(model, x44), y44)
    set_rate(model, TRAIN_RATE, base)
    arms["44k1_naive"] = score(run(model, x44), y44)
    # Input resampled segment by segment, so each 5 s span is the same audio.
    set_rate(model, TRAIN_RATE, base)
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

    record_result(result, args.no_record)


def record_result(result: dict, no_record: bool) -> None:
    print(json.dumps(result, indent=2))
    if no_record:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{result['run_id']}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with bench.RUNS_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
