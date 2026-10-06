#!/usr/bin/env python3
"""Known-answer bench (predictions.md): does SSM-WaveNet keep a 2nd-harmonic floor on
a symmetric target? Product lane.

Targets: KIPR measurements of an emulated DSP amp engine (datasets/raw/external/kipr/,
not committed), left output channel. `instrument` checks our harmonic measurement on
the targets against KIPR's metrics.json; `train <config> <seed>` trains one model on the
four train_noise segments and writes its record; `measure` reads every record and the
readings fixed in predictions.md. Each run writes demo/runs/kipr_<run>/model.pt and
record.json (a line in demo/RUNS.jsonl too); measure writes known_answer.json here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402

from fssr_nam.models.ssm_wavenet import DiagonalSSMLayer  # noqa: E402

DATA = ROOT / "datasets/raw/external/kipr/extracted/measurements"
RATE = 44_100
# Target advance per round: 198 - 16 in the first (the models did not learn the 16
# samples left); 196 in r2: the four first-round runs that followed the target
# (correlation 0.86 to 0.90) were 14 samples early at every level, so 182 + 14, which
# keeps the target causal (KIPR's correlation peak, 198, includes filter delay).
ADVANCES = {"": 198 - 16, "r2": 196, "odd": 196, "grey": 196}
TAG = ""
CONFIGS = {"sym": "v1_lead_amp", "asym": "v2_lead_bias_16383"}
LEVELS = (-24, -12, -6)
STEPS, HALVINGS = 3000, (1800, 2400)
MODEL = {
    "model": "ssm-wavenet",
    "device": "kipr",
    "num_blocks": 8,
    "channels": 16,
    "state_dim": 4,
    "output_act": "tanh",
    "discretization": "zoh",
    "cond": "film",
    "cond_head": "joint",
    "cond_hidden": (32,),
    "head_act": "leaky",
}


# Model per round: the r2 SSM-WaveNet, its odd gate (demo/odd_gate/predictions.md)
# or the grey box (demo/greybox/predictions.md).
MODEL_TAGS = {
    "odd": {"act_type": "odd", "head_act": "tanh"},
    "grey": {"model": "greybox"},
}


def model_args() -> dict:
    return {**MODEL, **MODEL_TAGS.get(TAG, {})}


# Rounds whose measured output takes the polarity of its correlation with the target
# on the train_noise segments (the trainer has no polarity guard and the MR-STFT is
# sign-blind; two r2 runs came out inverted). Harmonic ratios do not depend on it.
POLARITY_NORMALIZED = {"odd", "grey"}


def run_name(name: str, seed: int) -> str:
    return f"kipr_{name}_s{seed}" + (f"_{TAG}" if TAG else "")


def sequences(config: str) -> tuple[np.ndarray, np.ndarray]:
    """Full input and left output, the output advanced by this round's advance."""
    advance = ADVANCES[TAG]
    x = np.fromfile(DATA / "input.f32", dtype="<f4").astype(np.float64)
    y = np.fromfile(DATA / f"{config}.f32", dtype="<f4").reshape(-1, 2)[:, 0]
    y = np.concatenate([y[advance:], np.zeros(advance)]).astype(np.float64)
    return x, y


def segments() -> dict[str, tuple[int, int, int]]:
    """name -> (context start, segment start, segment end); the context is the
    silence just before the segment."""
    items = json.loads((DATA / "manifest.json").read_text())["segments"]
    out = {}
    for previous, item in itertools.pairwise(items):
        if item["kind"] != "silence":
            start = item["start"]
            out[item["name"]] = (previous["start"], start, start + item["length"])
    return out


def harmonics(signal: np.ndarray, start: int, f0: int, count: int = 5) -> list[float]:
    """Amplitudes of H1..Hcount over the steady 0.5 s from 0.1 s after start: an
    integer number of periods for 100 Hz to 3 kHz, 4-term Blackman-Harris window."""
    n = RATE // 2
    x = signal[start + RATE // 10 : start + RATE // 10 + n]
    i = np.arange(n)
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    w = (
        a[0]
        - a[1] * np.cos(2 * np.pi * i / n)
        + a[2] * np.cos(4 * np.pi * i / n)
        - a[3] * np.cos(6 * np.pi * i / n)
    )
    spectrum = np.abs(np.fft.rfft(x * w))
    return [float(spectrum[k * f0 * n // RATE]) for k in range(1, count + 1)]


def ratios(signal: np.ndarray, seg: dict, f0: int, level: int) -> dict:
    h = harmonics(signal, seg[f"tone_{f0}_{level}"][1], f0)
    return {
        "h2_db": 20 * math.log10(h[1] / h[0]),
        "h3_db": 20 * math.log10(h[2] / h[0]),
    }


def instrument() -> float:
    seg = segments()
    metrics = {
        m["configuration"]["id"]: m
        for m in json.loads((DATA / "metrics.json").read_text())
    }
    worst = 0.0
    for config in CONFIGS.values():
        _, y = sequences(config)
        for level in LEVELS:
            ours = ratios(y, seg, 1000, level)["h2_db"]
            tone = next(
                t
                for t in metrics[config]["tones"]
                if t["frequency"] == 1000 and t["input_peak_dbfs"] == level
            )
            theirs = 20 * math.log10(
                tone["harmonic_peak"][0] / tone["fundamental_peak"]
            )
            worst = max(worst, abs(ours - theirs))
            print(f"{config} {level}: H2/H1 ours {ours:.1f}, KIPR {theirs:.1f} dB")
    print(f"instrument {'OK' if worst <= 1.0 else 'FAILED'}: max gap {worst:.2f} dB")
    return worst


def processor_and_loss():
    DiagonalSSMLayer.fft_size = "fast"
    processor = bench.build_processor(argparse.Namespace(**model_args()))
    loss = bench.FlexibleLoss(
        losses=[
            {"name": "l1_loss", "weight": 0.5, "alias": "l1"},
            {
                "name": "mrstft_loss",
                "weight": 0.5,
                "alias": "mrstft",
                **bench.at_rate(processor, RATE),
            },
        ]
    )
    return processor, loss


def batch(config: str, prefix: str, device: str) -> tuple[torch.Tensor, ...]:
    """The four <prefix>_noise segments with their preceding silence; all have the
    same lengths, so they stack."""
    x, y = sequences(config)
    seg = segments()
    spans = [seg[f"{prefix}_noise_{level}"] for level in (-36, -24, -12, -6)]
    xs = torch.tensor(np.stack([x[a:c] for a, _, c in spans]), dtype=torch.float32)
    ys = torch.tensor(np.stack([y[a:c] for a, _, c in spans]), dtype=torch.float32)
    context = spans[0][1] - spans[0][0]
    return xs[:, None].to(device), ys[:, None].to(device), context


def train(name: str, seed: int, steps: int, smoke: bool = False) -> None:
    config = CONFIGS[name]
    run_id = run_name(name, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.nn.functional.pad = bench._reflect_pad_by_slicing
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    processor, loss = processor_and_loss()
    model = bench.BlackBoxModel(processor).to(device)
    groups: dict[tuple, list] = {}
    for parameter in model.parameters():
        hints = getattr(parameter, "_optim", {})
        groups.setdefault(tuple(sorted(hints.items())), []).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": p, **dict(h)} for h, p in groups.items()], lr=0.01
    )
    xs, ys, context = batch(config, "train", device)
    controls = torch.zeros(xs.shape[0], 0, device=device)
    started, history = time.perf_counter(), []
    for step in range(steps):
        if step in HALVINGS:
            for group in optimizer.param_groups:
                group["lr"] /= 2
        model.reset_states()
        out = model(xs, controls)
        total = loss(out[..., context:], ys[..., context:])[-1]
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            history.append([step, float(total)])
            print(run_id, step, f"{float(total):.5f}", flush=True)
    if smoke:
        peak = torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else 0
        print(
            f"smoke: {(time.perf_counter() - started) / steps:.3f} s/step, "
            f"{peak:.1f} GiB"
        )
        return
    run_dir = bench.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    record = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "benchmark": "KIPR known-answer bench (demo/kipr_known_answer/predictions.md)",
        "config": config,
        "seed": seed,
        "steps": steps,
        "halvings": HALVINGS,
        "advance": ADVANCES[TAG],
        "tag": TAG,
        "model": model_args(),
        "device": device,
        "final_loss": history[-1][1],
        "loss_history": history,
        "minutes": (time.perf_counter() - started) / 60,
    }
    (run_dir / "record.json").write_text(json.dumps(record, indent=1) + "\n")
    with bench.RUNS_LOG.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({k: v for k, v in record.items() if k != "loss_history"}) + "\n"
        )


def verdicts(result: dict) -> dict:
    """Readings fixed in predictions.md: per level (-12 and -6 dBFS), the median over
    seeds of each error; fidelity criteria level by level, so errors cannot cancel."""
    readings = {}
    for name, config in CONFIGS.items():
        runs = [r for r in result["runs"].values() if r["config"] == config]
        if len(runs) < 3:
            continue
        e2, e3 = (
            {
                level: statistics.median(r["tones"][f"1000_{level}"][key] for r in runs)
                for level in LEVELS
            }
            for key in ("e_h2_db", "e_h3_db")
        )
        readings[name] = {
            "e_h2_by_level": e2,
            "e_h3_by_level": e3,
            "e_primary": (e2[-12] + e2[-6]) / 2,
            "h3_ok": all(abs(e3[lv]) <= 3 for lv in (-12, -6)),
            "h2_ok": all(abs(e2[lv]) <= 5 for lv in (-12, -6)),
        }
    if len(readings) == 2:
        asym, sym = readings["asym"], readings["sym"]
        valid = result["instrument_gap_db"] <= 1.0
        aligned = TAG == "" or all(r["aligned"] for r in result["runs"].values())
        control = asym["h2_ok"] and asym["h3_ok"] and sym["h3_ok"]
        readings["aligned"] = aligned
        if not valid:
            verdict = "aucune lecture (instrument)"
        elif not aligned:
            verdict = "aucune lecture (alignement)"
        elif not control:
            verdict = "non concluant (témoin positif ou H3 de la cible symétrique)"
        elif sym["e_primary"] >= 15:
            verdict = "plancher"
        elif sym["e_primary"] <= 5:
            verdict = "pas de plancher"
        else:
            verdict = "plancher partiel"
        readings["control_ok"] = control
        readings["verdict"] = verdict
    return readings


@torch.no_grad()
def run_model(run_id: str, x: np.ndarray) -> np.ndarray:
    processor, _ = processor_and_loss()
    model = bench.BlackBoxModel(processor)
    model.load_state_dict(
        torch.load(bench.RUNS_DIR / run_id / "model.pt", map_location="cpu")
    )
    model.eval().reset_states()
    signal = torch.tensor(x, dtype=torch.float32)[None, None]
    return model(signal, torch.zeros(1, 0))[0, 0].double().numpy()


def lag(out: np.ndarray, target: np.ndarray, most: int = 60) -> int:
    """Shift L in [-most, most] maximising the correlation of out[n] with
    target[n + L]; positive: the target comes later than the model."""
    core = out[most:-most]
    return max(
        range(-most, most + 1),
        key=lambda s: float(np.dot(core, target[most + s : len(target) - most + s])),
    )


def esr(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(((ref - pred) ** 2).sum() / (ref**2).sum())


def measure() -> None:
    torch.set_num_threads(16)
    seg = segments()
    result = {"runs": {}, "targets": {}, "instrument_gap_db": instrument()}
    for name, config in CONFIGS.items():
        x, y = sequences(config)
        target = {
            f"{f0}_{level}": ratios(y, seg, f0, level)
            for f0 in (300, 1000, 3000)
            for level in LEVELS
        }
        result["targets"][config] = target
        for seed in (42, 43, 44):
            run_id = run_name(name, seed)
            if not (bench.RUNS_DIR / run_id / "record.json").exists():
                print(f"{run_id}: absent")
                continue
            out = run_model(run_id, x)
            train = [seg[f"train_noise_{lv}"][1:] for lv in (-36, -24, -12, -6)]
            inner = sum(float(np.dot(out[a:b], y[a:b])) for a, b in train)
            sign = -1.0 if inner < 0 and TAG in POLARITY_NORMALIZED else 1.0
            out = sign * out
            entry = {
                "config": config,
                "seed": seed,
                "tones": {},
                "test_esr": {},
                "train_inner_sign": 1 if inner >= 0 else -1,
                "polarity_applied": sign,
            }
            for key, t in target.items():
                f0, level = (int(v) for v in key.split("_"))
                m = ratios(out, seg, f0, level)
                entry["tones"][key] = {
                    "e_h2_db": m["h2_db"] - t["h2_db"],
                    "e_h3_db": m["h3_db"] - t["h3_db"],
                    "model_h2_db": m["h2_db"],
                }
            entry["lag"] = {}
            for level in (-36, -24, -12, -6):
                _, a, b = seg[f"test_noise_{level}"]
                entry["test_esr"][str(level)] = esr(out[a:b], y[a:b])
                entry["lag"][str(level)] = lag(out[a:b], y[a:b])
            entry["aligned"] = all(
                abs(entry["lag"][lv]) <= 2 and entry["test_esr"][lv] < 1
                for lv in entry["lag"]
            )
            result["runs"][run_id] = entry
    readings = verdicts(result)
    result["readings"] = readings
    print(json.dumps(readings, indent=1, ensure_ascii=False))
    (HERE / f"known_answer{'_' + TAG if TAG else ''}.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["instrument", "train", "measure"])
    parser.add_argument("config", nargs="?", choices=list(CONFIGS))
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--smoke", action="store_true", help="no record, no checkpoint")
    parser.add_argument("--tag", choices=list(ADVANCES), default="")
    args = parser.parse_args()
    global TAG
    TAG = args.tag
    if args.action == "instrument":
        instrument()
    elif args.action == "train":
        train(args.config, args.seed, args.steps, args.smoke)
    else:
        measure()


if __name__ == "__main__":
    main()
