#!/usr/bin/env python3
"""Per-segment measurements for DIAGNOSIS_ffuzz.md. (Read-only on the runs, CPU.)

Runs the last checkpoint of param_ssm_wavenet_ffuzz_r2 and param_s4_ffuzz_r2 on the
F-Fuzz test set, batch by batch as the bench does (batches of 8 segments of one
setting, states reset per batch), and keeps per segment what the hypotheses of
DIAGNOSIS_ffuzz.md section 3 need:

- esr: nablafx's esr_loss on the segment alone (the bench mean is the mean of these);
- target and input energy, file, offset, marker segment (first or last of its file);
- lag: the shift d in [-MAX_LAG, MAX_LAG] maximising the correlation of the model
  output with the target, and the interior ESR at lag 0 and at that lag (H1);
- the share of error energy above 5 kHz, and the interior ESR after a 5 kHz low-pass
  of output and target (H4).

Writes diagnosis/ffuzz/segments.json. Usage: measure.py [--limit N] (smoke test).
FIX_ffuzz.md section 4, on the adv4 runs with the targets they were trained on:
measure.py --runs param_ssm_wavenet_ffuzz_adv4_r2 param_s4_ffuzz_adv4
--target-advance 4 --out segments_adv4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import product_nablafx_bench as bench  # noqa: E402
from nablafx.core import BlackBoxModel  # noqa: E402
from nablafx.evaluation.registry import EvaluationRegistry  # noqa: E402

RUNS = ("param_ssm_wavenet_ffuzz_r2", "param_s4_ffuzz_r2")
RATE = 48_000
MAX_LAG = 400
TRIM = 1_000  # interior: drop the edges so shifted and unshifted ESR use the same span
CUTOFF_HZ = 5_000
ESR = EvaluationRegistry.get_function("esr_loss")


def load(run_id: str) -> torch.nn.Module:
    record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
    args = argparse.Namespace(
        model=record["model"], device="ffuzz", **(record["ssm"] or {})
    )
    processor = bench.build_processor(args)
    path = bench.RUNS_DIR / f"nablafx_{run_id}/checkpoints/last.ckpt"
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.processor."
    processor.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    )
    return BlackBoxModel(processor).eval()


def interior_esr(pred: torch.Tensor, target: torch.Tensor, lag: int) -> float:
    """ESR on [TRIM, len - TRIM) of the target against pred shifted by lag
    (pred[n - lag] compared with target[n])."""
    n = target.shape[-1]
    t = target[TRIM : n - TRIM]
    p = pred[TRIM - lag : n - TRIM - lag]
    return float(((t - p) ** 2).sum() / (t**2).sum())


def best_lag(pred: torch.Tensor, target: torch.Tensor) -> int:
    n = target.shape[-1]
    size = 1 << (2 * n - 1).bit_length()
    corr = torch.fft.irfft(
        torch.fft.rfft(target, size) * torch.fft.rfft(pred, size).conj(), size
    )
    lags = torch.cat([corr[: MAX_LAG + 1], corr[-MAX_LAG:]])
    index = int(lags.argmax())
    return index if index <= MAX_LAG else index - 2 * MAX_LAG - 1


def lowpass(x: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(x.shape[-1], 1 / RATE)
    spectrum[freqs > CUTOFF_HZ] = 0
    return torch.fft.irfft(spectrum, x.shape[-1])


@torch.no_grad()
def measure(model: torch.nn.Module, dataset, limit: int | None) -> list[dict]:
    groups: dict[str, list[int]] = {}
    for index in range(len(dataset)):
        setting = Path(dataset.samples[index]["target_file"]).parent.name
        groups.setdefault(setting, []).append(index)
    rows = []
    for setting, members in sorted(groups.items()):
        members = members[:limit] if limit else members
        for start in range(0, len(members), 8):
            chunk = members[start : start + 8]
            items = [dataset[i] for i in chunk]
            x, y, controls = (torch.stack(parts) for parts in zip(*items, strict=True))
            model.reset_states()
            pred = model(x, controls)
            for k, index in enumerate(chunk):
                sample = dataset.samples[index]
                p, t, i = pred[k, 0], y[k, 0], x[k, 0]
                error = t - p
                spectrum = torch.fft.rfft(error).abs() ** 2
                freqs = torch.fft.rfftfreq(error.shape[-1], 1 / RATE)
                lag = best_lag(p, t)
                n = t.shape[-1]
                rows.append(
                    {
                        "setting": setting,
                        "file": Path(sample["input_file"]).name.split(".")[0],
                        "offset": sample["offset"],
                        "marker": sample["offset"] == 0
                        or sample["offset"] + 2 * n > sample["frames"],
                        "esr": float(ESR(p.view(1, 1, -1), t.view(1, 1, -1))),
                        "target_energy": float((t**2).mean()),
                        "input_energy": float((i**2).mean()),
                        "error_energy": float((error**2).mean()),
                        "hf_error_share": float(
                            spectrum[freqs > CUTOFF_HZ].sum() / spectrum.sum()
                        ),
                        "lag": lag,
                        "esr_interior_lag0": interior_esr(p, t, 0),
                        "esr_interior_best_lag": interior_esr(p, t, lag),
                        "esr_interior_lowpass": interior_esr(lowpass(p), lowpass(t), 0),
                    }
                )
        print(setting, len(rows), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--target-advance", type=int, default=0)
    parser.add_argument("--out", default="segments.json")
    args = parser.parse_args()
    torch.set_num_threads(16)
    bench.TARGET_ADVANCE = args.target_advance
    data = bench.data_module("test", device="ffuzz")
    data.setup("test")
    out = {run: measure(load(run), data.test_dataset, args.limit) for run in args.runs}
    if args.limit:
        print(json.dumps({r: v[:3] for r, v in out.items()}, indent=1))
        return
    path = Path(__file__).parent / args.out
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"written {path}")


if __name__ == "__main__":
    main()
