#!/usr/bin/env python3
"""Nested per-take oracles of DIAGNOSIS_greybox.md (replacement measure for H1-H3, and
H4). Read-only on runs; GPU, float32 without TF32.

Per test take (split "test", not low signal), di_test at 48 kHz: the scored window
(0.5 s head, 0.1 s tail cut) is cut into 1 s blocks, A = blocks 1, 3, 5, ... and
B = blocks 2, 4, ... (1-based). The model runs over the whole section from a zero
state; ESR on A is the loss, ESR on B the reading. Every fit: Adam, lr 1e-3 after a
linear warm-up of WARMUP steps, cosine decay to 0 at STEPS, gradient norm clipped at
1 (lr chosen on the A loss of take 2, V0, before any B reading). All results are
oracles (fitted on the test take).
- GREY (conditioned, graines 42-44) and C (graine 42): read only.
- V0: DeepGreyBoxAmp.from_greybox(GREY at the take's knobs), fitted; from each GREY
  seed, and from the default initialisation.
- From V0 (seed 42): V1 taps 512, V2 bias, V4 two stages; from V2: V3 bias + slow
  envelopes (50 and 300 ms). Each starts from its parent's fitted coefficients, the
  addition neutral (checked: ESR before the fit equals the parent's).
Writes diagnosis/greybox/oracle.json.

--section di_validation --seeds 42 --variants V1 V4: the same on the validation
section (not test), for the design of demo/greybox_deep; writes
oracle_di_validation.json. --variants V2 V0c V2c --tag controls: the controls (each
parent refitted STEPS more steps, nothing added), oracle_controls.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "demo/train_length"))
import eval_train_length as tl  # noqa: E402

from fssr_nam.models.deep_greybox import DeepGreyBoxAmp  # noqa: E402

ev = tl.ev
FS = 48_000
STEPS = 1_500
WARMUP = 50
SEEDS = (42, 43, 44)
VARIANTS = {
    "V1": ("V0", {"taps": 512}),
    "V2": ("V0", {"bias": True}),
    "V3": ("V2", {"bias": True, "slow_s": (0.05, 0.3)}),
    "V4": ("V0", {"stages": 2}),
    # Controls (Codex review): the parent refitted for STEPS more steps, nothing added,
    # so a variant is read against the same total optimisation.
    "V0c": ("V0", {}),
    "V2c": ("V2", {"bias": True}),
}
DEVICE = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def masks(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean masks of the A and B blocks of the scored window."""
    a, b = FS // 2, n - FS // 10
    block = (np.arange(n) - a) // FS
    inside = (np.arange(n) >= a) & (np.arange(n) < b)
    mask_a = inside & (block % 2 == 0)
    mask_b = inside & (block % 2 == 1)
    return torch.from_numpy(mask_a).to(DEVICE), torch.from_numpy(mask_b).to(DEVICE)


def esr(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((y - pred)[mask] ** 2).sum() / (y[mask] ** 2).sum()


def extend(parent: DeepGreyBoxAmp, **kwargs) -> DeepGreyBoxAmp:
    """New configuration holding the parent's coefficients, additions neutral."""
    child = DeepGreyBoxAmp(branches=parent.branches, **kwargs)
    with torch.no_grad():
        for s in range(parent.stages):
            for k, v in parent.base[s].items():
                child.base[s][k].copy_(v)
            child.contract[s].weight.copy_(parent.contract[s].weight)
        for s in range(parent.stages - 1):
            child.direct[s].copy_(parent.direct[s])
        for mine, theirs in (
            (child.pre_fir, parent.pre_fir),
            (child.post_fir, parent.post_fir),
        ):
            mine.zero_()
            mine[: parent.taps].copy_(theirs)
    return child


def fit(model: DeepGreyBoxAmp, x, y, mask_a, mask_b) -> dict:
    model = model.to(DEVICE)
    with torch.no_grad():
        pred = model(x)[0, 0]
        before = {"A": float(esr(pred, y, mask_a)), "B": float(esr(pred, y, mask_b))}
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda i: (
            min(1.0, (i + 1) / WARMUP)
            * 0.5
            * (1 + math.cos(math.pi * min(i, STEPS) / STEPS))
        ),
    )
    for _ in range(STEPS):
        optimiser.zero_grad()
        loss = esr(model(x)[0, 0], y, mask_a)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
    with torch.no_grad():
        pred = model(x)[0, 0]
        after = {"A": float(esr(pred, y, mask_a)), "B": float(esr(pred, y, mask_b))}
    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    return {"before": before, "after": after, "finite": finite}


@torch.no_grad()
def read(model, x, y, mask_a, mask_b, controls) -> dict:
    model = model.to(DEVICE).eval()
    model.reset_states()
    pred = model(x, controls[None].to(DEVICE))[0, 0]
    return {"A": float(esr(pred, y, mask_a)), "B": float(esr(pred, y, mask_b))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default="di_test")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--no-default", action="store_true")
    parser.add_argument("--tag", default="", help="suffix of the output file")
    args = parser.parse_args()
    seeds = args.seeds
    variants = {k: VARIANTS[k] for k in args.variants}
    torch.manual_seed(0)
    takes = [t for t in ev.takes() if not t["low_signal"] and t["split"] == "test"]
    greys = {s: tl.load(f"vs100r48grey_s{s}", "last") for s in seeds}
    witness = tl.load("vs100r48fast_film_s42", "last")
    result = {
        "takes": [t["order"] for t in takes],
        "section": args.section,
        "steps": STEPS,
        "by_take": {},
    }
    for t in takes:
        loop, fx = (resample_poly(s, 1, 4) for s in ev.audio(t, name=args.section))
        x = torch.from_numpy(loop.astype(np.float32))[None, None].to(DEVICE)
        y = torch.from_numpy(fx.astype(np.float32)).to(DEVICE)
        mask_a, mask_b = masks(len(fx))
        controls = torch.tensor([v / 10 for v in t["gain_contour"]])
        entry = {
            "gain_contour": t["gain_contour"],
            "C_s42": read(witness, x, y, mask_a, mask_b, controls),
        }
        fitted = {}
        for s in seeds:
            entry[f"GREY_s{s}"] = read(greys[s], x, y, mask_a, mask_b, controls)
            start = DeepGreyBoxAmp.from_greybox(greys[s].processor.cpu(), controls)
            entry[f"V0_s{s}"] = fit(start, x, y, mask_a, mask_b)
            fitted[f"V0_s{s}"] = start
        if not args.no_default:
            entry["V0_default"] = fit(DeepGreyBoxAmp(), x, y, mask_a, mask_b)
        fitted["V0"] = fitted["V0_s42"]
        for name, (parent, kwargs) in variants.items():
            child = extend(fitted[parent].cpu(), **kwargs)
            entry[name] = fit(child, x, y, mask_a, mask_b)
            fitted[name] = child
        entry["V0"] = entry["V0_s42"]
        result["by_take"][t["order"]] = entry
        readings = {
            k: round(v["after"]["B"] if "after" in v else v["B"], 4)
            for k, v in entry.items()
            if isinstance(v, dict)
        }
        print(t["order"], readings, flush=True)

    def ratio(numerator: str, denominator: str) -> float:
        values = []
        for e in result["by_take"].values():
            top = e[numerator]
            values.append(
                (top["after"]["B"] if "after" in top else top["B"])
                / e[denominator]["after"]["B"]
            )
        return statistics.median(values)

    result["ratios"] = {
        f"{name}_vs_{parent}": ratio(parent, name)
        for name, (parent, _) in variants.items()
    }
    result["ratios"] |= {
        f"GREY_vs_V0_s{s}": ratio(f"GREY_s{s}", f"V0_s{s}") for s in seeds
    }
    if not args.no_default:
        result["ratios"]["V0_s42_vs_V0_default"] = ratio("V0_s42", "V0_default")
    name = "oracle" + ("" if args.section == "di_test" else f"_{args.section}")
    name += f"_{args.tag}.json" if args.tag else ".json"
    (HERE / name).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["ratios"], indent=1), flush=True)


if __name__ == "__main__":
    main()
