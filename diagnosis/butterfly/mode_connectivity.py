#!/usr/bin/env python3
"""Do the divergent children sit in the same basin as the control? (Read-only, CPU.)

Linear mode connectivity (Frankle et al. 2020): interpolate the weights between the
control and each child, theta(a) = (1 - a) theta_control + a theta_child, and measure the
test ESR along the way. A bump above the straight line between the two endpoints, the
barrier, means the two solutions sit in separate basins; no bump means one basin.

Alignment is unnecessary here, unlike the usual setting: the children descend from one
parent by a one-float32-step nudge, so channels and poles are in the same order by
construction. The checkpoints hold no normalisation buffers, so nothing has to be
recalibrated after interpolation.

The polarity guard negates the output layer, so two endpoints whose flip counts differ in
parity would cross a zero-output model at a = 0.5 and show a barrier that is only a sign
convention. The script refuses such a pair rather than measuring it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import product_nablafx_bench as bench  # noqa: E402
import torch  # noqa: E402

RECORDS = ROOT / "demo/butterfly"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONTROL = "butterfly_ssm_seed42_f100_decide_k0"
CHILDREN = [
    f"butterfly_ssm_seed42_f100_{arm}_k{k}" for arm in ("decide", "replay") for k in (1, 2, 3, 4)
]


def processor() -> torch.nn.Module:
    return bench.build_processor(
        SimpleNamespace(
            model="ssm-wavenet",
            num_blocks=8,
            channels=16,
            state_dim=4,
            output_act="tanh",
            discretization="zoh",
        )
    )


def weights(name: str) -> dict[str, torch.Tensor]:
    state = torch.load(
        bench.RUNS_DIR / f"nablafx_{name}/checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    return {
        key.removeprefix("model.processor."): value
        for key, value in state.items()
        if key.startswith("model.processor.")
    }


def distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float((a[k] - b[k]).pow(2).sum()) for k in a))


def esr(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> list[float]:
    """Test ESR per segment. The protocol's figure is the mean over the 12 segments."""
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs), 8):
            model.reset_states()
            predictions.append(model(inputs[start : start + 8]))
    predictions = torch.cat(predictions)
    per_segment = (predictions - targets).pow(2).sum((1, 2)) / targets.pow(2).sum((1, 2))
    return [float(v) for v in per_segment]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    torch.set_num_threads(8)
    data = bench.data_module("test")
    data.setup("test")
    inputs = torch.stack([x for x, _ in data.test_dataset])
    targets = torch.stack([y for _, y in data.test_dataset])

    model = processor()
    control = weights(CONTROL)
    record_control = json.loads((RECORDS / f"{CONTROL}.json").read_text())
    model.load_state_dict(control)
    segments_control = esr(model, inputs, targets)
    recorded = record_control["test_last"]["metric/test/esr"]
    print(f"{CONTROL}: ESR here {mean(segments_control):.5f}, recorded {recorded:.5f}")

    out = {
        "control": CONTROL,
        "esr_control": mean(segments_control),
        "alphas": list(ALPHAS),
        "children": {},
    }
    for child in CHILDREN:
        record = json.loads((RECORDS / f"{child}.json").read_text())
        assert (
            len(record["polarity_flips"]) % 2
            == len(record_control["polarity_flips"]) % 2
        ), child
        theta = weights(child)
        segments = [segments_control]
        for alpha in ALPHAS[1:]:
            model.load_state_dict(
                {k: (1 - alpha) * control[k] + alpha * theta[k] for k in control}
            )
            segments.append(esr(model, inputs, targets))
        curve = [mean(s) for s in segments]
        line = [(1 - a) * curve[0] + a * curve[-1] for a in ALPHAS]
        barrier = max(c - l for c, l in zip(curve, line))
        # A barrier on the mean could come from one segment: count the segments whose
        # own midpoint stands above their own straight line.
        middle = segments[ALPHAS.index(0.5)]
        above = sum(
            middle[i] > (segments[0][i] + segments[-1][i]) / 2
            for i in range(len(middle))
        )
        out["children"][child] = {
            "esr": curve,
            "esr_per_segment": segments,
            "barrier": barrier,
            "barrier_relative": barrier / curve[0],
            "segments_above_midline": above,
            "l2_distance": distance(theta, control),
            "recorded_esr": record["test_last"]["metric/test/esr"],
        }
        print(
            f"{child}: ESR {[round(v, 4) for v in curve]}"
            f" barrier {barrier:+.4f} ({barrier / curve[0]:+.0%})"
            f" segments au-dessus {above}/12"
            f" ||dtheta|| {out['children'][child]['l2_distance']:.4f}"
        )

    (Path(__file__).parent / "mode_connectivity.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
