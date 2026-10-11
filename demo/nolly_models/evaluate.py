#!/usr/bin/env python3
"""Readings of demo/nolly_models/predictions.md (D-0015). Read-only, GPU (fp32, no
TF32), last checkpoints.

Arms C, GREY, L, F3 (seeds 42-44) on the Nolly renders at 48 kHz, read as the dataset
was built (common advance of nolly_48k/manifest.json). P Nolly: mean of ln ESR on
di_test over the 8 combination-test takes (settings never trained on); d_i = s(arm) -
s(reference) per seed, labels and acceptance bound of demo/train_length. Descriptive:
the same over the 29 train takes (seen settings, unseen audio); the delay-compensated
co-reading (rule of coreading_delay.py: one common delay on its grid, the one that
minimises the median ESR over the test takes on di_validation, applied to di_test);
history slopes against the device on the 8 test takes; the device's slopes along the
gain ladder. Writes demo/nolly_models/evaluation.json.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from fssr_nam.product import nolly
from fssr_nam.product.history import history_deviation, history_slopes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Device-generic helpers of the deeper grey-box reading (load, checks, compare, memory).
_spec = importlib.util.spec_from_file_location(
    "greybox_deep_evaluate", ROOT / "demo/greybox_deep/evaluate.py"
)
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)
cd, tl, ev = gd.cd, gd.tl, gd.ev

SEEDS = (42, 43, 44)
ARMS = {
    "C": "nolly_C_s{}",
    "GREY": "nolly_grey_s{}",
    "L": "nolly_gbL_s{}",
    "F3": "nolly_gbF3_s{}",
}
COMPARISONS = (
    ("F3", "GREY", True),
    ("F3", "C", True),
    ("L", "GREY", False),
    ("L", "C", False),
)
GAIN_LADDER = (30, 16, 1, 6, 7)  # gain 0, 2.5, 5, 7.5, 10, the other knobs at 5
DATASET = nolly.DATA / "nolly_48k"


def knobs(take: dict) -> torch.Tensor:
    return torch.tensor([v / 10 for v in take["knobs"]])


def common_delay(model, takes: list[dict], validation: dict) -> float:
    """cd.common_delay with the five knobs of a Nolly take."""
    curves = []
    for take in takes:
        x, y = validation[take["order"]]
        shift = cd.Shifter(cd.run(model, x, knobs(take)))
        curves.append([ev.esr(shift(d), y) for d in cd.DELAYS])
    return float(cd.DELAYS[int(np.argmin(np.median(np.array(curves), axis=0)))])


def edge_effect(take: dict, renders: Path, advance: int) -> float:
    """cd.edge_effect on a render: di_test delayed alone against delayed with 1 s of
    context on each side, over the scored region."""
    m = nolly.RATE
    alone = nolly.output(take, "di_test", renders=renders, advance=advance)
    context = nolly.output(take, "di_test", renders=renders, advance=advance, margin=m)
    alone = cd.Shifter(alone)(cd.EDGE_DELAY)
    k = m // nolly.DOWN
    return ev.esr(alone, cd.Shifter(context)(cd.EDGE_DELAY)[k:-k])


def reading(model, takes: list[dict], data: dict, validation: dict) -> dict:
    """One model: ln ESR per take on di_test, the aggregates, the co-reading and the
    history slopes on the test takes."""
    tests = [t for t in takes if t["split"] == "test"]
    seen = [t for t in takes if t["split"] == "train"]
    pred = {t["order"]: cd.run(model, data[t["order"]][0], knobs(t)) for t in takes}
    ln_esr = {o: math.log(ev.esr(p, data[o][1])) for o, p in pred.items()}
    tau = common_delay(model, tests, validation)
    aligned = {
        t["order"]: math.log(
            ev.esr(cd.Shifter(pred[t["order"]])(tau), data[t["order"]][1])
        )
        for t in tests
    }
    return {
        "s_test": statistics.mean(ln_esr[t["order"]] for t in tests),
        "s_seen": statistics.mean(ln_esr[t["order"]] for t in seen),
        "ln_esr_by_take": ln_esr,
        "tau": tau,
        "grid_edge": tau in (cd.DELAYS[0], cd.DELAYS[-1]),
        "s_test_aligned": statistics.mean(aligned.values()),
        "ln_esr_aligned_by_take": aligned,
        "history": [
            history_deviation(*data[t["order"]], pred[t["order"]], 48_000)
            for t in tests
        ],
    }


def aligned_reading(runs: dict, arm: str, reference: str, raw: dict) -> dict:
    """gd.aligned_reading on the Nolly runs: a seed whose run or reference is on a grid
    edge is dropped; with a seed dropped, no label."""
    d_i = {}
    for seed in SEEDS:
        a, r = (runs[gd.attempt(ARMS[k].format(seed))] for k in (arm, reference))
        if not (a["grid_edge"] or r["grid_edge"]):
            d_i[seed] = a["s_test_aligned"] - r["s_test_aligned"]
    out = {"d_i": d_i, "d": statistics.mean(d_i.values()) if d_i else None}
    if len(d_i) == len(SEEDS):
        label = tl.verdict(list(d_i.values()))
        accept = tl.acceptable(list(d_i.values()))
        same = (label, accept) == (raw["label"], raw["acceptable"])
        out |= {
            "label": label,
            "acceptable": accept,
            "against_raw": "robuste au calage" if same else "divergence",
        }
    return out


@torch.no_grad()
def main() -> None:
    # The renders and the advance the dataset was built from (its manifest).
    manifest = json.loads((DATASET / "manifest.json").read_text())
    renders, advance = Path(manifest["renders"]), manifest["advance_48k"]
    takes = nolly.model_takes()
    tests = [t for t in takes if t["split"] == "test"]
    x, xv = nolly.source("di_test"), nolly.source("di_validation")
    data = {
        t["order"]: (x, nolly.output(t, "di_test", renders=renders, advance=advance))
        for t in takes
    }
    validation = {
        t["order"]: (
            xv,
            nolly.output(t, "di_validation", renders=renders, advance=advance),
        )
        for t in tests
    }
    runs, raw, seen = {}, {}, {}
    for arm, pattern in ARMS.items():
        for seed in SEEDS:
            run_id = gd.attempt(pattern.format(seed))
            if not gd.exists(run_id):
                print(f"{run_id}: absent", flush=True)
                continue
            model = gd.load(run_id, "nolly")
            entry = {
                "arm": arm,
                "seed": seed,
                **reading(model, takes, data, validation),
                "parameters": gd.parameters(model),
                **gd.checks(run_id),
            }
            # predictions.md, before reading: 15 000 steps, the queue's check ok
            # (segments, and the split of C at the same seed), finite values.
            entry["valid"] = (
                entry["global_step"] == 15_000
                and entry["check_ok"] is True
                and math.isfinite(entry["s_test"])
                and math.isfinite(entry["s_test_aligned"])
            )
            runs[run_id] = entry
            if entry["valid"]:
                raw.setdefault(arm, {})[seed] = entry["s_test"]
                # Seen settings: read only from finite values, the arm otherwise
                # incomplete for that reading.
                if math.isfinite(entry["s_seen"]):
                    seen.setdefault(arm, {})[seed] = entry["s_seen"]
            print(
                f"{run_id}: s_test {entry['s_test']:.3f}, aligned"
                f" {entry['s_test_aligned']:.3f} (tau {entry['tau']:+.2f})",
                flush=True,
            )
    complete = {a for a in ARMS if len(raw.get(a, {})) == len(SEEDS)}
    seen_complete = {a for a in ARMS if len(seen.get(a, {})) == len(SEEDS)}
    edge = edge_effect(tests[0], renders, advance)
    # Bound of coreading_delay.py: 1 % of the smallest delay-compensated ESR.
    smallest = min(
        (
            math.exp(v)
            for r in runs.values()
            if r["valid"] and not r["grid_edge"]
            for v in r["ln_esr_aligned_by_take"].values()
        ),
        default=0.0,
    )
    edge_ok = edge < 0.01 * smallest
    readings = {}
    for arm, reference, primary in COMPARISONS:
        if {arm, reference} <= complete:
            row = {"raw": gd.compare(raw, arm, reference, primary, {"C": "C"})}
            if {arm, reference} <= seen_complete:
                row["seen"] = gd.compare(seen, arm, reference, False, {})
            if edge_ok:
                row["aligned"] = aligned_reading(runs, arm, reference, row["raw"])
            readings[f"{arm}_vs_{reference}"] = row
    result = {
        "advance_48k": advance,
        "runs": runs,
        "readings": readings,
        "edge_effect": {"energy": edge, "bound": 0.01 * smallest, "reliable": edge_ok},
        "history": {
            arm: gd.memory(
                [h for r in runs.values() if r["arm"] == arm for h in r["history"]]
            )
            for arm in complete
        },
        "history_references": gd.HISTORY_REFERENCES,
        "device_gain_ladder": {
            str(o): history_slopes(*data[o], 48_000) for o in GAIN_LADDER
        },
    }
    (HERE / "evaluation.json").write_text(
        json.dumps(result, indent=1, default=str) + "\n"
    )
    print(json.dumps(readings, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
