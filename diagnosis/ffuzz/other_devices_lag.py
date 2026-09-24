#!/usr/bin/env python3
"""Follow-up of DIAGNOSIS_ffuzz.md section 6: do the Big Muff and Marshall targets hide
a latency too? (Read-only, CPU.) Same instrument as measure.py: per test segment, the
lag maximising model output / target correlation, and the interior ESR at lag 0 and at
that lag.

Reading written before running: a lag of 0 in (almost) every segment means the latency
is specific to the F-Fuzz renders; a constant non-zero lag means part of the bench ESR
of that device is the same artefact. Written to other_devices_lag.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from measure import best_lag, interior_esr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import product_nablafx_bench as bench
from nablafx.core import BlackBoxModel

RUNS = {
    "bigmuff": ("paired_ssm_wavenet_s925", "paired_s4_tf_l_16_s925"),
    "marshall": ("param_ssm_wavenet_marshall", "param_s4_marshall"),
}


def load(run: str, device: str) -> BlackBoxModel:
    record = json.loads((bench.RESULTS_DIR / f"{run}.json").read_text())
    args = argparse.Namespace(
        model=record["model"], device=device, **(record["ssm"] or {})
    )
    processor = bench.build_processor(args)
    path = bench.RUNS_DIR / f"nablafx_{run}/checkpoints/last.ckpt"
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.processor."
    processor.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    )
    return BlackBoxModel(processor).eval()


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    out = {}
    for device, runs in RUNS.items():
        data = bench.data_module("test", device=device)
        data.setup("test")
        items = [data.test_dataset[i] for i in range(len(data.test_dataset))]
        for run in runs:
            model = load(run, device)
            rows = []
            for start in range(0, len(items), 8):
                parts = list(zip(*items[start : start + 8], strict=True))
                x, y = torch.stack(parts[0]), torch.stack(parts[1])
                controls = torch.stack(parts[2]) if len(parts) > 2 else None
                model.reset_states()
                pred = model(x, controls)
                for p, t in zip(pred[:, 0], y[:, 0], strict=True):
                    lag = best_lag(p, t)
                    rows.append(
                        {
                            "lag": lag,
                            "esr_lag0": interior_esr(p, t, 0),
                            "esr_best": interior_esr(p, t, lag),
                        }
                    )
            out[run] = rows
            print(
                f"{device:9s} {run:28s} "
                f"lags {Counter(r['lag'] for r in rows).most_common(3)}"
                f"  ESR lag0 {statistics.mean(r['esr_lag0'] for r in rows):.4f}"
                f"  best {statistics.mean(r['esr_best'] for r in rows):.4f}",
                flush=True,
            )
    (Path(__file__).with_name("other_devices_lag.json")).write_text(
        json.dumps(out) + "\n"
    )


if __name__ == "__main__":
    main()
