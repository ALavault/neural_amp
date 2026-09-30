"""Post-hoc look at paired_ssm_wavenet_foh_s688 (predictions_foh_training.md).

CPU, read-only. Per Big Muff test segment: best output->target lag, and the interior
ESR at lag 0 and at that lag (as diagnosis/ffuzz/measure.py).
Usage: foh_lag.py <run_id>... Writes demo/sr_transfer/foh_lag.json.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path("/fastdata/lavaulta/neural_amp")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnosis/ffuzz"))
import product_nablafx_bench as bench  # noqa: E402
from measure import ESR, best_lag, interior_esr  # noqa: E402
from nablafx.core import BlackBoxModel  # noqa: E402


def load(run_id: str) -> BlackBoxModel:
    record = json.loads((bench.RESULTS_DIR / f"{run_id}.json").read_text())
    processor = bench.build_processor(
        argparse.Namespace(model=record["model"], **record["ssm"])
    )
    path = bench.RUNS_DIR / f"nablafx_{run_id}/checkpoints/last.ckpt"
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    prefix = "model.processor."
    processor.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    )
    return BlackBoxModel(processor).eval()


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(16)
    data = bench.data_module("test")
    data.setup("test")
    dataset = data.test_dataset
    out = {}
    for run in sys.argv[1:]:
        model = load(run)
        rows = []
        for start in range(0, len(dataset), 8):
            items = [dataset[i] for i in range(start, min(start + 8, len(dataset)))]
            parts = list(zip(*items, strict=True))
            x, y = torch.stack(parts[0]), torch.stack(parts[1])
            model.reset_states()
            controls = (
                torch.stack(parts[2]) if len(parts) > 2 else torch.zeros(len(items), 0)
            )
            pred = model(x, controls)
            for k in range(len(items)):
                p, t = pred[k, 0], y[k, 0]
                lag = best_lag(p, t)
                rows.append(
                    {
                        "esr": float(ESR(p.view(1, 1, -1), t.view(1, 1, -1))),
                        "lag": lag,
                        "lag0": interior_esr(p, t, 0),
                        "best": interior_esr(p, t, lag),
                    }
                )
        n = len(rows)
        out[run] = {
            "segments": n,
            "lags": dict(sorted(Counter(r["lag"] for r in rows).items())),
            "esr_mean": sum(r["esr"] for r in rows) / n,
            "interior_lag0": sum(r["lag0"] for r in rows) / n,
            "interior_best_lag": sum(r["best"] for r in rows) / n,
        }
        print(run, out[run], flush=True)
    Path(__file__).with_name("foh_lag.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
