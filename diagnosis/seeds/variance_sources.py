#!/usr/bin/env python3
"""Where does the run-to-run variance come from? (Read-only, CPU.)

Two of the candidate sources can be separated on checkpoints already written.

A - measurement against model. The reported ESR is a mean over twelve test segments. A
first attempt subtracted the sampling variance of that mean from the variance observed
between runs; that is wrong, because every run is evaluated on the SAME twelve segments,
so their difficulty is common and contributes nothing to the differences between runs.
What the segment draw does threaten is the ORDERING of runs, which a bootstrap over the
segments measures directly. The variance itself is split two ways, run against segment,
with their interaction saying whether runs differ in profile or only by a global factor.

B - the stopping rule. Each run keeps two checkpoints: the one the protocol tests,
"last", and the best by validation loss. They stop at different steps. Comparing the
spread of the two selection rules says how much of the variance is "the run stopped
elsewhere" rather than "the model is different".

Runs are SSM-WaveNet with both training changes: three seeds and their repeats, where a
repeat differs only by GPU nondeterminism; and the nine distinct children of fork 100,
which differ by one float32 step.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import product_nablafx_bench as bench  # noqa: E402
import torch  # noqa: E402

SEEDS = [f"ssmzoh_guard_seed{s}{r}" for s in (42, 43, 44) for r in ("", "_repeat")]
CHILDREN = [f"butterfly_ssm_seed42_f100_decide_k{k}" for k in range(5)] + [
    f"butterfly_ssm_seed42_f100_replay_k{k}" for k in range(1, 5)
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


def checkpoints(run: str) -> dict[str, Path]:
    directory = bench.RUNS_DIR / f"nablafx_{run}/checkpoints"
    best = [p for p in directory.glob("*.ckpt") if p.name != "last.ckpt"]
    return {"last": directory / "last.ckpt", **({"best": best[0]} if best else {})}


def per_segment(model: torch.nn.Module, path: Path, x, y) -> list[float]:
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(
        {
            k.removeprefix("model.processor."): v
            for k, v in state.items()
            if k.startswith("model.processor.")
        }
    )
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 8):
            model.reset_states()
            out.append(model(x[start : start + 8]))
    pred = torch.cat(out)
    return [float(v) for v in (pred - y).pow(2).sum((1, 2)) / y.pow(2).sum((1, 2))]


def decompose(table: np.ndarray, name: str) -> dict:
    """Two-way split of the log ESR, plus a bootstrap over the twelve test segments."""
    logs = np.log(table)
    grand = logs.mean()
    run, segment = logs.mean(1) - grand, logs.mean(0) - grand
    interaction = logs - grand - run[:, None] - segment[None, :]
    order = np.argsort(logs.mean(1))
    rng = np.random.default_rng(0)
    kept = sum(
        np.array_equal(
            np.argsort(logs[:, rng.integers(0, logs.shape[1], logs.shape[1])].mean(1)),
            order,
        )
        for _ in range(10_000)
    )
    out = {
        "run_sd": float(run.std(ddof=1)),
        "segment_sd": float(segment.std(ddof=1)),
        "interaction_sd": float(interaction.std(ddof=1)),
        "ranking_kept": kept / 10_000,
    }
    print(
        f"{name}: effet run {out['run_sd']:.3f}, effet segment {out['segment_sd']:.3f}"
        f" (commun, ne separe rien), interaction {out['interaction_sd']:.3f};"
        f" classement conserve par le bootstrap {out['ranking_kept']:.1%}"
    )
    return out


def main() -> None:
    written = Path(__file__).parent / "variance_sources.json"
    if written.exists():
        saved = json.loads(written.read_text())
        for name in ("graines", "enfants f100"):
            for rule in ("last", "best"):
                table = np.array([v[rule] for v in saved[name].values() if rule in v])
                decompose(table, f"{name} [{rule}]")
        return
    torch.set_num_threads(8)
    data = bench.data_module("test")
    data.setup("test")
    x = torch.stack([a for a, _ in data.test_dataset])
    y = torch.stack([b for _, b in data.test_dataset])
    model = processor()

    out: dict = {}
    for name, runs in (("graines", SEEDS), ("enfants f100", CHILDREN)):
        values: dict[str, dict[str, list[float]]] = {}
        for run in runs:
            values[run] = {
                rule: per_segment(model, path, x, y)
                for rule, path in checkpoints(run).items()
            }
            shown = {r: round(statistics.fmean(v), 4) for r, v in values[run].items()}
            print(f"  {run[-24:]:28s} {shown}")
        out[name] = values
        print()
        for rule in ("last", "best"):
            table = np.array([v[rule] for v in values.values() if rule in v])
            if len(table) > 1:
                out.setdefault("decomposition", {})[f"{name}/{rule}"] = decompose(
                    table, f"{name} [{rule}]"
                )
        print()

    (Path(__file__).parent / "variance_sources.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
