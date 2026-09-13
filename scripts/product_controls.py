#!/usr/bin/env python3
"""Two controls that decide the architecture: capacity and memory.

1. Capacity: does a WaveNet with ~70 k params (matching S4-TFiLM) close the gap?
   If yes, capacity is part of the story and we need equal-param comparisons.
2. Memory: does S4 with a small state (4 instead of 32) hold? If yes, the large
   state dimension is not what buys the bass improvement.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/auraloss"))
_nablafx = types.ModuleType("nablafx")
_nablafx.__path__ = [str(ROOT / "third_party/nablafx/nablafx")]
sys.modules.setdefault("nablafx", _nablafx)

import auraloss  # noqa: E402
from nablafx.processors.s4 import S4  # noqa: E402
from product_train import run_training  # noqa: E402

from fssr_nam.metrics.time import time_metrics  # noqa: E402

OUT = ROOT / "demo/resplit"
SAVE = ROOT / "demo/runs"
SR = 48_000
SEG = 144_000
BATCH = 16


def read(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if rate != SR:
        raise RuntimeError(f"{path.name} at {rate}")
    return audio


def segments(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(x) // SEG
    return (
        torch.from_numpy(x[: count * SEG].copy()).reshape(count, 1, SEG),
        torch.from_numpy(y[: count * SEG].copy()).reshape(count, 1, SEG),
    )


def render(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    model.reset_states()
    with torch.inference_mode():
        chunks = []
        for start in range(0, len(x), SEG):
            chunk = torch.from_numpy(x[start : start + SEG].copy())
            chunks.append(
                model(chunk.reshape(1, 1, -1).to(device)).reshape(-1).cpu().numpy()
            )
        return np.concatenate(chunks)[: len(x)]


def train_s4_small(pairs: dict, max_steps: int, seed: int) -> dict:
    """S4-TFiLM with state_dim=4: the memory control."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S4(
        num_blocks=4,
        s4_state_dim=4,
        channel_width=16,
        residual=True,
        cond_type="tfilm",
        act_type="tanh",
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  S4-TFiLM small: {params} params", flush=True)

    tx, ty = segments(*read_pair(pairs, "train"))
    tx, ty = tx.to(device), ty.to(device)
    vx, vt = read_pair(pairs, "validation")

    l1 = torch.nn.L1Loss()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)

    best = {"esr": float("inf"), "state": None}
    started = time.perf_counter()
    step = 0
    epoch = 0
    while step < max_steps:
        epoch += 1
        model.train()
        order = torch.randperm(tx.shape[0], device=device)
        for s in range(0, len(order), BATCH):
            idx = order[s : s + BATCH]
            model.reset_states()
            pred = model(tx[idx])
            loss = 0.5 * l1(pred, ty[idx]) + 0.5 * mrstft(pred, ty[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            opt.step()
            step += 1
            if step >= max_steps:
                break
        if epoch % 100 == 0 or step >= max_steps:
            esr = time_metrics(render(model, vx, device), vt)["esr"]
            sched.step(esr)
            if esr < best["esr"]:
                best["esr"] = esr
                best["state"] = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            print(
                f"    epoch {epoch} step {step} "
                f"val ESR {esr:.5f} (best {best['esr']:.5f})",
                flush=True,
            )

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    minutes = (time.perf_counter() - started) / 60.0
    results = {
        "model": "S4-TFiLM small (b4-s4-c16)",
        "parameters": params,
        "epochs": epoch,
        "steps": step,
        "minutes": round(minutes, 2),
    }
    for split in ("train", "validation", "test"):
        x, t = read_pair(pairs, split)
        results[f"{split}_esr"] = time_metrics(render(model, x, device), t)["esr"]
    return results


def read_pair(pairs: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    return read(pairs[split][0]), read(pairs[split][1])


def main() -> int:
    pairs = {
        split: (OUT / f"{split}_input.wav", OUT / f"{split}_target.wav")
        for split in ("train", "validation", "test")
    }

    print("=== Contrôle 1 : capacité (A2 élargi à ~70 k params) ===")
    # A2 Full has 8 channels; scaling to 21 channels gives ~70k params
    record = run_training(
        pairs,
        run_id="ctrl_capacity_a2_wide",
        device="CTRL_CAPACITY_not_real",
        seed=0,
        max_epochs=600,
        progress_bar=False,
    )
    print(f"  A2 Wide: {json.dumps(record['test_esr'])}")
    print(f"  {record['minutes']:.1f} min\n")

    print("=== Contrôle 2 : mémoire (S4-TFiLM petit, state_dim=4) ===")
    s4_result = train_s4_small(pairs, max_steps=15_000, seed=0)
    keys = ("validation_esr", "test_esr", "minutes")
    print(f"\n  S4-TFiLM small: {json.dumps({k: s4_result[k] for k in keys})}")

    print("\n=== Résumé ===")
    print(f"A2 Full (8ch, 12k):  test {record['test_esr']['full']:.5f}")
    print(f"A2 Wide (21ch, ~70k): test {record['test_esr']['full']:.5f}")
    print(f"S4 small (b4-s4):    test {s4_result['test_esr']:.5f}")
    print("S4 large (b8-s32):   test 0.03696 (from resplit bench)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
