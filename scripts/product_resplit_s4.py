#!/usr/bin/env python3
"""Train S4-TFiLM large on the resplit Big Muff data.

Saves a checkpoint every 500 steps so a kill on this shared machine does not
lose hours of work. Pass --resume to continue from the last checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/auraloss"))
_nablafx = types.ModuleType("nablafx")
_nablafx.__path__ = [str(ROOT / "third_party/nablafx/nablafx")]
sys.modules.setdefault("nablafx", _nablafx)

import auraloss  # noqa: E402
from nablafx.processors.s4 import S4  # noqa: E402

from fssr_nam.metrics.time import time_metrics  # noqa: E402

OUT = ROOT / "demo/resplit"
SAVE_DIR = ROOT / "demo/runs/resplit_sota_S4-TFiLM_large"
SR = 48_000
SEG = 144_000
BATCH = 16
MAX_STEPS = 15_000
CKPT_EVERY = 500


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
        tensor = torch.from_numpy(x).reshape(1, 1, -1).to(device)
        return model(tensor).reshape(-1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S4(
        num_blocks=8,
        s4_state_dim=32,
        channel_width=16,
        residual=True,
        cond_type="tfilm",
        act_type="tanh",
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"S4-TFiLM large, {params} params", flush=True)

    tx, ty = segments(read(OUT / "train_input.wav"), read(OUT / "train_target.wav"))
    tx, ty = tx.to(device), ty.to(device)
    vx = read(OUT / "validation_input.wav")
    vt = read(OUT / "validation_target.wav")

    l1 = torch.nn.L1Loss()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)

    best_esr = float("inf")
    best_state = None
    start_step = 0
    epoch = 0

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = SAVE_DIR / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        best_esr = ckpt["best_esr"]
        best_state = ckpt["best_state"]
        start_step = ckpt["step"]
        epoch = ckpt["epoch"]
        print(
            f"  repris depuis le pas {start_step}, best val ESR {best_esr:.5f}",
            flush=True,
        )

    started = time.perf_counter()
    step = start_step
    while step < MAX_STEPS:
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
            if step >= MAX_STEPS:
                break
            if step % CKPT_EVERY == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "best_esr": best_esr,
                        "best_state": best_state,
                        "step": step,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )
        if epoch % 50 == 0 or step >= MAX_STEPS:
            esr = time_metrics(render(model, vx, device), vt)["esr"]
            sched.step(esr)
            if esr < best_esr:
                best_esr = esr
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            print(
                f"  epoch {epoch} step {step} val ESR {esr:.5f} (best {best_esr:.5f})",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(best_state, SAVE_DIR / "state.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "best_esr": best_esr,
            "best_state": best_state,
            "step": step,
            "epoch": epoch,
        },
        ckpt_path,
    )

    minutes = (time.perf_counter() - started) / 60.0
    results = {
        "model": "S4-TFiLM large",
        "epochs": epoch,
        "steps": step,
        "minutes": round(minutes, 2),
        "parameters": params,
    }
    for split in ("train", "validation", "test"):
        x = read(OUT / f"{split}_input.wav")
        t = read(OUT / f"{split}_target.wav")
        results[f"{split}_esr"] = time_metrics(render(model, x, device), t)["esr"]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
