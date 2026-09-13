#!/usr/bin/env python3
"""Train SSM-WaveNet on the resplit Big Muff data.

Same training loop as the S4-TFiLM control (L1 + MRSTFT, AdamW, ReduceLROnPlateau,
gradient clipping at 10), same segments and batch size. Checkpoints every 500 steps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/auraloss"))
sys.path.insert(0, str(ROOT / "scripts"))

import auraloss  # noqa: E402

from fssr_nam.metrics.time import time_metrics  # noqa: E402
from fssr_nam.models.ssm_wavenet import SSMWaveNet  # noqa: E402

OUT = ROOT / "demo/resplit"
SR = 48_000
SEG = 144_000
BATCH = 16
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
        chunks = []
        for start in range(0, len(x), SEG):
            chunk = torch.from_numpy(x[start : start + SEG].copy())
            chunks.append(
                model(chunk.reshape(1, 1, -1).to(device)).reshape(-1).cpu().numpy()
            )
        return np.concatenate(chunks)[: len(x)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or (
        f"ssm_wavenet_b{args.num_blocks}_c{args.channels}_s{args.state_dim}"
    )
    save_dir = ROOT / "demo/runs" / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSMWaveNet(
        num_blocks=args.num_blocks,
        channels=args.channels,
        state_dim=args.state_dim,
    ).to(device)
    params = model.param_count()
    print(
        f"SSM-WaveNet b{args.num_blocks} c{args.channels} s{args.state_dim}: "
        f"{params:,} params",
        flush=True,
    )

    tx, ty = segments(read(OUT / "train_input.wav"), read(OUT / "train_target.wav"))
    tx, ty = tx.to(device), ty.to(device)
    vx = read(OUT / "validation_input.wav")
    vt = read(OUT / "validation_target.wav")

    l1 = torch.nn.L1Loss()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)

    best_esr = float("inf")
    best_state = None
    start_step = 0
    epoch = 0

    ckpt_path = save_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        best_esr = ckpt["best_esr"]
        best_state = ckpt["best_state"]
        start_step = ckpt["step"]
        epoch = ckpt["epoch"]
        print(f"  resumed from step {start_step}, best {best_esr:.5f}", flush=True)

    started = time.perf_counter()
    step = start_step
    while step < args.max_steps:
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
            if step >= args.max_steps:
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
        if epoch % 50 == 0 or step >= args.max_steps:
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
    torch.save(best_state, save_dir / "state.pt")

    minutes = (time.perf_counter() - started) / 60.0
    results = {
        "model": f"SSM-WaveNet b{args.num_blocks} c{args.channels} s{args.state_dim}",
        "run_id": run_id,
        "parameters": params,
        "epochs": epoch,
        "steps": step,
        "minutes": round(minutes, 2),
    }
    for split in ("train", "validation", "test"):
        x = read(OUT / f"{split}_input.wav")
        t = read(OUT / f"{split}_target.wav")
        results[f"{split}_esr"] = time_metrics(render(model, x, device), t)["esr"]

    runs_log = ROOT / "demo/RUNS.jsonl"
    with runs_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
