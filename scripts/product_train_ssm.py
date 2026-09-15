#!/usr/bin/env python3
"""Train SSM-WaveNet on the resplit Big Muff data.

Same training loop as the S4-TFiLM control (L1 + MRSTFT, AdamW, ReduceLROnPlateau,
gradient clipping at 10), same segments and batch size. Checkpoints every 500 steps.

Improvements over baseline:
  --circuit-init: initialise SSM poles on known RC time constants
  --deriv-weight: add ||d(pred)/dt - d(target)/dt||^2 to the loss
  --act-type sine: replace tanh*sigmoid gate with x + sin(w*x)
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
from fssr_nam.models.ssm_wavenet import BIG_MUFF_TAU_S, SSMWaveNet  # noqa: E402

OUT = ROOT / "demo/resplit"
SR = 48_000
SEG = 144_000
BATCH = 16
CKPT_EVERY = 100


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


def derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on the first difference: forces the model to track transients."""
    dp = pred[:, :, 1:] - pred[:, :, :-1]
    dt = target[:, :, 1:] - target[:, :, :-1]
    return torch.nn.functional.mse_loss(dp, dt)


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
    parser.add_argument(
        "--circuit-init",
        action="store_true",
        help="init SSM poles on the Big Muff RC constants",
    )
    parser.add_argument(
        "--deriv-weight",
        type=float,
        default=0.0,
        help="weight of the derivative loss term",
    )
    parser.add_argument(
        "--act-type",
        choices=["gated", "sine"],
        default="gated",
        help="activation: gated (tanh*sigmoid) or sine (x+sin(wx))",
    )
    parser.add_argument(
        "--output-act",
        choices=["tanh", "softsign", "none"],
        default="tanh",
        help="final activation: tanh (bounded), softsign, or none",
    )
    parser.add_argument(
        "--cosine-lr",
        action="store_true",
        help="use cosine annealing instead of ReduceLROnPlateau",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="override the data directory (default: demo/resplit)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else OUT
    run_id = args.run_id or (
        f"ssm_wavenet_b{args.num_blocks}_c{args.channels}_s{args.state_dim}"
    )
    save_dir = ROOT / "demo/runs" / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    circuit_tau = list(BIG_MUFF_TAU_S) if args.circuit_init else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSMWaveNet(
        num_blocks=args.num_blocks,
        channels=args.channels,
        state_dim=args.state_dim,
        act_type=args.act_type,
        output_act=args.output_act,
        circuit_tau_s=circuit_tau,
    ).to(device)
    params = model.param_count()
    tag = f"SSM-WaveNet b{args.num_blocks} c{args.channels} s{args.state_dim}"
    extras = []
    if args.circuit_init:
        extras.append("circuit-init")
    if args.deriv_weight > 0:
        extras.append(f"deriv={args.deriv_weight}")
    if args.act_type != "gated":
        extras.append(f"act={args.act_type}")
    if args.output_act != "tanh":
        extras.append(f"out={args.output_act}")
    if args.cosine_lr:
        extras.append("cosine")
    if extras:
        tag += " (" + ", ".join(extras) + ")"
    print(f"{tag}: {params:,} params", flush=True)

    tx, ty = segments(
        read(data_dir / "train_input.wav"),
        read(data_dir / "train_target.wav"),
    )
    tx, ty = tx.to(device), ty.to(device)
    vx = read(data_dir / "validation_input.wav")
    vt = read(data_dir / "validation_target.wav")

    l1 = torch.nn.L1Loss()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.cosine_lr:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.max_steps, eta_min=args.lr * 0.01
        )
    else:
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
        print(
            f"  resumed from step {start_step}, best {best_esr:.5f}",
            flush=True,
        )

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
            if args.deriv_weight > 0:
                loss = loss + args.deriv_weight * derivative_loss(pred, ty[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            opt.step()
            if args.cosine_lr:
                sched.step()
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
            if not args.cosine_lr:
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
        "model": tag,
        "run_id": run_id,
        "parameters": params,
        "epochs": epoch,
        "steps": step,
        "minutes": round(minutes, 2),
        "config": {
            "circuit_init": args.circuit_init,
            "deriv_weight": args.deriv_weight,
            "act_type": args.act_type,
        },
        "args": vars(args),
    }
    for split in ("train", "validation", "test"):
        x = read(data_dir / f"{split}_input.wav")
        t = read(data_dir / f"{split}_target.wav")
        results[f"{split}_esr"] = time_metrics(render(model, x, device), t)["esr"]

    runs_log = ROOT / "demo/RUNS.jsonl"
    with runs_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
