#!/usr/bin/env python3
"""Pilot A (diagnosis/butterfly/pilot_A.md): forks of one deterministic SSM-WaveNet run.

parent: train with the benchmark protocol in deterministic mode; at the start of the
    epoch after each fork epoch (so after that epoch's validation, scheduler, early
    stopping and polarity decisions) save checkpoints/fork_epoch<E>.ckpt; stop after
    the last one.
child: resume a fork checkpoint; for k > 0 move every weight by one float32 step;
    train to the end either deciding for itself ("decide": scheduler, early stopping
    and guard restored from the checkpoint) or replaying the learning rate of every
    epoch and the final step of the k = 0 "decide" child ("replay": no early stopping).

Model, data, loss, optimizer and trainer settings are those of
scripts/product_nablafx_bench.py (SSM-WaveNet, zero-order hold, weight-decay exemption,
polarity guard, seed 42). Records go to demo/butterfly/<run_id>.json and
demo/RUNS.jsonl, never to demo/nablafx_bench.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# The bench module puts third_party/auraloss and third_party/nablafx on sys.path.
import product_nablafx_bench as bench  # noqa: E402, I001
import lightning as pl  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import CSVLogger  # noqa: E402
from nablafx.core import BlackBoxModel  # noqa: E402
from nablafx.evaluation.flexible_loss import FlexibleLoss  # noqa: E402

SEED = 42
PARENT = "butterfly_ssm_seed42_parent"
RESULTS = ROOT / "demo/butterfly"


class SaveForks(pl.Callback):
    """Checkpoint at the start of epoch E + 1 for each fork epoch E, then stop."""

    def __init__(self, epochs: list[int], directory: Path) -> None:
        self.epochs = epochs
        self.directory = directory

    def on_train_epoch_start(self, trainer, system) -> None:
        finished = trainer.current_epoch - 1
        if finished in self.epochs:
            trainer.save_checkpoint(self.directory / f"fork_epoch{finished}.ckpt")
            if finished == max(self.epochs):
                trainer.should_stop = True


class Fork(pl.Callback):
    """Start from a fork checkpoint; for k > 0, nudge every weight by one float32 step.

    Lightning's own resume (ckpt_path) restores the loop counters of a checkpoint saved
    at an epoch start as if the epoch were over, skips its validation and early
    stopping fails. So a child is a fresh run whose weights, AdamW moments,
    ReduceLROnPlateau state and early-stopping and guard states are loaded before the
    first step; its steps and epochs count from 0.

    Each weight moves to the next float32 value up or down, the direction drawn with a
    generator of seed 1000 + k. Moving a single weight by one step was absorbed by
    rounding in the mechanics test: after 100 steps only that weight differed.
    """

    def __init__(self, path: Path, k: int, early_stopping, guard) -> None:
        self.path = path
        self.k = k
        self.early_stopping = early_stopping
        self.guard = guard
        self.nudge: dict | None = None
        self.fork_step: int | None = None

    def on_train_start(self, trainer, system) -> None:
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        self.fork_step = state["global_step"]
        system.load_state_dict(state["state_dict"])
        trainer.optimizers[0].load_state_dict(state["optimizer_states"][0])
        trainer.lr_scheduler_configs[0].scheduler.load_state_dict(
            state["lr_schedulers"][0]
        )
        if self.early_stopping is not None:
            self.early_stopping.load_state_dict(
                state["callbacks"][self.early_stopping.state_key]
            )
        self.guard.load_state_dict(state["callbacks"][self.guard.state_key])
        if self.k == 0:
            return
        generator = torch.Generator().manual_seed(1000 + self.k)
        largest = 0.0
        for parameter in system.model.processor.parameters():
            up = torch.randint(0, 2, parameter.shape, generator=generator).bool()
            limit = torch.where(up, 1e30, -1e30).to(parameter)
            moved = torch.nextafter(parameter.data, limit)
            largest = max(largest, float((moved - parameter.data).abs().max()))
            parameter.data.copy_(moved)
        self.nudge = {
            "kind": "every weight one float32 step, direction drawn",
            "seed": 1000 + self.k,
            "largest_change": largest,
        }


class LearningRates(pl.Callback):
    """Record the learning rate of every epoch; with a schedule, impose it instead."""

    def __init__(self, replay: dict[str, float] | None = None) -> None:
        self.replay = replay
        self.by_epoch: dict[str, float] = {}

    def on_train_epoch_start(self, trainer, system) -> None:
        groups = trainer.optimizers[0].param_groups
        if self.replay is not None:
            for group in groups:
                group["lr"] = self.replay[str(trainer.current_epoch)]
        self.by_epoch[str(trainer.current_epoch)] = groups[0]["lr"]


def weights(run_id: str) -> dict[str, torch.Tensor]:
    return torch.load(
        bench.RUNS_DIR / f"nablafx_{run_id}/checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
    )["state_dict"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["parent", "child"])
    parser.add_argument("--fork", type=int, help="fork epoch (child)")
    parser.add_argument("--arm", choices=["decide", "replay"], default="decide")
    parser.add_argument("--k", type=int, default=0, help="0: no perturbation")
    parser.add_argument("--fork-epochs", type=int, nargs="+", default=[5, 100])
    parser.add_argument("--batch-size", type=int, default=16, help="16 except in tests")
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--tag", default="", help="suffix for test runs")
    args = parser.parse_args()
    run_id = (
        PARENT
        if args.role == "parent"
        else f"butterfly_ssm_seed42_f{args.fork}_{args.arm}_k{args.k}"
    ) + args.tag
    if (RESULTS / f"{run_id}.json").exists():
        print(f"{run_id}: done, skipped")
        return
    # No record but a directory: an interrupted attempt. Its checkpoints would make
    # ModelCheckpoint save to last-v1.ckpt and the test read the stale last.ckpt.
    if (bench.RUNS_DIR / f"nablafx_{run_id}").exists():
        shutil.rmtree(bench.RUNS_DIR / f"nablafx_{run_id}")
    commits = {
        "commit": bench.git_head(ROOT),
        "nablafx_commit": bench.git_head(ROOT / "third_party/nablafx"),
    }

    # The benchmark's settings with --deterministic.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.nn.functional.pad = bench._reflect_pad_by_slicing
    pl.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    wandb.init(mode="disabled")
    processor = bench.build_processor(
        SimpleNamespace(
            model="ssm-wavenet",
            num_blocks=8,
            channels=16,
            state_dim=4,
            output_act="tanh",
            discretization="zoh",
        )
    )
    system = bench.System(
        model=BlackBoxModel(processor),
        loss=FlexibleLoss(
            losses=[
                {"name": "l1_loss", "weight": 1.0, "alias": "l1"},
                {"name": "mrstft_loss", "weight": 0.1, "alias": "mrstft"},
            ]
        ),
        lr=0.01,
        log_media_every_n_steps=3000,
        use_callbacks=True,
    )
    system.honor_optim = True

    run_dir = bench.RUNS_DIR / f"nablafx_{run_id}"
    parent_dir = bench.RUNS_DIR / f"nablafx_{PARENT}{args.tag}"
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        save_last=True,
        save_top_k=1,
        monitor="loss/val/tot",
        every_n_train_steps=100,
        filename="{epoch}-{step}",
    )
    early_stopping = EarlyStopping(monitor="loss/val/tot", patience=50, verbose=True)
    guard = bench.PolarityGuard()
    fork = Fork(
        parent_dir / f"checkpoints/fork_epoch{args.fork}.ckpt",
        args.k,
        early_stopping if args.arm == "decide" else None,
        guard,
    )
    # A child's steps count from the fork, so the protocol's cap moves with it.
    max_steps = args.max_steps
    if args.role == "child":
        max_steps -= torch.load(fork.path, map_location="cpu", weights_only=False)[
            "global_step"
        ]
    replay = None
    if args.arm == "replay":
        control = json.loads(
            (
                RESULTS / f"butterfly_ssm_seed42_f{args.fork}_decide_k0{args.tag}.json"
            ).read_text(encoding="utf-8")
        )
        replay = control["lr_by_epoch"]
        max_steps = control["global_step"]
    learning_rates = LearningRates(replay)
    callbacks = [
        bench.metrics_callback(),
        checkpoint,
        ModelSummary(max_depth=2),
        LearningRateMonitor(),
        guard,
        learning_rates,
        *([early_stopping] if args.arm == "decide" else []),
        *(
            [SaveForks(args.fork_epochs, run_dir / "checkpoints")]
            if args.role == "parent"
            else [fork]
        ),
    ]
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        log_every_n_steps=100,
        sync_batchnorm=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        enable_progress_bar=False,
        deterministic=True,
        benchmark=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
        max_steps=max_steps,
        logger=CSVLogger(save_dir=run_dir, name="", version="logs"),
        callbacks=callbacks,
    )
    started = time.perf_counter()
    trainer.fit(system, datamodule=bench.data_module("trainval", args.batch_size))
    minutes = (time.perf_counter() - started) / 60.0

    record = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        **commits,
        "pilot": "diagnosis/butterfly/pilot_A.md",
        "role": args.role,
        "fork_epoch": args.fork,
        "arm": args.arm,
        "k": args.k,
        "fork_step": fork.fork_step,
        "nudge": fork.nudge,
        "seed": SEED,
        "batch_size": args.batch_size,
        "global_step": trainer.global_step,
        "stopped_epoch": early_stopping.stopped_epoch,
        "polarity_flips": guard.flips,
        "lr_by_epoch": learning_rates.by_epoch,
        "minutes_this_session": round(minutes, 2),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    if args.role == "child":
        tester = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            deterministic=True,
            benchmark=False,
            callbacks=[bench.metrics_callback()],
        )
        output = tester.test(
            system,
            datamodule=bench.data_module("test"),
            ckpt_path=str(run_dir / "checkpoints/last.ckpt"),
            verbose=False,
        )
        record["test_last"] = {key: float(value) for key, value in output[0].items()}
        if args.arm == "replay" and args.k == 0:
            control = weights(f"butterfly_ssm_seed42_f{args.fork}_decide_k0{args.tag}")
            mine = weights(run_id)
            record["identical_to_decide_k0"] = all(
                torch.equal(mine[key], control[key]) for key in control
            )
    print(json.dumps(record, indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{run_id}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if not args.tag:  # test runs stay out of the run ledger
        with bench.RUNS_LOG.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
