#!/usr/bin/env python3
"""Train and test one model on ToneTwist Big Muff inside the nablafx framework.

Everything except the processor is nablafx's own code and settings (commit
045db6e): the data module (44.1 -> 48 kHz resampling, 3 s segments, train and
val files pooled then split 90/10), AdamW, ReduceLROnPlateau per epoch, early
stopping (patience 50), value clipping at 10, 15k-step cap, the checkpoint
callback, and the test loop (5 s segments, metrics averaged over segments).
Trainer settings mirror cfg/trainer/trainer_bb.yaml and scripts/main.py. The
published numbers use last.ckpt, so that is the primary result; best.ckpt is
recorded too.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for sub in ("third_party/auraloss", "third_party/nablafx", "third_party_shims"):
    sys.path.insert(0, str(ROOT / sub))
os.environ.setdefault("WANDB_MODE", "disabled")

import lightning as pl  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
import wandb  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import CSVLogger  # noqa: E402
from nablafx.callbacks import MetricsLoggingCallback  # noqa: E402
from nablafx.core import BlackBoxModel, BlackBoxSystem  # noqa: E402
from nablafx.data.datamodules import DryWetFilesPluginDataModule  # noqa: E402
from nablafx.evaluation.flexible_loss import FlexibleLoss  # noqa: E402
from nablafx.processors import S4  # noqa: E402

from fssr_nam.models.ssm_wavenet import SSMWaveNet  # noqa: E402

DATA = ROOT / "datasets/raw/external/tone_twist_bigmuff/extracted"
WET = DATA / "ElectroHarmonix-BigMuff"
SETTING = "S050_V100"
RUNS_DIR = ROOT / "demo/runs"
RESULTS_DIR = ROOT / "demo/nablafx_bench"
RUNS_LOG = ROOT / "demo/RUNS.jsonl"

# Published Big Muff settings, appendix Tables 20 and 24: lr, (L1, MR-STFT) weights.
PUBLISHED = {
    "s4-tf-l-16": (0.01, (1.0, 0.1)),
    "s4-l-16": (0.01, (10.0, 1.0)),
}
METRICS = [
    {"name": name, "alias": alias}
    for name, alias in (
        ("l1_loss", "l1"),
        ("mse_loss", "mse"),
        ("esr_loss", "esr"),
        ("mape_loss", "mape"),
        ("mrstft_loss", "mrstft"),
    )
]


def _info(path: str) -> SimpleNamespace:
    return SimpleNamespace(num_frames=sf.info(path).frames)


def _load(
    path: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
) -> tuple[torch.Tensor, int]:
    audio, rate = sf.read(
        path, start=frame_offset, frames=num_frames, dtype="float32", always_2d=True
    )
    return torch.from_numpy(audio.T.copy()), rate


# torchaudio 2.11 dropped info() and moved load() to torchcodec. nablafx's
# PluginDataset calls both; soundfile's float32 PCM16 decoding is the same
# int/32768 scaling as torchaudio.load(normalize=True). Resampling stays
# torchaudio.functional.resample, as in nablafx.
torchaudio.info = _info
torchaudio.load = _load


class System(BlackBoxSystem):
    """BaseSystem.configure_optimizers minus verbose=True, which torch 2.13 removed."""

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20
        )
        return [optimizer], [
            {
                "scheduler": scheduler,
                "monitor": "loss/val/tot",
                "interval": "epoch",
                "frequency": 1,
            }
        ]


class SSMWaveNetProcessor(SSMWaveNet):
    """BlackBoxModel reads num_controls; the model has no controls."""

    num_controls = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["s4-tf-l-16", "s4-l-16", "ssm-wavenet"]
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--l1-weight", type=float)
    parser.add_argument("--mrstft-weight", type=float)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument(
        "--output-act", default="tanh", choices=["tanh", "softsign", "none"]
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-record", action="store_true", help="smoke test: do not write results"
    )
    return parser.parse_args()


def build_processor(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "ssm-wavenet":
        return SSMWaveNetProcessor(
            num_blocks=args.num_blocks,
            channels=args.channels,
            state_dim=args.state_dim,
            output_act=args.output_act,
        )
    return S4(
        num_inputs=1,
        num_outputs=1,
        num_controls=0,
        num_blocks=8,
        s4_state_dim=32,
        channel_width=16,
        batchnorm=False,
        residual=True,
        direct_path=False,
        cond_type="tfilm" if args.model == "s4-tf-l-16" else None,
        cond_block_size=128,
        cond_num_layers=1,
        act_type="tanh",
        s4_learning_rate=0.01,
    )


def data_module(split: str) -> DryWetFilesPluginDataModule:
    if split == "trainval":
        return DryWetFilesPluginDataModule(
            root_dir_dry=str(DATA / "DRY/trainval"),
            root_dir_wet=str(WET / "trainval" / SETTING),
            data_to_use=1.0,
            trainval_split=0.9,
            sample_length=144_000,
            sample_rate=48_000,
            preload=True,
            batch_size=16,
            num_workers=4,
        )
    return DryWetFilesPluginDataModule(
        root_dir_dry=str(DATA / "DRY/test"),
        root_dir_wet=str(WET / "test" / SETTING),
        data_to_use=1.0,
        sample_length=240_000,
        sample_rate=48_000,
        preload=True,
        batch_size=8,
        num_workers=4,
    )


def metrics_callback() -> MetricsLoggingCallback:
    return MetricsLoggingCallback(
        metrics=METRICS,
        log_on_step=False,
        log_on_epoch=True,
        sync_dist=True,
        prefix="metric",
    )


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    # scripts/main.py settings, applied before the model is built.
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_num_threads(1)
    # BaseSystem.on_train_start calls wandb.watch unconditionally.
    wandb.init(mode="disabled")

    lr, (l1_weight, mrstft_weight) = PUBLISHED.get(args.model, (0.01, (1.0, 0.1)))
    lr = args.lr if args.lr is not None else lr
    l1_weight = args.l1_weight if args.l1_weight is not None else l1_weight
    mrstft_weight = (
        args.mrstft_weight if args.mrstft_weight is not None else mrstft_weight
    )

    processor = build_processor(args)
    parameters = sum(p.numel() for p in processor.parameters())
    loss = FlexibleLoss(
        losses=[
            {"name": "l1_loss", "weight": l1_weight, "alias": "l1"},
            {"name": "mrstft_loss", "weight": mrstft_weight, "alias": "mrstft"},
        ]
    )
    system = System(
        model=BlackBoxModel(processor),
        loss=loss,
        lr=lr,
        log_media_every_n_steps=3000,
        use_callbacks=True,
    )

    run_dir = RUNS_DIR / f"nablafx_{args.run_id}"
    last = run_dir / "checkpoints/last.ckpt"
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        save_last=True,
        save_top_k=1,
        monitor="loss/val/tot",
        every_n_train_steps=100,
        filename="{epoch}-{step}",
    )
    early_stopping = EarlyStopping(monitor="loss/val/tot", patience=50, verbose=True)
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
        deterministic=None,
        benchmark=True,
        gradient_clip_val=10.0,
        gradient_clip_algorithm="value",
        max_steps=args.max_steps,
        logger=CSVLogger(save_dir=run_dir, name="", version="logs"),
        callbacks=[
            metrics_callback(),
            checkpoint,
            ModelSummary(max_depth=2),
            LearningRateMonitor(),
            early_stopping,
        ],
    )
    print(
        f"{args.model} {args.run_id}: {parameters:,} params, lr {lr}, "
        f"weights {l1_weight}/{mrstft_weight}",
        flush=True,
    )

    started = time.perf_counter()
    resume_from = str(last) if args.resume and last.exists() else None
    trainer.fit(system, datamodule=data_module("trainval"), ckpt_path=resume_from)
    minutes = (time.perf_counter() - started) / 60.0

    tests = {"best": {}}
    checkpoints = [("last", last)]
    if checkpoint.best_model_path:  # empty only in smoke tests shorter than 100 steps
        checkpoints.append(("best", Path(checkpoint.best_model_path)))
    for label, path in checkpoints:
        tester = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            benchmark=True,
            callbacks=[metrics_callback()],
        )
        output = tester.test(
            system, datamodule=data_module("test"), ckpt_path=str(path), verbose=False
        )
        tests[label] = {key: float(value) for key, value in output[0].items()}

    record = {
        "run_id": args.run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": git_head(ROOT),
        "nablafx_commit": git_head(ROOT / "third_party/nablafx"),
        "benchmark": f"ToneTwist Big Muff {SETTING}, nablafx protocol",
        "model": args.model,
        "parameters": parameters,
        "seed": args.seed,
        "lr": lr,
        "loss_weights": {"l1": l1_weight, "mrstft": mrstft_weight},
        "ssm": (
            {
                "num_blocks": args.num_blocks,
                "channels": args.channels,
                "state_dim": args.state_dim,
                "output_act": args.output_act,
            }
            if args.model == "ssm-wavenet"
            else None
        ),
        "resumed": resume_from is not None,
        "global_step": trainer.global_step,
        "stopped_epoch": early_stopping.stopped_epoch,
        "best_val_loss": float(checkpoint.best_model_score or "nan"),
        "best_checkpoint": Path(checkpoint.best_model_path).name,
        "minutes_this_session": round(minutes, 2),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "test_last": tests["last"],
        "test_best": tests["best"],
    }
    print(json.dumps(record, indent=2))
    if args.no_record:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.run_id}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    with RUNS_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
