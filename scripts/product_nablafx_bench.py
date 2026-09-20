#!/usr/bin/env python3
"""Train and test one model on ToneTwist Big Muff inside the nablafx framework.

Everything except the processor is nablafx's own code and settings (commit
045db6e): the data module (44.1 -> 48 kHz resampling, 3 s segments, train and
val files pooled then split 90/10), AdamW, ReduceLROnPlateau per epoch, early
stopping (patience 50), 15k-step cap, the checkpoint callback, and the test loop
(5 s segments, metrics averaged over segments). Trainer settings mirror
cfg/trainer/trainer_bb.yaml and scripts/main.py, except gradient clipping: Table 5
of Comunità et al. (Frontiers 2025) gives value clipping at 1 for S4, the released
YAML sets 10, and the default here follows the paper. The released test script
loads last.ckpt, so that is the primary result; best.ckpt is recorded too.

One addition to training, for every model unless --no-polarity-guard: PolarityGuard
flips the output sign whenever the validation output anticorrelates with the target
(see its docstring).

Extra packages, installed with `uv pip install` and not in uv.lock:
lightning==2.6.1, jsonargparse[signatures]==4.52.0, natsort==8.4.0, wandb==0.30.0,
torchvision==0.28.0 (https://download.pytorch.org/whl/cu130).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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

# Worker sockets must not live in a directory named tmp. On 2026-09-20 the shared
# /fastdata/lavaulta/tmp was deleted while a run was training; its DataLoader workers died
# with it and the run hung for 28 minutes holding 19.9 GiB of GPU memory at 0 % use. These
# two lines put the sockets under the project and make PyTorch's own timeout fail loudly
# instead of waiting for ever; neither touches the training itself.
_IPC = Path(__file__).resolve().parents[1] / "demo/runs/.ipc"
_IPC.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_IPC)
tempfile.tempdir = None

_DataLoader = torch.utils.data.DataLoader


def _dataloader_with_timeout(*args, **kwargs):
    """nablafx builds its loaders without a timeout, so a dead worker hangs the run."""
    if kwargs.get("num_workers", 0) and "timeout" not in kwargs:
        kwargs["timeout"] = 600
    return _DataLoader(*args, **kwargs)


torch.utils.data.DataLoader = _dataloader_with_timeout


_pad = torch.nn.functional.pad


def _reflect_pad_by_slicing(input, pad, mode="constant", value=None):
    """Reflection padding built from slices, whose CUDA backward is deterministic.

    torch.stft pads its input this way inside the MR-STFT loss, and
    reflection_pad1d_backward_out_cuda has no deterministic implementation. The
    padded values are identical; installed only with --deterministic.
    """
    if mode != "reflect" or len(pad) != 2:
        return _pad(input, pad, mode, value)
    left, right = pad
    length = input.shape[-1]
    return torch.cat(
        [
            input[..., 1 : left + 1].flip(-1),
            input,
            input[..., length - right - 1 : length - 1].flip(-1),
        ],
        dim=-1,
    )


class System(BlackBoxSystem):
    """BaseSystem.configure_optimizers minus verbose=True, which torch 2.13 removed.

    With honor_optim, parameters that declare an _optim dict (nablafx's DSSM and
    our SSM layers mark log_dt, log_A_real and A_imag with weight_decay 0) get
    their own AdamW group with those settings; the released code ignores _optim.
    """

    honor_optim = False
    # PyTorch's own default; --plateau-threshold raises it above the noise floor of the
    # validation curve, which diagnosis/butterfly/plateau_margin.md measured at 4 to 9e-3.
    plateau_threshold = 1e-4

    def configure_optimizers(self):
        groups: dict[tuple, list] = {}
        for parameter in self.model.parameters():
            hints = getattr(parameter, "_optim", {}) if self.honor_optim else {}
            groups.setdefault(tuple(sorted(hints.items())), []).append(parameter)
        optimizer = torch.optim.AdamW(
            [{"params": params, **dict(hints)} for hints, params in groups.items()],
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            threshold=self.plateau_threshold,
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


class PolarityGuard(pl.Callback):
    """Negate the output when the validation output anticorrelates with the target.

    The MR-STFT term cannot see the output's sign and, at the published 1:0.1
    weighting, outweighs L1, so gradient descent can settle on a well-fitted but
    inverted output (ESR near 4); an SSM-WaveNet run did (RUNS.jsonl,
    ssmzoh_seed42). Both processors end in an odd tanh, so negating the weight and
    bias of the layer before it negates the output exactly; Adam's first moments
    are negated with them. Checked after every validation epoch except the sanity
    check; the epochs of the flips are recorded.
    """

    def __init__(self) -> None:
        self.flips: list[int] = []
        self.inner = 0.0
        self.pred: torch.Tensor | None = None
        self.hook = None

    def state_dict(self) -> dict:
        return {"flips": self.flips}

    def load_state_dict(self, state_dict: dict) -> None:
        self.flips = state_dict["flips"]

    def on_validation_epoch_start(self, trainer, system) -> None:
        self.inner = 0.0
        self.hook = system.model.processor.register_forward_hook(self._store)

    def _store(self, module, inputs, output) -> None:
        self.pred = output

    def on_validation_batch_end(
        self, trainer, system, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.inner += float((self.pred * batch[1]).sum())

    def on_validation_epoch_end(self, trainer, system) -> None:
        self.hook.remove()
        if trainer.sanity_checking or self.inner >= 0:
            return
        processor = system.model.processor
        layer = (
            processor.output_net[3]
            if isinstance(processor, SSMWaveNet)
            else processor.contract
        )
        state = trainer.optimizers[0].state
        with torch.no_grad():
            for parameter in (layer.weight, layer.bias):
                parameter.neg_()
                if "exp_avg" in state.get(parameter, {}):
                    state[parameter]["exp_avg"].neg_()
        self.flips.append(trainer.current_epoch)
        print(f"polarity guard: output negated after epoch {trainer.current_epoch}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["s4-tf-l-16", "s4-l-16", "ssm-wavenet"]
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=1e-4,
        help="relative improvement ReduceLROnPlateau requires; PyTorch's default is 1e-4",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="reseed just before fit, so the train/val split and the batch order are"
        " common to every run whatever --seed is",
    )
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
    parser.add_argument("--clip-value", type=float, default=1.0)
    parser.add_argument("--discretization", default="free", choices=["free", "zoh"])
    parser.add_argument(
        "--honor-optim",
        action="store_true",
        help="apply the _optim hints (no weight decay on state-space parameters)",
    )
    parser.add_argument(
        "--no-polarity-guard",
        action="store_true",
        help="train exactly as the released code does, without PolarityGuard",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="deterministic CUDA algorithms, so that a run can be repeated bit for bit",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="training batch size; smaller only for smoke tests next to another run",
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
            discretization=args.discretization,
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


def data_module(split: str, batch_size: int = 16) -> DryWetFilesPluginDataModule:
    if split == "trainval":
        return DryWetFilesPluginDataModule(
            root_dir_dry=str(DATA / "DRY/trainval"),
            root_dir_wet=str(WET / "trainval" / SETTING),
            data_to_use=1.0,
            trainval_split=0.9,
            sample_length=144_000,
            sample_rate=48_000,
            preload=True,
            batch_size=batch_size,
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
    # Taken now, not when the run ends hours later, so commits made meanwhile
    # are not attributed to this run.
    commits = {
        "commit": git_head(ROOT),
        "nablafx_commit": git_head(ROOT / "third_party/nablafx"),
    }
    # scripts/main.py settings, applied before the model is built. cuBLAS reads
    # its workspace setting when CUDA starts, so it is set before any CUDA call.
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.nn.functional.pad = _reflect_pad_by_slicing
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(
        args.deterministic, warn_only=not args.deterministic
    )
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
    system.honor_optim = args.honor_optim
    system.plateau_threshold = args.plateau_threshold

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
    polarity_guard = PolarityGuard()
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
        deterministic=True if args.deterministic else None,
        benchmark=not args.deterministic,
        gradient_clip_val=args.clip_value,
        gradient_clip_algorithm="value",
        max_steps=args.max_steps,
        logger=CSVLogger(save_dir=run_dir, name="", version="logs"),
        callbacks=[
            metrics_callback(),
            checkpoint,
            ModelSummary(max_depth=2),
            LearningRateMonitor(),
            early_stopping,
            *([] if args.no_polarity_guard else [polarity_guard]),
        ],
    )
    print(
        f"{args.model} {args.run_id}: {parameters:,} params, lr {lr}, "
        f"weights {l1_weight}/{mrstft_weight}",
        flush=True,
    )

    started = time.perf_counter()
    # The split and the batch order are drawn from the global generator inside fit, so
    # reseeding here makes them common to every run whatever --seed is.
    if args.split_seed is not None:
        pl.seed_everything(args.split_seed, workers=True)
    resume_from = str(last) if args.resume and last.exists() else None
    trainer.fit(
        system,
        datamodule=data_module("trainval", args.batch_size),
        ckpt_path=resume_from,
    )
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
            benchmark=not args.deterministic,
            callbacks=[metrics_callback()],
        )
        output = tester.test(
            system, datamodule=data_module("test"), ckpt_path=str(path), verbose=False
        )
        tests[label] = {key: float(value) for key, value in output[0].items()}

    record = {
        "run_id": args.run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        **commits,
        "benchmark": f"ToneTwist Big Muff {SETTING}, nablafx protocol",
        "model": args.model,
        "parameters": parameters,
        "seed": args.seed,
        "lr": lr,
        "loss_weights": {"l1": l1_weight, "mrstft": mrstft_weight},
        "gradient_clip_val": args.clip_value,
        "honor_optim": args.honor_optim,
        "plateau_threshold": args.plateau_threshold,
        "split_seed": args.split_seed,
        "polarity_guard": not args.no_polarity_guard,
        "polarity_flips": polarity_guard.flips,
        "deterministic": args.deterministic,
        "batch_size": args.batch_size,
        "ssm": (
            {
                "num_blocks": args.num_blocks,
                "channels": args.channels,
                "state_dim": args.state_dim,
                "output_act": args.output_act,
                "discretization": args.discretization,
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
