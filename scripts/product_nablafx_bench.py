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
import math
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
from nablafx.evaluation.registry import EvaluationRegistry  # noqa: E402
from nablafx.processors import S4  # noqa: E402

from fssr_nam.models.deep_greybox import DeepGreyBoxAmp  # noqa: E402
from fssr_nam.models.greybox import GreyBoxAmp  # noqa: E402
from fssr_nam.models.ssm_wavenet import DiagonalSSMLayer, SSMWaveNet  # noqa: E402
from fssr_nam.product.data import advance  # noqa: E402

DATA = ROOT / "datasets/raw/external/tone_twist_bigmuff/extracted"
WET = DATA / "ElectroHarmonix-BigMuff"
SETTING = "S050_V100"
PARAM = ROOT / "datasets/raw/external/tone_twist_param/extracted"
MARSHALL = PARAM / "Marshall-JVM410H-ChOD1-trimmed"
FFUZZ = PARAM / "MultidrivePedalPro-F-Fuzz"
# Exploratory VS100R capture of 2026-09-26 (scripts/product_valvestate_dataset.py).
VS100R = ROOT / "datasets/raw/internal/valvestate_vs100r/explore_48k"
# The same takes, not decimated (demo/capture/valvestate/predictions_192k.md).
VS100R_192K = ROOT / "datasets/raw/internal/valvestate_vs100r/explore_192k"
# auraloss's MultiResolutionSTFTLoss defaults, in samples at 48 kHz; a device at another
# rate scales them so the loss keeps the same resolution in seconds and Hz.
MRSTFT_48K = {
    "fft_sizes": [1024, 2048, 512],
    "hop_sizes": [120, 240, 50],
    "win_lengths": [600, 1200, 240],
}
# Parametric devices, nablafx's data configs (cfg/data/data-param_*): dry and wet roots
# per split, knob indices, fraction of trainval used. F-Fuzz has the same eleven
# settings in trainval and test, so two are held out of training (demo/param).
DEVICES = {
    "marshall": {
        "dry": MARSHALL / "DRY",
        "wet": MARSHALL / "SpeakerOut",
        "params": [0, 1, 2, 3],  # bass, mid, treble, gain
        "data_to_use": 1.0,
        "held_out": [],
    },
    "ffuzz": {
        # The targets were rendered from the dry files with calibration markers; the
        # plain DRY record is 2-3 s out of step with them (demo/param/DEVIATION.md).
        "dry": PARAM / "DIGITAL-PARAMETRIC/DRY-with-markers",
        "wet": FFUZZ,
        "params": [0],  # fuzz
        "data_to_use": 0.1,
        "held_out": ["F030_V100", "F070_V100"],
    },
    "vs100r": {
        # Input = each take's loopback, target = FX loop send, OD2, Power Dimension off.
        # The send is taken before the ECC83 stage and the tone stack (VS100 schematic
        # V100-60-02): bass, middle and treble do not reach it (repeat.json), so only
        # gain and contour are given; takes differing in those three are pooled.
        "dry": VS100R / "dry",
        "wet": VS100R / "wet",
        "params": [0, 1],  # gain, contour (names also carry bass, middle, treble)
        "data_to_use": 1.0,
        "held_out": [],
        "benchmark": "Marshall VS100R exploratory capture 2026-09-26, OD2, PD off",
    },
    "vs100r_192k": {
        "dry": VS100R_192K / "dry",
        "wet": VS100R_192K / "wet",
        "params": [0, 1],
        "data_to_use": 1.0,
        "held_out": [],
        "rate": 192_000,
        "benchmark": (
            "Marshall VS100R exploratory capture 2026-09-26, OD2, PD off, 192 kHz"
        ),
    },
}


def device_rate(device: str) -> int:
    return DEVICES.get(device, {}).get("rate", 48_000)


def at_rate(processor: torch.nn.Module, rate: int) -> dict:
    """For a device not at 48 kHz: the same continuous-time initialisation (every SSM
    step dt scaled by 48000/rate, in place, no random draw) and MR-STFT sizes scaled so
    the loss keeps its resolution in seconds and Hz. Returns the loss entry's params."""
    if rate == 48_000:
        return {}
    with torch.no_grad():
        for module in processor.modules():
            if hasattr(module, "log_dt"):
                module.log_dt.add_(math.log(48_000 / rate))
    return {
        "params": {k: [v * rate // 48_000 for v in vs] for k, vs in MRSTFT_48K.items()}
    }


RUNS_DIR = ROOT / "demo/runs"
RESULTS_DIR = ROOT / "demo/nablafx_bench"
RUNS_LOG = ROOT / "demo/RUNS.jsonl"

# Published Big Muff settings, appendix Tables 20 and 24: lr, (L1, MR-STFT) weights.
PUBLISHED = {
    "s4-tf-l-16": (0.01, (1.0, 0.1)),
    "s4-l-16": (0.01, (10.0, 1.0)),
    # nablafx's parametric configs (cfg/model/s4-param) weigh both losses equally.
    "s4-param": (0.01, (0.5, 0.5)),
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


# --target-advance (F-Fuzz only): targets are read this many samples earlier.
TARGET_ADVANCE = 0


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
    if TARGET_ADVANCE and ".target." in Path(path).name:
        if frame_offset or num_frames != -1:
            raise ValueError("--target-advance needs whole-file loads (preload=True)")
        audio = advance(audio, TARGET_ADVANCE)
    return torch.from_numpy(audio.T.copy()), rate


# torchaudio 2.11 dropped info() and moved load() to torchcodec. nablafx's
# PluginDataset calls both; soundfile's float32 PCM16 decoding is the same
# int/32768 scaling as torchaudio.load(normalize=True). Resampling stays
# torchaudio.functional.resample, as in nablafx.
torchaudio.info = _info
torchaudio.load = _load

# Worker sockets must not live in a directory named tmp. On 2026-09-20 the shared
# /fastdata/lavaulta/tmp was deleted while a run was training; its DataLoader
# workers died with it and the run hung for 28 minutes holding 19.9 GiB of GPU
# memory at 0 % use. These two lines put the sockets under the project and make
# PyTorch's own timeout fail loudly instead of waiting for ever; neither touches
# the training itself.
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
    # PyTorch's own default; --plateau-threshold raises it above the noise floor of
    # the validation curve, which diagnosis/butterfly/plateau_margin.md measured at
    # 4 to 9e-3.
    plateau_threshold = 1e-4
    # Pilot C: when a fixed schedule is given, the plateau detector is removed entirely
    # rather than tuned, since diagnosis/seeds/plateau_simulation.md showed no threshold
    # makes it fire reproducibly.
    lr_halvings: list[int] | None = None
    lr_halving_steps: list[int] | None = None

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
        if self.lr_halvings is not None or self.lr_halving_steps is not None:
            return optimizer
        return [optimizer], [
            {
                "scheduler": scheduler,
                "monitor": "loss/val/tot",
                "interval": "epoch",
                "frequency": 1,
            }
        ]


class RecordingSampler(torch.utils.data.RandomSampler):
    """RandomSampler that keeps the order it drew at chosen epochs (0-based).

    It is the sampler shuffle=True would build, drawing its seed from the global
    generator at the start of every epoch exactly as the original does, so training is
    unchanged; it only remembers the order (diagnosis/seeds/paired_protocol.md: is the
    batch order common to two architectures beyond the first epoch?)."""

    def __init__(self, dataset, epochs: list[int]) -> None:
        super().__init__(dataset)
        self.epochs = set(epochs)
        self.epoch = -1
        self.log: dict[int, list[int]] = {}

    def __iter__(self):
        self.epoch += 1
        order = list(super().__iter__())
        if self.epoch in self.epochs:
            self.log[self.epoch] = order
        yield from order


def record_sampler(datamodule, epochs: list[int]) -> RecordingSampler:
    """Rebuild nablafx's train loader (datamodules.py) with a RecordingSampler."""
    holder = {}

    def train_dataloader():
        holder["sampler"] = RecordingSampler(datamodule.train_dataset, epochs)
        return torch.utils.data.DataLoader(
            datamodule.train_dataset,
            batch_size=datamodule.batch_size,
            sampler=holder["sampler"],
            num_workers=datamodule.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    datamodule.train_dataloader = train_dataloader
    return holder


def keep_segments_per_take(datamodule, count: int | None) -> None:
    """After nablafx's split (drawn inside fit), keep in the training subset only the
    segments at `count` evenly spaced positions of their file, the same positions in
    every file. No random draw: the split, hence validation, is the full run's. None
    leaves the datamodule untouched."""
    if count is None:
        return
    setup = datamodule.setup

    def wrapped(stage=None):
        setup(stage)
        if getattr(datamodule, "train_dataset", None) is None:
            return
        samples = datamodule.trainval_dataset.samples
        length = datamodule.sample_length

        def kept(index: int) -> bool:
            sample = samples[index]
            total = sample["frames"] // length
            positions = {k * total // count for k in range(count)}
            return sample["offset"] // length in positions

        subset = datamodule.train_dataset
        subset.indices = [i for i in subset.indices if kept(i)]

    datamodule.setup = wrapped


class FixedSchedule(pl.Callback):
    """Halve the learning rate at fixed epochs, in place of the plateau detector."""

    def __init__(self, epochs: list[int]) -> None:
        self.epochs = set(epochs)

    def on_train_epoch_start(self, trainer, system) -> None:
        if trainer.current_epoch in self.epochs:
            for group in trainer.optimizers[0].param_groups:
                group["lr"] /= 2


class FixedStepSchedule(pl.Callback):
    """Halve the learning rate before the optimizer steps of these global step numbers
    (the steps where a twin run's plateau detector halved it), plateau detector off."""

    def __init__(self, steps: list[int]) -> None:
        self.steps = set(steps)

    def on_train_batch_start(self, trainer, system, batch, batch_idx) -> None:
        if trainer.global_step in self.steps:
            for group in trainer.optimizers[0].param_groups:
                group["lr"] /= 2


class SSMWaveNetProcessor(SSMWaveNet):
    """BlackBoxModel reads num_controls, which SSMWaveNet now sets itself."""


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
        if isinstance(processor, SSMWaveNet):
            layer = processor.output_net[3]
            parameters = (layer.weight, layer.bias)
        elif isinstance(processor, DeepGreyBoxAmp):
            parameters = (processor.post_fir,)
        else:
            parameters = (processor.contract.weight, processor.contract.bias)
        state = trainer.optimizers[0].state
        with torch.no_grad():
            # act_type 'odd' and the grey boxes have no bias there.
            for parameter in (p for p in parameters if p is not None):
                parameter.neg_()
                if "exp_avg" in state.get(parameter, {}):
                    state[parameter]["exp_avg"].neg_()
        self.flips.append(trainer.current_epoch)
        print(f"polarity guard: output negated after epoch {trainer.current_epoch}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "s4-tf-l-16",
            "s4-l-16",
            "ssm-wavenet",
            "s4-param",
            "greybox",
            "deepgreybox",
        ],
    )
    parser.add_argument(
        "--gb-stages", type=int, default=1, help="deepgreybox: stages in series"
    )
    parser.add_argument(
        "--gb-taps", type=int, default=32, help="deepgreybox: input/output FIR taps"
    )
    parser.add_argument(
        "--gb-bias", action="store_true", help="deepgreybox: bias before the clipper"
    )
    parser.add_argument(
        "--gb-slow",
        type=float,
        nargs="*",
        default=[],
        help="deepgreybox: time constants (s) of the slow envelopes",
    )
    parser.add_argument(
        "--device",
        default="bigmuff",
        choices=["bigmuff", *DEVICES],
        help="bigmuff: the original benchmark; the others are parametric (knob inputs)",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=1e-4,
        help="relative improvement ReduceLROnPlateau requires;"
        " PyTorch's default is 1e-4",
    )
    parser.add_argument(
        "--lr-halvings",
        type=int,
        nargs="+",
        default=None,
        help="halve the learning rate at the start of these epochs and drop the plateau"
        " detector altogether (pilot C)",
    )
    parser.add_argument(
        "--fft-size",
        choices=["pow2", "fast"],
        default="pow2",
        help="SSM convolution FFT length: next power of 2 (the trained runs' bits) or"
        " next 2-3-5-smooth length (288000 instead of 524288 for 144000 samples)",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="full float32 matmuls and cuDNN convolutions (TF32 off); with TF32 the"
        " SSM gradients are ~20 %% off float64 (demo/fft_size/NOTE.md)",
    )
    parser.add_argument(
        "--lr-halving-steps",
        type=int,
        nargs="+",
        default=None,
        help="halve the learning rate before these global steps and drop the plateau"
        " detector (replay of a twin run's schedule, predictions_192k.md)",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="run to --max-steps whatever the validation loss does (pilot C)",
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
    parser.add_argument(
        "--discretization", default="free", choices=["free", "zoh", "foh"]
    )
    parser.add_argument(
        "--cond",
        default="film",
        choices=["film", "poles"],
        help="parametric devices: knobs act on the gated 1x1 layers (FiLM) or move the"
        " SSM poles (demo/param/predictions_poles.md)",
    )
    parser.add_argument("--cond-head", default="joint", choices=["joint", "additive"])
    parser.add_argument(
        "--head-act",
        default="relu",
        choices=["relu", "leaky", "tanh"],
        help="SSM-WaveNet output head: leaky keeps a gradient once every skip sum < 0;"
        " tanh is odd (needed by --act-type odd)",
    )
    parser.add_argument(
        "--act-type",
        default="gated",
        choices=["gated", "odd"],
        help="SSM-WaveNet gate: gated tanh * sigmoid, or odd (OddGate, no bias,"
        " explicit asymmetry; demo/odd_gate/predictions.md)",
    )
    parser.add_argument(
        "--cond-hidden",
        type=int,
        nargs="+",
        default=[32],
        help="hidden width of a joint head, or one width per knob of an additive one",
    )
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
    parser.add_argument(
        "--target-advance",
        type=int,
        default=0,
        help="F-Fuzz only: read every target this many samples earlier, compensating"
        " the latency of its renders (DIAGNOSIS_ffuzz.md, FIX_ffuzz.md)",
    )
    parser.add_argument(
        "--train-segments-per-take",
        type=int,
        default=None,
        help="train on this many evenly spaced segments of each file, at the same"
        " positions in every file, after the split (validation unchanged;"
        " demo/train_length/predictions.md)",
    )
    parser.add_argument(
        "--log-sampler-epochs",
        type=int,
        nargs="+",
        default=None,
        help="record the training order at these epochs (0-based), in the record",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-record", action="store_true", help="smoke test: do not write results"
    )
    return parser.parse_args()


def build_processor(args: argparse.Namespace) -> torch.nn.Module:
    device = getattr(args, "device", "bigmuff")
    controls = len(DEVICES[device]["params"]) if device in DEVICES else 0
    if args.model == "ssm-wavenet":
        return SSMWaveNetProcessor(
            num_blocks=args.num_blocks,
            channels=args.channels,
            state_dim=args.state_dim,
            output_act=args.output_act,
            discretization=args.discretization,
            num_controls=controls,
            # getattr: records written before these options load without them.
            cond=getattr(args, "cond", "film"),
            cond_head=getattr(args, "cond_head", "joint"),
            cond_hidden=tuple(getattr(args, "cond_hidden", (32,))),
            head_act=getattr(args, "head_act", "relu"),
            act_type=getattr(args, "act_type", "gated"),
        )
    if args.model == "greybox":
        return GreyBoxAmp(num_controls=controls)
    if args.model == "deepgreybox":
        return DeepGreyBoxAmp(
            num_controls=controls,
            stages=args.gb_stages,
            taps=args.gb_taps,
            bias=args.gb_bias,
            slow_s=tuple(args.gb_slow),
            # Nominal device rate, as for the SSM steps (at_rate): 48 kHz unless set.
            sample_rate=device_rate(device),
        )
    if args.model == "s4-param":
        # cfg/model/s4-param/model_bb-param_s4-f-b8-s32-c16.yaml: static FiLM (MLP).
        return S4(
            num_inputs=1,
            num_outputs=1,
            num_controls=controls,
            num_blocks=8,
            s4_state_dim=32,
            channel_width=16,
            batchnorm=False,
            residual=True,
            direct_path=False,
            cond_type="film",
            cond_block_size=128,
            cond_num_layers=1,
            act_type="tanh",
            s4_learning_rate=0.01,
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


def held_out_view(device: str) -> Path:
    """Trainval wet directory without the held-out settings, as symlinks."""
    spec = DEVICES[device]
    source = spec["wet"] / "trainval"
    if not spec["held_out"]:
        return source
    view = RUNS_DIR / f".views/{device}_trainval"
    view.mkdir(parents=True, exist_ok=True)
    for setting in sorted(source.iterdir()):
        link = view / setting.name
        if setting.name not in spec["held_out"] and not link.exists():
            link.symlink_to(setting)
    return view


def data_module(
    split: str, batch_size: int = 16, device: str = "bigmuff"
) -> DryWetFilesPluginDataModule:
    if device in DEVICES:
        spec = DEVICES[device]
        trainval = split == "trainval"
        return DryWetFilesPluginDataModule(
            root_dir_dry=str(spec["dry"] / split),
            root_dir_wet=str(
                held_out_view(device) if trainval else spec["wet"] / split
            ),
            params_idxs_to_use=spec["params"],
            data_to_use=spec["data_to_use"] if trainval else 1.0,
            trainval_split=0.9,
            sample_length=144_000 if trainval else 240_000,
            sample_rate=device_rate(device),
            preload=True,
            batch_size=batch_size if trainval else 8,
            num_workers=4,
        )
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


@torch.no_grad()
def metrics_by_setting(system: System, checkpoint: Path, dataset) -> dict:
    """ESR and MR-STFT of one checkpoint on each setting of `dataset` (a nablafx
    parametric dataset, or a Subset of one), averaged over batches of eight
    weighted by size as Lightning's test loop does."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    system.load_state_dict(state["state_dict"])
    model = system.model.to("cuda").eval()
    base = getattr(dataset, "dataset", dataset)
    indices = getattr(dataset, "indices", range(len(dataset)))
    groups: dict[str, list[int]] = {}
    for index in indices:
        setting = Path(base.samples[index]["target_file"]).parent.name
        groups.setdefault(setting, []).append(index)
    esr = EvaluationRegistry.get_function("esr_loss")
    mrstft = EvaluationRegistry.get_function("mrstft_loss")
    results = {}
    for setting, members in sorted(groups.items()):
        totals = {"esr": 0.0, "mrstft": 0.0}
        for start in range(0, len(members), 8):
            items = [base[i] for i in members[start : start + 8]]
            x, y, controls = (
                torch.stack(parts).to("cuda") for parts in zip(*items, strict=True)
            )
            model.reset_states()
            pred = model(x, controls)
            totals["esr"] += float(esr(pred, y)) * len(items)
            totals["mrstft"] += float(mrstft(pred, y)) * len(items)
        results[setting] = {
            key: value / len(members) for key, value in totals.items()
        } | {"segments": len(members)}
    return results


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
    if args.target_advance:
        if args.device != "ffuzz":
            raise SystemExit("--target-advance is for --device ffuzz only")
        global TARGET_ADVANCE
        TARGET_ADVANCE = args.target_advance
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
    torch.set_float32_matmul_precision("highest" if args.no_tf32 else "high")
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
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

    DiagonalSSMLayer.fft_size = args.fft_size
    processor = build_processor(args)
    rate = device_rate(args.device)
    mrstft_params = at_rate(processor, rate)
    parameters = sum(p.numel() for p in processor.parameters())
    loss = FlexibleLoss(
        losses=[
            {"name": "l1_loss", "weight": l1_weight, "alias": "l1"},
            {
                "name": "mrstft_loss",
                "weight": mrstft_weight,
                "alias": "mrstft",
                **mrstft_params,
            },
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
    system.lr_halvings = args.lr_halvings
    system.lr_halving_steps = args.lr_halving_steps

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
            *([] if args.no_early_stopping else [early_stopping]),
            *([] if args.lr_halvings is None else [FixedSchedule(args.lr_halvings)]),
            *(
                []
                if args.lr_halving_steps is None
                else [FixedStepSchedule(args.lr_halving_steps)]
            ),
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
    trainval = data_module("trainval", args.batch_size, args.device)
    keep_segments_per_take(trainval, args.train_segments_per_take)
    sampler = (
        None
        if args.log_sampler_epochs is None
        else record_sampler(trainval, args.log_sampler_epochs)
    )
    trainer.fit(system, datamodule=trainval, ckpt_path=resume_from)
    minutes = (time.perf_counter() - started) / 60.0
    # Which twelve segments this run validated on. The split is drawn inside fit,
    # after the processor is built, so two architectures at the same --seed normally
    # get almost disjoint splits - one segment in twelve, measured between
    # ssm-wavenet and s4-tf-l-16. Recording it makes a run say which data it saw
    # rather than leaving it to be reproduced, and lets a paired design check that
    # its arms really shared a split.
    val_indices = sorted(getattr(trainval.val_dataset, "indices", []))

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
            system,
            datamodule=data_module("test", device=args.device),
            ckpt_path=str(path),
            verbose=False,
        )
        tests[label] = {key: float(value) for key, value in output[0].items()}
    by_setting = val_by_setting = None
    if args.device in DEVICES:
        test_data = data_module("test", device=args.device)
        test_data.setup("test")
        by_setting = metrics_by_setting(system, last, test_data.test_dataset)
        val_by_setting = metrics_by_setting(system, last, trainval.val_dataset)

    record = {
        "run_id": args.run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        **commits,
        "benchmark": (
            f"ToneTwist Big Muff {SETTING}, nablafx protocol"
            if args.device == "bigmuff"
            else DEVICES[args.device].get(
                "benchmark", f"ToneTwist {args.device} parametric, nablafx protocol"
            )
        ),
        "device": args.device,
        "held_out_settings": DEVICES.get(args.device, {}).get("held_out"),
        "target_advance": args.target_advance,
        "model": args.model,
        "parameters": parameters,
        "seed": args.seed,
        "lr": lr,
        "loss_weights": {"l1": l1_weight, "mrstft": mrstft_weight},
        "gradient_clip_val": args.clip_value,
        "honor_optim": args.honor_optim,
        "plateau_threshold": args.plateau_threshold,
        "lr_halvings": args.lr_halvings,
        "lr_halving_steps": args.lr_halving_steps,
        "fft_size": args.fft_size,
        "no_tf32": args.no_tf32,
        "sample_rate": rate,
        "no_early_stopping": args.no_early_stopping,
        "split_seed": args.split_seed,
        "val_indices": val_indices,
        "train_segments_per_take": args.train_segments_per_take,
        "train_segments": len(trainval.train_dataset),
        # Training order at the requested epochs, as indices of the trainval dataset.
        "sampler_order": (
            None
            if sampler is None
            else {
                str(epoch): [trainval.train_dataset.indices[i] for i in order]
                for epoch, order in sampler["sampler"].log.items()
            }
        ),
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
                "cond": args.cond,
                "cond_head": args.cond_head,
                "cond_hidden": args.cond_hidden,
                "head_act": args.head_act,
                "act_type": args.act_type,
            }
            if args.model == "ssm-wavenet"
            else None
        ),
        "greybox": (
            {
                "gb_stages": args.gb_stages,
                "gb_taps": args.gb_taps,
                "gb_bias": args.gb_bias,
                "gb_slow": args.gb_slow,
            }
            if args.model == "deepgreybox"
            else None
        ),
        "resumed": resume_from is not None,
        "global_step": trainer.global_step,
        # None, not 0, when the callback was never installed: zero would read as
        # "early stopping was watching and never fired", which is a different fact.
        "stopped_epoch": (
            None if args.no_early_stopping else early_stopping.stopped_epoch
        ),
        "max_steps": args.max_steps,
        "best_val_loss": float(checkpoint.best_model_score or "nan"),
        "best_checkpoint": Path(checkpoint.best_model_path).name,
        "minutes_this_session": round(minutes, 2),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        # Allocator cache included: what the GPU must really have free.
        "peak_gpu_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "test_last": tests["last"],
        "test_best": tests["best"],
        "test_last_by_setting": by_setting,
        # Validation segments come from the same dry files as training: seen settings,
        # optimistic material.
        "val_last_by_setting": val_by_setting,
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
