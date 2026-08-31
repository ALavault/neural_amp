"""Frozen curriculum and loss for AMP-QUALITY-TEACHER-v1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import toeplitz
from scipy.signal import correlate
from torch import Tensor, nn

from fssr_nam.losses.nablafx import NablafxLoss
from fssr_nam.losses.wright import esr_loss, preemphasize
from fssr_nam.models.quality_teacher import (
    QualityTeacherAmplifier,
    build_quality_teacher_model,
)
from fssr_nam.training.objectives import projection_gain_loss


def delay_audio(signal: Tensor, samples: int) -> Tensor:
    """Delay batched audio exactly without depending on another campaign."""
    if signal.ndim != 2 or samples < 0:
        raise ValueError("audio must be batched and delay must be non-negative")
    if samples == 0:
        return signal.clone()
    if samples >= signal.shape[-1]:
        return torch.zeros_like(signal)
    return torch.nn.functional.pad(signal, (samples, 0))[..., :-samples]


@dataclass(frozen=True)
class TeacherLossComponents:
    total: Tensor
    l1: Tensor
    mrstft: Tensor
    preemphasized_esr: Tensor
    projection_gain: Tensor


class TeacherLoss(nn.Module):
    """10 L1 + MR-STFT + pre-emphasized ESR + projection-gain penalty."""

    def __init__(
        self,
        *,
        l1_weight: float = 10.0,
        mrstft_weight: float = 1.0,
        preemphasized_esr_weight: float = 1.0,
        projection_gain_weight: float = 0.05,
        preemphasis: float = 0.95,
        spectral_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        weights = (
            l1_weight,
            mrstft_weight,
            preemphasized_esr_weight,
            projection_gain_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("teacher loss weights must be non-negative and nonzero")
        if not 0.0 <= preemphasis < 1.0:
            raise ValueError("teacher preemphasis must be in [0,1)")
        self.l1_weight = float(l1_weight)
        self.mrstft_weight = float(mrstft_weight)
        self.preemphasized_esr_weight = float(preemphasized_esr_weight)
        self.projection_gain_weight = float(projection_gain_weight)
        self.preemphasis = float(preemphasis)
        self.mrstft = (
            spectral_loss
            if spectral_loss is not None
            else NablafxLoss(l1_weight=0.0, mrstft_weight=1.0).mrstft
        )

    def components(self, output: Tensor, target: Tensor) -> TeacherLossComponents:
        if output.shape != target.shape or output.ndim != 2:
            raise ValueError("teacher loss expects paired (batch,time) tensors")
        if output.shape[-1] < 1:
            raise ValueError("teacher loss requires nonempty audio")
        l1 = torch.nn.functional.l1_loss(output, target)
        spectral = self.mrstft(output[:, None], target[:, None])
        emphasized_output = preemphasize(output, self.preemphasis)
        emphasized_target = preemphasize(target, self.preemphasis)
        emphasized_esr = esr_loss(emphasized_output, emphasized_target)
        gain = projection_gain_loss(output, target)
        total = (
            self.l1_weight * l1
            + self.mrstft_weight * spectral
            + self.preemphasized_esr_weight * emphasized_esr
            + self.projection_gain_weight * gain
        )
        return TeacherLossComponents(total, l1, spectral, emphasized_esr, gain)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return self.components(output, target).total


@dataclass(frozen=True)
class FIRInitialization:
    coefficients: NDArray[np.float32]
    lower_quartile_rms: float
    passages_total: int
    passages_selected: int
    passage_samples: int


def build_initialized_teacher(
    family: str,
    *,
    seed: int,
    fir_initialization: FIRInitialization,
    device: torch.device | None = None,
) -> QualityTeacherAmplifier:
    """Build paired seeded weights, then install the frozen low-RMS FIR fit."""
    model = build_quality_teacher_model(family, seed=seed)
    model.initialize_fir(torch.from_numpy(fir_initialization.coefficients.copy()))
    return model if device is None else model.to(device)


def _paired_float32(dry: ArrayLike, wet: ArrayLike) -> tuple[NDArray, NDArray]:
    x = np.asarray(dry, dtype=np.float32)
    y = np.asarray(wet, dtype=np.float32)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("FIR initialization expects equal-length mono pairs")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("FIR initialization pairs must be finite")
    return x, y


def fit_lower_quartile_fir(
    pairs: Sequence[tuple[ArrayLike, ArrayLike]],
    *,
    taps: int = 257,
    passage_samples: int = 48_000,
    ridge: float = 1.0e-6,
) -> FIRInitialization:
    """Fit a deterministic causal FIR on lower-dry-RMS training passages."""
    if taps < 1 or passage_samples < taps or ridge < 0.0 or not pairs:
        raise ValueError("FIR initialization dimensions are invalid")
    passages: list[tuple[NDArray, NDArray, float]] = []
    for dry, wet in pairs:
        x, y = _paired_float32(dry, wet)
        for start in range(0, len(x), passage_samples):
            stop = min(start + passage_samples, len(x))
            if stop - start < taps:
                continue
            x_passage = np.asarray(x[start:stop], dtype=np.float64)
            y_passage = np.asarray(y[start:stop], dtype=np.float64)
            rms = float(np.sqrt(np.mean(np.square(x_passage))))
            passages.append((x_passage, y_passage, rms))
    if not passages:
        raise ValueError("no complete FIR initialization passage is available")
    threshold = float(np.quantile([row[2] for row in passages], 0.25))
    selected = [row for row in passages if row[2] <= threshold]
    autocorrelation = np.zeros(taps, dtype=np.float64)
    cross_correlation = np.zeros(taps, dtype=np.float64)
    for dry, wet, _ in selected:
        center = len(dry) - 1
        autocorrelation += correlate(dry, dry, mode="full", method="fft")[
            center : center + taps
        ]
        cross_correlation += correlate(wet, dry, mode="full", method="fft")[
            center : center + taps
        ]
    normal = toeplitz(autocorrelation)
    scale = max(float(autocorrelation[0]), 1.0)
    normal.flat[:: taps + 1] += ridge * scale
    lag_coefficients = np.linalg.solve(normal, cross_correlation)
    convolution_coefficients = np.asarray(lag_coefficients[::-1], dtype=np.float32)
    if not np.isfinite(convolution_coefficients).all():
        raise RuntimeError("FIR initialization produced non-finite coefficients")
    return FIRInitialization(
        coefficients=convolution_coefficients,
        lower_quartile_rms=threshold,
        passages_total=len(passages),
        passages_selected=len(selected),
        passage_samples=passage_samples,
    )


@dataclass(frozen=True)
class TrainingSource:
    source_id: str
    dry: Tensor
    target: Tensor


@dataclass(frozen=True)
class Microchunk:
    source_id: str
    dry: Tensor
    delayed_target: Tensor
    valid_samples: int
    reset_before: bool


class ContiguousMicrochunkCursor:
    """Visit every source in order without crossing a state boundary."""

    def __init__(
        self,
        sources: Sequence[TrainingSource],
        *,
        chunk_samples: int,
        latency_samples: int,
    ) -> None:
        if not sources or chunk_samples < 1 or latency_samples < 0:
            raise ValueError("microchunk cursor configuration is invalid")
        normalized: list[TrainingSource] = []
        delayed: list[Tensor] = []
        for source in sources:
            if not source.source_id:
                raise ValueError("training source_id must be nonempty")
            if (
                source.dry.ndim != 1
                or source.target.ndim != 1
                or source.dry.shape != source.target.shape
                or source.dry.dtype != torch.float32
                or source.target.dtype != torch.float32
                or source.dry.shape[-1] < 1
                or not torch.isfinite(source.dry).all()
                or not torch.isfinite(source.target).all()
            ):
                raise ValueError("training sources must be finite paired float32 mono")
            normalized.append(source)
            delayed.append(delay_audio(source.target[None], latency_samples)[0])
        self.sources = tuple(normalized)
        self.delayed_targets = tuple(delayed)
        self.chunk_samples = chunk_samples
        self.source_index = 0
        self.position = 0

    def next(self) -> Microchunk:
        source = self.sources[self.source_index]
        reset_before = self.position == 0
        stop = min(self.position + self.chunk_samples, len(source.dry))
        valid = stop - self.position
        dry = source.dry.new_zeros(self.chunk_samples)
        target = source.target.new_zeros(self.chunk_samples)
        dry[:valid] = source.dry[self.position : stop]
        target[:valid] = self.delayed_targets[self.source_index][self.position : stop]
        self.position = stop
        if self.position == len(source.dry):
            self.source_index = (self.source_index + 1) % len(self.sources)
            self.position = 0
        return Microchunk(source.source_id, dry, target, valid, reset_before)


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    updates: int
    train_fast_fir: bool
    train_slow: bool


DEFAULT_CURRICULUM = (
    CurriculumPhase("fast_fir", 2_500, True, False),
    CurriculumPhase("s4_film", 2_500, False, True),
    CurriculumPhase("joint", 2_500, True, True),
)
DEFAULT_CHECKPOINTS = tuple(range(500, 7_501, 500))


def _set_phase_trainability(
    model: QualityTeacherAmplifier, phase: CurriculumPhase
) -> tuple[nn.Parameter, ...]:
    trainable: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        is_slow = name.startswith("observer.")
        enabled = phase.train_slow if is_slow else phase.train_fast_fir
        if is_slow and not model.slow_modulation_enabled:
            enabled = False
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    return tuple(trainable)


def _stream_state_snapshot(model: nn.Module) -> dict[tuple[str, str], object]:
    snapshot: dict[tuple[str, str], object] = {}
    for module_name, module in model.named_modules():
        for attribute in ("_state", "_stream_state"):
            value = getattr(module, attribute, None)
            if isinstance(value, Tensor):
                snapshot[(module_name, attribute)] = value.detach().clone()
            elif value is None:
                snapshot[(module_name, attribute)] = None
    return snapshot


def _restore_stream_state(
    model: nn.Module, snapshot: dict[tuple[str, str], object]
) -> None:
    modules = dict(model.named_modules())
    for (module_name, attribute), value in snapshot.items():
        setattr(modules[module_name], attribute, value)


def evaluate_validation_loss(
    model: QualityTeacherAmplifier,
    sources: Sequence[TrainingSource],
    loss_module: TeacherLoss,
    *,
    chunk_samples: int = 48_000,
) -> dict[str, float]:
    """Evaluate each source independently with causal state and no post-fit."""
    if not sources or chunk_samples < 1:
        raise ValueError("validation sources and chunk size are required")
    totals = {
        name: 0.0
        for name in ("total", "l1", "mrstft", "preemphasized_esr", "projection_gain")
    }
    total_samples = 0
    model.eval()
    with torch.inference_mode():
        for source in sources:
            model.reset_state()
            delayed = delay_audio(source.target[None], model.latency_samples)
            for start in range(0, len(source.dry), chunk_samples):
                stop = min(start + chunk_samples, len(source.dry))
                output = model.stream(source.dry[None, start:stop])
                components = loss_module.components(output, delayed[:, start:stop])
                weight = stop - start
                total_samples += weight
                for name in totals:
                    totals[name] += float(getattr(components, name).cpu()) * weight
    if total_samples == 0:
        raise ValueError("validation set contains no samples")
    return {name: value / total_samples for name, value in totals.items()}


@dataclass(frozen=True)
class TeacherTrainingResult:
    family: str
    seed: int
    updates: int
    selected_update: int
    history: tuple[dict[str, object], ...]
    checkpoints: tuple[dict[str, object], ...]


def train_teacher_trajectory(
    model: QualityTeacherAmplifier,
    *,
    train_sources: Sequence[TrainingSource],
    validation_sources: Sequence[TrainingSource],
    seed: int,
    loss_module: TeacherLoss | None = None,
    curriculum: Sequence[CurriculumPhase] = DEFAULT_CURRICULUM,
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
    chunk_samples: int = 48_000,
    microchunks_per_update: int = 3,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    gradient_clip_norm: float = 1.0,
) -> tuple[TeacherTrainingResult, dict[int, dict[str, Tensor]]]:
    """Train one state-carrying trajectory and select by validation loss only."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("teacher seed must be a non-negative integer")
    if (
        isinstance(model, QualityTeacherAmplifier)
        and getattr(model, "initialization_seed", None) != seed
    ):
        raise ValueError("teacher model must be built with the trajectory seed")
    if (
        not curriculum
        or any(phase.updates < 1 for phase in curriculum)
        or chunk_samples < 1
        or microchunks_per_update != 3
        or learning_rate <= 0.0
        or weight_decay < 0.0
        or gradient_clip_norm != 1.0
    ):
        raise ValueError("teacher optimization configuration drifted")
    total_updates = sum(phase.updates for phase in curriculum)
    checkpoint_tuple = tuple(checkpoints)
    expected_checkpoints = tuple(range(500, total_updates + 1, 500))
    if total_updates % 500 or checkpoint_tuple != expected_checkpoints:
        raise ValueError("teacher checkpoints must be every 500 through completion")
    torch.manual_seed(seed)
    if next(model.parameters()).is_cuda:
        torch.cuda.manual_seed_all(seed)
    loss_module = loss_module or TeacherLoss()
    device = next(model.parameters()).device
    normalized_train = tuple(
        TrainingSource(
            source.source_id,
            source.dry.to(device),
            source.target.to(device),
        )
        for source in train_sources
    )
    normalized_validation = tuple(
        TrainingSource(
            source.source_id,
            source.dry.to(device),
            source.target.to(device),
        )
        for source in validation_sources
    )
    cursor = ContiguousMicrochunkCursor(
        normalized_train,
        chunk_samples=chunk_samples,
        latency_samples=model.latency_samples,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    checkpoint_set = set(checkpoint_tuple)
    history: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    checkpoint_states: dict[int, dict[str, Tensor]] = {}
    update = 0
    model.train()
    for phase in curriculum:
        trainable = _set_phase_trainability(model, phase)
        for _ in range(phase.updates):
            update += 1
            optimizer.zero_grad(set_to_none=True)
            component_sums = {
                name: 0.0
                for name in (
                    "total",
                    "l1",
                    "mrstft",
                    "preemphasized_esr",
                    "projection_gain",
                )
            }
            processed = 0
            for _microchunk in range(microchunks_per_update):
                chunk = cursor.next()
                if chunk.reset_before:
                    model.reset_state()
                dry = chunk.dry[None]
                target = chunk.delayed_target[None, : chunk.valid_samples]
                if trainable:
                    output = model.stream(dry)[..., : chunk.valid_samples]
                    components = loss_module.components(output, target)
                    (components.total / microchunks_per_update).backward()
                else:
                    with torch.no_grad():
                        output = model.stream(dry)[..., : chunk.valid_samples]
                        components = loss_module.components(output, target)
                model.detach_stream_state()
                processed += chunk.valid_samples
                for name in component_sums:
                    component_sums[name] += float(
                        getattr(components, name).detach().cpu()
                    )
            if trainable:
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in trainable
                ):
                    raise RuntimeError(
                        f"non-finite teacher gradient at update {update}"
                    )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, gradient_clip_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(
                        f"non-finite teacher gradient norm at update {update}"
                    )
                optimizer.step()
                gradient_norm_value = float(gradient_norm.detach().cpu())
            else:
                gradient_norm_value = 0.0
            averaged = {
                name: value / microchunks_per_update
                for name, value in component_sums.items()
            }
            if not all(np.isfinite(value) for value in averaged.values()):
                raise RuntimeError(f"non-finite teacher loss at update {update}")
            history.append(
                {
                    "update": update,
                    "phase": phase.name,
                    "samples": processed,
                    "gradient_norm": gradient_norm_value,
                    **averaged,
                }
            )
            if update in checkpoint_set:
                stream_state = _stream_state_snapshot(model)
                validation = evaluate_validation_loss(
                    model,
                    normalized_validation,
                    loss_module,
                    chunk_samples=chunk_samples,
                )
                _restore_stream_state(model, stream_state)
                model.train()
                checkpoint_rows.append({"update": update, "validation": validation})
                checkpoint_states[update] = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
    selected = min(
        checkpoint_rows,
        key=lambda row: (float(row["validation"]["total"]), int(row["update"])),
    )
    return (
        TeacherTrainingResult(
            family=model.family,
            seed=seed,
            updates=total_updates,
            selected_update=int(selected["update"]),
            history=tuple(history),
            checkpoints=tuple(checkpoint_rows),
        ),
        checkpoint_states,
    )
