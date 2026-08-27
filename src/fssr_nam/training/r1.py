"""Numerical loss and staged-optimization controls for FSSR-R1."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from fssr_nam.losses import WrightLoss as WrightLoss
from fssr_nam.losses import dc_loss, esr_loss, preemphasize

WRIGHT_PREEMPHASIS = (-0.85, 1.0)
WRIGHT_ESR_WEIGHT = 0.75
WRIGHT_DC_WEIGHT = 0.25
WRIGHT_EPSILON = 1.0e-5


wright_preemphasis = preemphasize


def wright_esr(output: Tensor, target: Tensor) -> Tensor:
    """CoreAudioML ESR after the official causal pre-emphasis filter."""
    emphasized_output = wright_preemphasis(output)
    emphasized_target = wright_preemphasis(target)
    return esr_loss(emphasized_output, emphasized_target, WRIGHT_EPSILON)


def wright_dc(output: Tensor, target: Tensor) -> Tensor:
    """CoreAudioML DC loss, with time represented by the final axis."""
    return dc_loss(output, target, WRIGHT_EPSILON)


@dataclass(frozen=True)
class TrainingPhase:
    """Inclusive optimizer-step interval and semantic trainable groups."""

    name: str
    start_step: int
    end_step: int
    trainable: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("phase name must not be empty")
        if (
            isinstance(self.start_step, bool)
            or not isinstance(self.start_step, int)
            or isinstance(self.end_step, bool)
            or not isinstance(self.end_step, int)
        ):
            raise TypeError("phase bounds must be integers")
        if self.start_step < 1 or self.end_step < self.start_step:
            raise ValueError("phase bounds must be positive and ordered")
        if not self.trainable or any(not item for item in self.trainable):
            raise ValueError("phase trainable groups must not be empty")
        if len(set(self.trainable)) != len(self.trainable):
            raise ValueError("phase trainable groups must be unique")


class PhaseScheduler:
    """Apply pre-registered freeze masks without rebuilding an optimizer."""

    def __init__(self, phases: Sequence[TrainingPhase]) -> None:
        if not phases:
            raise ValueError("at least one phase is required")
        self.phases = tuple(phases)
        if self.phases[0].start_step != 1:
            raise ValueError("phase schedule must start at step 1")
        for previous, current in zip(self.phases, self.phases[1:], strict=False):
            if current.start_step != previous.end_step + 1:
                raise ValueError("phase schedule must be contiguous and ordered")

    @classmethod
    def from_config(cls, config: Sequence[Mapping[str, Any]]) -> PhaseScheduler:
        phases: list[TrainingPhase] = []
        for item in config:
            if "start_step" in item or "end_step" in item:
                start_key, end_key = "start_step", "end_step"
            else:
                start_key, end_key = "start_epoch", "end_epoch"
            required = {"name", start_key, end_key, "trainable"}
            if set(item) != required:
                raise ValueError(f"unexpected phase fields for {item.get('name')!r}")
            trainable = item["trainable"]
            if isinstance(trainable, str) or not isinstance(trainable, Sequence):
                raise TypeError("trainable must be a sequence of group names")
            phases.append(
                TrainingPhase(
                    name=str(item["name"]),
                    start_step=item[start_key],
                    end_step=item[end_key],
                    trainable=tuple(str(group) for group in trainable),
                )
            )
        return cls(phases)

    def phase_at(self, step: int) -> TrainingPhase:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        for phase in self.phases:
            if phase.start_step <= step <= phase.end_step:
                return phase
        raise ValueError(f"step {step} is outside the phase schedule")

    def apply(self, model: nn.Module, step: int) -> TrainingPhase:
        """Set ``requires_grad`` exactly for the phase containing ``step``."""
        phase = self.phase_at(step)
        parameters = tuple(model.parameters())
        if "all" in phase.trainable:
            if phase.trainable != ("all",):
                raise ValueError("all cannot be combined with other trainable groups")
            selected = {id(parameter) for parameter in parameters}
        else:
            groups = _parameter_groups(model)
            unknown = set(phase.trainable) - set(groups)
            if unknown:
                raise ValueError(f"unknown trainable groups: {sorted(unknown)}")
            selected = {
                id(parameter) for name in phase.trainable for parameter in groups[name]
            }
        for parameter in parameters:
            trainable = id(parameter) in selected
            parameter.requires_grad_(trainable)
            if not trainable:
                parameter.grad = None
        return phase


def _parameter_groups(model: nn.Module) -> Mapping[str, tuple[nn.Parameter, ...]]:
    provider = getattr(model, "r1_parameter_groups", None)
    if provider is None or not callable(provider):
        raise TypeError("model must provide r1_parameter_groups()")
    raw_groups = provider()
    if not isinstance(raw_groups, Mapping):
        raise TypeError("r1_parameter_groups() must return a mapping")
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    groups: dict[str, tuple[nn.Parameter, ...]] = {}
    for name, raw_parameters in raw_groups.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parameter group names must be non-empty strings")
        if not isinstance(raw_parameters, Iterable):
            raise TypeError(f"parameter group {name!r} must be iterable")
        parameters = tuple(raw_parameters)
        if any(not isinstance(parameter, nn.Parameter) for parameter in parameters):
            raise TypeError(f"parameter group {name!r} contains a non-parameter")
        if any(id(parameter) not in model_parameter_ids for parameter in parameters):
            raise ValueError(f"parameter group {name!r} contains a foreign parameter")
        groups[name] = parameters
    return groups


@dataclass(frozen=True)
class ResidualPenaltyRamp:
    """Clamped linear residual-regularization schedule."""

    start: float = 1.0e-2
    end: float = 1.0e-3
    start_step: int = 1
    end_step: int = 5_000

    def __post_init__(self) -> None:
        if self.start < 0.0 or self.end < 0.0:
            raise ValueError("residual penalties must be non-negative")
        if self.start_step < 1 or self.end_step <= self.start_step:
            raise ValueError("residual ramp bounds must be positive and increasing")

    def __call__(self, step: int) -> float:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= self.start_step:
            return float(self.start)
        if step >= self.end_step:
            return float(self.end)
        fraction = (step - self.start_step) / (self.end_step - self.start_step)
        return float(self.start + fraction * (self.end - self.start))


def residual_penalty_weight(step: int) -> float:
    """Return the locked R1 ``1e-2 -> 1e-3`` penalty at an optimizer step."""
    return ResidualPenaltyRamp()(step)
