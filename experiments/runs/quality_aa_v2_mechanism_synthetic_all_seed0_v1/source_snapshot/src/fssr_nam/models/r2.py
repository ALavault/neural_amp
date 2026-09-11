"""Prospective antialiased model primitives for FSSR-R2-v1."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

from .oversampling import FullRateIsland
from .r1 import RF2047_DILATIONS, DepthwiseSeparableResidual
from .slow import SlowStateController
from .spline import SmoothHermiteSpline
from .structured import CausalDelay, CausalFIR, _batch

AA_MODES = ("off", "full_island_x2", "adaa1", "teacher_x4")


class HermiteCustomTanh(nn.Module):
    """C1 Hermite activation initialized to ``tanh(x / alpha)``."""

    def __init__(
        self,
        num_knots: int = 33,
        minimum: float = -4.0,
        maximum: float = 4.0,
        alpha: float = 1.8,
    ) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError("CustomTanh alpha must be positive")
        self.alpha = float(alpha)
        self.spline = SmoothHermiteSpline(num_knots, minimum, maximum)
        with torch.no_grad():
            values = torch.tanh(self.spline.knots / self.alpha)
            slopes = (1.0 - values.square()) / self.alpha
            self.spline.values.copy_(values)
            self.spline.slopes.copy_(slopes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.spline(inputs)

    def derivative(self, inputs: Tensor) -> Tensor:
        return self.spline.derivative(inputs)

    def antiderivative(self, inputs: Tensor) -> Tensor:
        return self.spline.antiderivative(inputs)

    def curvature_penalty(self) -> Tensor:
        return self.spline.curvature_penalty()


class FirstOrderADAA(nn.Module):
    """Stateful first-order ADAA with a stable divided-difference limit."""

    latency_samples = 1

    def __init__(
        self, activation: HermiteCustomTanh, threshold: float = 1.0e-4
    ) -> None:
        super().__init__()
        if threshold <= 0.0:
            raise ValueError("ADAA threshold must be positive")
        self.activation = activation
        self.threshold = float(threshold)
        self.register_buffer("_previous", torch.empty(0), persistent=False)

    def reset_state(self) -> None:
        self._previous = self._previous.new_empty(0)

    def _run(self, inputs: Tensor, previous: Tensor) -> Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] < 1:
            raise ValueError("ADAA input must have a non-empty time axis")
        if not torch.isfinite(inputs).all():
            raise ValueError("ADAA input must be finite")
        if previous.shape != inputs.shape[:-1]:
            raise ValueError("ADAA stream shape changed without reset")
        delayed = torch.cat((previous[..., None], inputs[..., :-1]), dim=-1)
        difference = inputs - delayed
        midpoint = 0.5 * (inputs + delayed)
        small = torch.abs(difference) < self.threshold
        safe_difference = torch.where(small, torch.ones_like(difference), difference)
        divided = (
            self.activation.antiderivative(inputs)
            - self.activation.antiderivative(delayed)
        ) / safe_difference
        return torch.where(small, self.activation(midpoint), divided)

    def forward(self, inputs: Tensor) -> Tensor:
        previous = inputs.new_zeros(inputs.shape[:-1])
        return self._run(inputs, previous)

    def stream(self, inputs: Tensor) -> Tensor:
        if self._previous.numel() == 0:
            previous = inputs.new_zeros(inputs.shape[:-1])
        else:
            previous = self._previous
        output = self._run(inputs, previous)
        self._previous = inputs[..., -1].detach().clone()
        return output

    def curvature_penalty(self) -> Tensor:
        return self.activation.curvature_penalty()


def _make_shaper(aa_mode: str, num_knots: int) -> nn.Module:
    activation = HermiteCustomTanh(num_knots=num_knots)
    return FirstOrderADAA(activation) if aa_mode == "adaa1" else activation


class R2SplineCore(nn.Module):
    """Mono- or two-spline core with an ADAA-capable analytic shaper."""

    def __init__(
        self,
        *,
        kind: str,
        aa_mode: str,
        taps: int,
        num_knots: int,
    ) -> None:
        super().__init__()
        if kind not in {"mono", "cascade"}:
            raise ValueError("R2 core kind must be mono or cascade")
        if aa_mode not in {"off", "adaa1", "full_island_x2", "teacher_x4"}:
            raise ValueError("unsupported R2 AA mode")
        self.kind = kind
        self.filters = nn.ModuleList(
            CausalFIR(taps) for _ in range(2 if kind == "mono" else 3)
        )
        count = 1 if kind == "mono" else 2
        shaper_mode = "adaa1" if aa_mode == "adaa1" else "off"
        self.shapers = nn.ModuleList(
            _make_shaper(shaper_mode, num_knots) for _ in range(count)
        )
        self.drives = nn.Parameter(torch.ones(count))
        self.offsets = nn.Parameter(torch.zeros(count))
        self.output_gain = nn.Parameter(torch.tensor(1.0))

    def reset_state(self) -> None:
        for module in (*self.filters, *self.shapers):
            reset = getattr(module, "reset_state", None)
            if reset is not None:
                reset()

    def _shape(self, index: int, inputs: Tensor, *, streaming: bool) -> Tensor:
        shaper = self.shapers[index]
        process = getattr(shaper, "stream", None) if streaming else None
        return (process or shaper)(inputs)

    def _run(
        self,
        signal: Tensor,
        modulation: Tensor | None,
        *,
        streaming: bool,
    ) -> Tensor:
        batched, scalar = _batch(signal)
        if modulation is None:
            drive_factor = torch.ones_like(batched)
            offset_delta = torch.zeros_like(batched)
            gain_factor = torch.ones_like(batched)
        else:
            if modulation.shape != (len(batched), 3, batched.shape[-1]):
                raise ValueError("slow modulation must have shape (batch,3,time)")
            drive_factor = modulation[:, 0]
            offset_delta = modulation[:, 1]
            gain_factor = modulation[:, 2]
        filtered = (
            self.filters[0].stream(batched) if streaming else self.filters[0](batched)
        )
        shaper_input = (
            self.drives[0] * drive_factor * filtered + self.offsets[0] + offset_delta
        )
        shaped = self._shape(0, shaper_input, streaming=streaming)
        filtered = (
            self.filters[1].stream(shaped) if streaming else self.filters[1](shaped)
        )
        if self.kind == "cascade":
            shaped = self._shape(
                1,
                self.drives[1] * filtered + self.offsets[1],
                streaming=streaming,
            )
            filtered = (
                self.filters[2].stream(shaped) if streaming else self.filters[2](shaped)
            )
        output = self.output_gain * gain_factor * filtered
        return output[0] if scalar else output

    def forward_modulated(self, signal: Tensor, modulation: Tensor | None) -> Tensor:
        return self._run(signal, modulation, streaming=False)

    def stream_modulated(self, signal: Tensor, modulation: Tensor | None) -> Tensor:
        return self._run(signal, modulation, streaming=True)

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_modulated(signal, None)

    def stream(self, signal: Tensor) -> Tensor:
        return self.stream_modulated(signal, None)

    def regularization(self) -> Tensor:
        penalties = [shaper.curvature_penalty() for shaper in self.shapers]
        return torch.stack(penalties).sum()


class R2FSSRBranch(nn.Module):
    """RF2047/16-channel/slow-GRU16 FSSR branch at one internal rate."""

    def __init__(
        self,
        *,
        core_kind: str,
        aa_mode: str,
        rate_scale: int,
        taps: int = 17,
        num_knots: int = 33,
        residual_channels: int = 16,
        slow_hidden_size: int = 16,
        slow_decimation: int = 64,
        residual_dilations: tuple[int, ...] = RF2047_DILATIONS,
    ) -> None:
        super().__init__()
        if rate_scale not in {1, 2, 4}:
            raise ValueError("rate_scale must be one, two, or four")
        scaled_taps = (taps - 1) * rate_scale + 1
        self.core = R2SplineCore(
            kind=core_kind,
            aa_mode=aa_mode,
            taps=scaled_taps,
            num_knots=num_knots,
        )
        if residual_channels < 1 or slow_hidden_size < 1:
            raise ValueError("R2 residual and slow widths must be positive")
        if slow_decimation < 1:
            raise ValueError("R2 slow cadence must be positive")
        if not residual_dilations or any(value < 1 for value in residual_dilations):
            raise ValueError("R2 residual dilations must be positive")
        self.slow = SlowStateController(
            hidden_size=slow_hidden_size,
            decimation=slow_decimation * rate_scale,
        )
        self.residual = DepthwiseSeparableResidual(
            channels=residual_channels,
            dilations=tuple(dilation * rate_scale for dilation in residual_dilations),
        )
        self.rate_scale = rate_scale
        self.receptive_field = self.residual.receptive_field
        self.core_receptive_field = 1 + sum(
            module.taps - 1 for module in self.core.filters
        )

    def reset_state(self) -> None:
        self.core.reset_state()
        self.slow.reset_state()
        self.residual.reset_state()

    def _run(self, signal: Tensor, *, streaming: bool) -> Tensor:
        batched, scalar = _batch(signal)
        modulation = self.slow.stream(batched) if streaming else self.slow(batched)
        core = (
            self.core.stream_modulated(batched, modulation)
            if streaming
            else self.core.forward_modulated(batched, modulation)
        )
        features = torch.stack((batched, core), dim=1)
        residual = (
            self.residual.stream(features) if streaming else self.residual(features)
        )
        output = core + residual
        return output[0] if scalar else output

    def forward(self, signal: Tensor) -> Tensor:
        return self._run(signal, streaming=False)

    def stream(self, signal: Tensor) -> Tensor:
        return self._run(signal, streaming=True)


class AAFSSR(nn.Module):
    """Deployable or teacher AA-FSSR composition defined by the R2 protocol."""

    sample_rate_hz = 48_000

    def __init__(
        self,
        *,
        core_kind: str,
        aa_mode: str,
        taps: int = 17,
        num_knots: int = 33,
        residual_channels: int = 16,
        slow_hidden_size: int = 16,
        slow_decimation: int = 64,
        residual_dilations: tuple[int, ...] = RF2047_DILATIONS,
    ) -> None:
        super().__init__()
        if aa_mode not in AA_MODES:
            raise ValueError(f"aa_mode must be one of {AA_MODES}")
        factor = {"off": 1, "adaa1": 1, "full_island_x2": 2, "teacher_x4": 4}[aa_mode]
        branch = R2FSSRBranch(
            core_kind=core_kind,
            aa_mode=aa_mode,
            rate_scale=factor,
            taps=taps,
            num_knots=num_knots,
            residual_channels=residual_channels,
            slow_hidden_size=slow_hidden_size,
            slow_decimation=slow_decimation,
            residual_dilations=residual_dilations,
        )
        self.processor = FullRateIsland(branch, factor=factor) if factor > 1 else branch
        self.output_delay = CausalDelay(1) if aa_mode == "adaa1" else None
        self.core_kind = core_kind
        self.aa_mode = aa_mode
        self.internal_sample_rate = self.sample_rate_hz * factor
        self.dilation_scale = factor
        self.latency_samples = (
            16
            if factor > 1
            else int(getattr(branch.core.shapers[0], "latency_samples", 0))
        )
        self.core_receptive_field = branch.core_receptive_field
        self.residual_channels = residual_channels
        self.slow_hidden_size = slow_hidden_size
        self.slow_decimation = slow_decimation

    def reset_state(self) -> None:
        self.processor.reset_state()
        if self.output_delay is not None:
            self.output_delay.reset_state()

    def forward(self, signal: Tensor) -> Tensor:
        output = self.processor(signal)
        return self.output_delay(output) if self.output_delay is not None else output

    def stream(self, signal: Tensor) -> Tensor:
        output = self.processor.stream(signal)
        return (
            self.output_delay.stream(output)
            if self.output_delay is not None
            else output
        )


def scale_a2_config_for_internal_rate(config: dict[str, Any], factor: int) -> dict:
    """Scale packed A2 temporal declarations while preserving physical RF."""
    if factor not in {1, 2, 4}:
        raise ValueError("A2 internal-rate factor must be one, two, or four")
    resolved = deepcopy(config)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "dilations" in value:
                value["dilations"] = [factor * int(item) for item in value["dilations"]]
            head = value.get("head")
            if isinstance(head, dict) and "kernel_size" in head:
                taps = int(head["kernel_size"])
                head["kernel_size"] = (taps - 1) * factor + 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(resolved)
    return resolved


def replace_tanh_with_hermite(model: nn.Module, *, adaa: bool = False) -> int:
    """Replace all PyTorch Tanh activations in an instantiated A2 topology."""
    replaced = 0
    for name, child in tuple(model.named_children()):
        if isinstance(child, nn.Tanh):
            activation: nn.Module = HermiteCustomTanh()
            if adaa:
                activation = FirstOrderADAA(activation)
            setattr(model, name, activation)
            replaced += 1
        else:
            replaced += replace_tanh_with_hermite(child, adaa=adaa)
    return replaced
