"""Causal long-horizon residuals and spline compositions for FSSR-R1."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .residual import FastResidualTCN, residual_energy_ratio
from .slow import SlowStateController
from .spline import SmoothHermiteSpline
from .structured import CausalFIR, S0Structured, _batch

RF31_DILATIONS = (1, 2, 4, 8)
RF2047_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


class CausalDepthwiseSeparableConv1d(nn.Module):
    """Length-preserving depthwise/pointwise convolution with stream state."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if dilation < 1:
            raise ValueError("dilation must be positive")
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.history = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.register_buffer("_stream_state", torch.empty(0), persistent=False)

    def _validate(self, signal: Tensor) -> None:
        if signal.ndim != 3 or signal.shape[1] != self.channels:
            raise ValueError(f"signal must have shape (batch, {self.channels}, time)")
        if signal.shape[-1] < 1:
            raise ValueError("stream blocks must contain at least one sample")

    def reset_state(self) -> None:
        self._stream_state = self._stream_state.new_empty(0)

    def forward(self, signal: Tensor) -> Tensor:
        self._validate(signal)
        padded = functional.pad(signal, (self.history, 0))
        return self.pointwise(self.depthwise(padded))

    def stream(self, signal: Tensor) -> Tensor:
        self._validate(signal)
        if self._stream_state.numel() == 0:
            self._stream_state = signal.new_zeros(
                signal.shape[0], signal.shape[1], self.history
            )
        elif self._stream_state.shape[:2] != signal.shape[:2]:
            raise ValueError("stream shape changed without reset")
        joined = torch.cat((self._stream_state, signal), dim=-1)
        output = self.pointwise(self.depthwise(joined))
        self._stream_state = (
            joined[..., -self.history :] if self.history else joined[..., :0]
        )
        return output


class DepthwiseSeparableResidual(nn.Module):
    """Bounded causal residual with an exact dilation-defined receptive field."""

    def __init__(
        self,
        input_channels: int = 2,
        channels: int = 8,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = RF2047_DILATIONS,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if channels < 1:
            raise ValueError("channels must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if not dilations or any(dilation < 1 for dilation in dilations):
            raise ValueError("dilations must be positive and non-empty")
        self.input_channels = input_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.input_projection = nn.Conv1d(input_channels, channels, 1)
        self.layers = nn.ModuleList(
            CausalDepthwiseSeparableConv1d(channels, kernel_size, dilation)
            for dilation in self.dilations
        )
        self.output_projection = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.scale_logit = nn.Parameter(torch.tensor(math.log(0.05 / 0.95)))
        self.receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)

    @property
    def residual_scale(self) -> Tensor:
        return 0.5 * torch.sigmoid(self.scale_logit)

    @property
    def estimated_macs_per_sample(self) -> int:
        projection = self.input_channels * self.channels + self.channels
        per_layer = self.channels * self.kernel_size + self.channels**2
        return projection + len(self.layers) * per_layer

    def _validate(self, features: Tensor) -> None:
        if features.ndim != 3 or features.shape[1] != self.input_channels:
            raise ValueError(
                f"features must have shape (batch, {self.input_channels}, time)"
            )
        if features.shape[-1] < 1:
            raise ValueError("features must contain at least one sample")

    def reset_state(self) -> None:
        for layer in self.layers:
            layer.reset_state()

    def _run(self, features: Tensor, *, streaming: bool) -> Tensor:
        self._validate(features)
        hidden = self.input_projection(features)
        for layer in self.layers:
            update = layer.stream(hidden) if streaming else layer(hidden)
            hidden = hidden + functional.leaky_relu(update, negative_slope=0.01)
        raw = self.output_projection(hidden)
        return self.residual_scale * torch.tanh(raw[:, 0])

    def forward(self, features: Tensor) -> Tensor:
        return self._run(features, streaming=False)

    def stream(self, features: Tensor) -> Tensor:
        return self._run(features, streaming=True)


class RF31Residual(FastResidualTCN):
    """Historical S3 full-convolution control with receptive field 31."""

    def __init__(self, input_channels: int = 2, channels: int = 8) -> None:
        super().__init__(
            input_channels=input_channels,
            channels=channels,
            kernel_size=3,
            dilations=RF31_DILATIONS,
        )


class RF2047Residual(DepthwiseSeparableResidual):
    """Ten-layer depthwise-separable candidate with receptive field 2047."""

    def __init__(self, input_channels: int = 2, channels: int = 8) -> None:
        super().__init__(
            input_channels=input_channels,
            channels=channels,
            kernel_size=3,
            dilations=RF2047_DILATIONS,
        )


class CascadeSplineCore(nn.Module):
    """Identity-initialized ``H0 -> spline1 -> H1 -> spline2 -> H2`` core."""

    latency_samples = 0

    def __init__(self, taps: int = 17, num_knots: int = 17) -> None:
        super().__init__()
        self.h0 = CausalFIR(taps)
        self.spline1 = SmoothHermiteSpline(num_knots)
        self.h1 = CausalFIR(taps)
        self.spline2 = SmoothHermiteSpline(num_knots)
        self.h2 = CausalFIR(taps)
        self.drive1 = nn.Parameter(torch.tensor(1.0))
        self.offset1 = nn.Parameter(torch.tensor(0.0))
        self.drive2 = nn.Parameter(torch.tensor(1.0))
        self.offset2 = nn.Parameter(torch.tensor(0.0))
        self.output_gain = nn.Parameter(torch.tensor(1.0))

    def reset_state(self) -> None:
        self.h0.reset_state()
        self.h1.reset_state()
        self.h2.reset_state()

    def forward_modulated(self, signal: Tensor, modulation: Tensor | None) -> Tensor:
        return self._modulated(signal, modulation, streaming=False)

    def stream_modulated(self, signal: Tensor, modulation: Tensor | None) -> Tensor:
        return self._modulated(signal, modulation, streaming=True)

    def _modulated(
        self, signal: Tensor, modulation: Tensor | None, *, streaming: bool
    ) -> Tensor:
        batched, squeeze = _batch(signal)
        if modulation is None:
            drive_factor = 1.0
            offset_delta = 0.0
            gain_factor = 1.0
        else:
            if squeeze and modulation.ndim == 2:
                modulation = modulation[None, :, :]
            if modulation.shape != (len(batched), 3, batched.shape[-1]):
                raise ValueError("modulation must have shape (batch,3,samples)")
            drive_factor = modulation[:, 0]
            offset_delta = modulation[:, 1]
            gain_factor = modulation[:, 2]
        filtered0 = self.h0.stream(batched) if streaming else self.h0(batched)
        shaped1 = self.spline1(
            self.drive1 * drive_factor * filtered0 + self.offset1 + offset_delta
        )
        filtered1 = self.h1.stream(shaped1) if streaming else self.h1(shaped1)
        shaped2 = self.spline2(self.drive2 * filtered1 + self.offset2)
        filtered2 = self.h2.stream(shaped2) if streaming else self.h2(shaped2)
        result = self.output_gain * gain_factor * filtered2
        return result[0] if squeeze else result

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_modulated(signal, None)

    def stream(self, signal: Tensor) -> Tensor:
        return self.stream_modulated(signal, None)

    def regularization(self) -> Tensor:
        return self.spline1.curvature_penalty() + self.spline2.curvature_penalty()

    def r1_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        filters = (
            tuple(self.h0.parameters())
            + tuple(self.h1.parameters())
            + tuple(self.h2.parameters())
        )
        gains = (
            self.drive1,
            self.offset1,
            self.drive2,
            self.offset2,
            self.output_gain,
        )
        splines = tuple(self.spline1.parameters()) + tuple(self.spline2.parameters())
        return {
            "filters": filters,
            "gains": gains,
            "splines": splines,
            "core": tuple(self.parameters()),
        }


def _make_residual(
    receptive_field: int, channels: int
) -> FastResidualTCN | DepthwiseSeparableResidual:
    if receptive_field == 31:
        return RF31Residual(channels=channels)
    if receptive_field == 2047:
        return RF2047Residual(channels=channels)
    raise ValueError("receptive_field must be 31 or 2047")


class _R1Composition(nn.Module):
    latency_samples = 0

    def __init__(
        self,
        core: nn.Module,
        *,
        receptive_field: int,
        residual_channels: int,
        slow_hidden_size: int,
        slow_decimation: int,
    ) -> None:
        super().__init__()
        self.core = core
        self.slow = SlowStateController(slow_hidden_size, slow_decimation)
        self.residual = _make_residual(receptive_field, residual_channels)
        self.receptive_field = self.residual.receptive_field

    def reset_state(self) -> None:
        self.core.reset_state()
        self.slow.reset_state()
        self.residual.reset_state()

    def _components(
        self, signal: Tensor, *, streaming: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        batched, squeeze = _batch(signal)
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
        if squeeze:
            return output[0], core[0], residual[0]
        return output, core, residual

    def forward_components(self, signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self._components(signal, streaming=False)

    def stream_components(self, signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self._components(signal, streaming=True)

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_components(signal)[0]

    def stream(self, signal: Tensor) -> Tensor:
        return self.stream_components(signal)[0]

    def energy_ratio(self, signal: Tensor) -> Tensor:
        output, _, residual = self.forward_components(signal)
        return residual_energy_ratio(residual, output)

    def regularization(self) -> Tensor:
        return self.core.regularization()

    def r1_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        provider = getattr(self.core, "r1_parameter_groups", None)
        if provider is None:
            core_groups = _mono_core_parameter_groups(self.core)
        else:
            core_groups = provider()
        slow = tuple(self.slow.parameters())
        return {
            **core_groups,
            "core": core_groups["core"] + slow,
            "slow": slow,
            "residual": tuple(self.residual.parameters()),
        }

    def load_promoted_residual_state(
        self, source: RF2047Residual | Mapping[str, Tensor]
    ) -> None:
        """Atomically load a finite, shape- and dtype-exact promoted RF2047."""
        if not isinstance(self.residual, RF2047Residual):
            raise ValueError("promoted residual state requires an RF2047 composition")
        raw_state = (
            source.state_dict() if isinstance(source, RF2047Residual) else source
        )
        if not isinstance(raw_state, Mapping):
            raise TypeError(
                "promoted residual state must be a mapping or RF2047Residual"
            )
        expected = self.residual.state_dict()
        if set(raw_state) != set(expected):
            raise ValueError("promoted residual state keys do not match RF2047")
        validated: dict[str, Tensor] = {}
        for name, target in expected.items():
            value = raw_state[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"promoted residual tensor {name} is not a Tensor")
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(
                    f"promoted residual tensor {name} is incompatible with RF2047"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"promoted residual tensor {name} must be finite")
            validated[name] = value.detach().clone()
        self.residual.load_state_dict(validated, strict=True)
        self.residual.reset_state()


def _mono_core_parameter_groups(
    core: nn.Module,
) -> dict[str, tuple[nn.Parameter, ...]]:
    if not isinstance(core, S0Structured):
        raise TypeError("unsupported mono core")
    return {
        "filters": tuple(core.pre.parameters()) + tuple(core.post.parameters()),
        "gains": (core.drive, core.offset, core.output_gain),
        "splines": tuple(core.shaper.parameters()),
        "core": tuple(core.parameters()),
    }


class R1Mono(_R1Composition):
    """One-spline R1 control/candidate with an RF31 or RF2047 residual."""

    kind = "mono"

    def __init__(
        self,
        taps: int = 17,
        num_knots: int = 17,
        residual_channels: int = 8,
        receptive_field: int = 2047,
        slow_hidden_size: int = 8,
        slow_decimation: int = 64,
    ) -> None:
        super().__init__(
            S0Structured(taps=taps, num_knots=num_knots),
            receptive_field=receptive_field,
            residual_channels=residual_channels,
            slow_hidden_size=slow_hidden_size,
            slow_decimation=slow_decimation,
        )


class R1Cascade(_R1Composition):
    """Two-spline cascade candidate with the reused RF2047 residual control."""

    kind = "cascade"

    def __init__(
        self,
        taps: int = 17,
        num_knots: int = 17,
        residual_channels: int = 8,
        receptive_field: int = 2047,
        slow_hidden_size: int = 8,
        slow_decimation: int = 64,
        promoted_residual_state: RF2047Residual | Mapping[str, Tensor] | None = None,
    ) -> None:
        super().__init__(
            CascadeSplineCore(taps=taps, num_knots=num_knots),
            receptive_field=receptive_field,
            residual_channels=residual_channels,
            slow_hidden_size=slow_hidden_size,
            slow_decimation=slow_decimation,
        )
        if promoted_residual_state is not None:
            self.load_promoted_residual_state(promoted_residual_state)
