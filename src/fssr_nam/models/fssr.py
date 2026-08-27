"""FSSR-NAM S1, S2, and S3 compositions."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .oversampling import LocalOversampledSpline2x
from .residual import FastResidualTCN, residual_energy_ratio
from .slow import SlowStateController
from .structured import CausalDelay, S0Structured, _batch


class S1Slow(nn.Module):
    code = "S1"
    latency_samples = 0

    def __init__(
        self,
        taps: int = 17,
        num_knots: int = 17,
        hidden_size: int = 8,
        decimation: int = 64,
    ):
        super().__init__()
        self.core = S0Structured(taps, num_knots)
        self.slow = SlowStateController(hidden_size, decimation)

    def reset_state(self) -> None:
        self.core.reset_state()
        self.slow.reset_state()

    def forward(self, signal: Tensor) -> Tensor:
        return self.core.forward_modulated(signal, self.slow(signal))

    def stream(self, signal: Tensor) -> Tensor:
        return self.core.stream_modulated(signal, self.slow.stream(signal))


class S2Residual(nn.Module):
    code = "S2"
    latency_samples = 0

    def __init__(self, taps: int = 17, num_knots: int = 17, residual_channels: int = 8):
        super().__init__()
        self.core = S0Structured(taps, num_knots)
        self.residual = FastResidualTCN(channels=residual_channels)

    def reset_state(self) -> None:
        self.core.reset_state()
        self.residual.reset_state()

    def forward_components(self, signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batched, scalar = _batch(signal)
        core = self.core(batched)
        residual = self.residual(torch.stack((batched, core), dim=1))
        output = core + residual
        if scalar:
            return output[0], core[0], residual[0]
        return output, core, residual

    def stream_components(self, signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batched, scalar = _batch(signal)
        core = self.core.stream(batched)
        residual = self.residual.stream(torch.stack((batched, core), dim=1))
        output = core + residual
        if scalar:
            return output[0], core[0], residual[0]
        return output, core, residual

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_components(signal)[0]

    def stream(self, signal: Tensor) -> Tensor:
        return self.stream_components(signal)[0]

    def energy_ratio(self, signal: Tensor) -> Tensor:
        output, _, residual = self.forward_components(signal)
        return residual_energy_ratio(residual, output)


class S3FastSlowResidual(nn.Module):
    code = "S3"
    latency_samples = 0

    def __init__(
        self,
        taps: int = 17,
        num_knots: int = 17,
        hidden_size: int = 8,
        decimation: int = 64,
        residual_channels: int = 8,
    ):
        super().__init__()
        self.core = S0Structured(taps, num_knots)
        self.slow = SlowStateController(hidden_size, decimation)
        self.residual = FastResidualTCN(channels=residual_channels)

    def reset_state(self) -> None:
        self.core.reset_state()
        self.slow.reset_state()
        self.residual.reset_state()

    def _components(self, signal: Tensor, *, streaming: bool):
        batched, scalar = _batch(signal)
        modulation = self.slow.stream(batched) if streaming else self.slow(batched)
        core = (
            self.core.stream_modulated(batched, modulation)
            if streaming
            else self.core.forward_modulated(batched, modulation)
        )
        residual = (
            self.residual.stream(torch.stack((batched, core), dim=1))
            if streaming
            else self.residual(torch.stack((batched, core), dim=1))
        )
        output = core + residual
        if scalar:
            return output[0], core[0], residual[0]
        return output, core, residual

    def forward_components(self, signal: Tensor):
        return self._components(signal, streaming=False)

    def stream_components(self, signal: Tensor):
        return self._components(signal, streaming=True)

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_components(signal)[0]

    def stream(self, signal: Tensor) -> Tensor:
        return self.stream_components(signal)[0]

    def energy_ratio(self, signal: Tensor) -> Tensor:
        output, _, residual = self.forward_components(signal)
        return residual_energy_ratio(residual, output)


class S4Antialiased(S3FastSlowResidual):
    code = "S4"

    def __init__(
        self,
        taps: int = 17,
        num_knots: int = 17,
        hidden_size: int = 8,
        decimation: int = 64,
        residual_channels: int = 8,
        oversampling_filter_taps: int = 33,
    ):
        nn.Module.__init__(self)
        shaper = LocalOversampledSpline2x(num_knots, oversampling_filter_taps)
        self.core = S0Structured(taps, num_knots, shaper=shaper)
        self.slow = SlowStateController(hidden_size, decimation)
        self.residual = FastResidualTCN(channels=residual_channels)
        self.latency_samples = shaper.latency_samples
        self.input_delay = CausalDelay(self.latency_samples)

    def reset_state(self) -> None:
        super().reset_state()
        self.input_delay.reset_state()

    def _components(self, signal: Tensor, *, streaming: bool):
        batched, scalar = _batch(signal)
        modulation = self.slow.stream(batched) if streaming else self.slow(batched)
        core = (
            self.core.stream_modulated(batched, modulation)
            if streaming
            else self.core.forward_modulated(batched, modulation)
        )
        delayed = (
            self.input_delay.stream(batched) if streaming else self.input_delay(batched)
        )
        features = torch.stack((delayed, core), dim=1)
        residual = (
            self.residual.stream(features) if streaming else self.residual(features)
        )
        output = core + residual
        if scalar:
            return output[0], core[0], residual[0]
        return output, core, residual
