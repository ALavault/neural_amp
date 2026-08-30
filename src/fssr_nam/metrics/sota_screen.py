"""Frozen synthetic mechanism screen for AMP-SOTA-PROTOTYPE-v1.1."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from scipy.signal import firwin, freqz
from torch import Tensor, nn

from fssr_nam.campaign.amp_sota_prototype_v1 import CAMPAIGN_VERSION
from fssr_nam.data.arch_fixtures import apply_arch_fixture, generate_mechanism_source
from fssr_nam.metrics.quality_aliasing import known_reference_alias_residual
from fssr_nam.models.approximants import (
    CausalControlReconstructor,
    QuinticHermiteSpline,
    SafeRationalActivation,
)
from fssr_nam.models.equiripple import design_equiripple_halfband


class _CubicHermiteSpline(nn.Module):
    """Uniform trainable C1 cubic Hermite control."""

    def __init__(self, knots: int, minimum: float, maximum: float) -> None:
        super().__init__()
        grid = torch.linspace(minimum, maximum, knots, dtype=torch.float64)
        self.register_buffer("knots", grid)
        self.values = nn.Parameter(grid.clone())
        self.slopes = nn.Parameter(torch.ones_like(grid))

    def forward(self, inputs: Tensor) -> Tensor:
        spacing = (self.knots[-1] - self.knots[0]) / (len(self.knots) - 1)
        coordinate = (inputs - self.knots[0]) / spacing
        index = coordinate.floor().long().clamp(0, len(self.knots) - 2)
        t = coordinate - index
        y0, y1 = self.values[index], self.values[index + 1]
        m0, m1 = self.slopes[index], self.slopes[index + 1]
        inside = (
            (2 * t**3 - 3 * t**2 + 1) * y0
            + (t**3 - 2 * t**2 + t) * spacing * m0
            + (-2 * t**3 + 3 * t**2) * y1
            + (t**3 - t**2) * spacing * m1
        )
        left = self.values[0] + self.slopes[0] * (inputs - self.knots[0])
        right = self.values[-1] + self.slopes[-1] * (inputs - self.knots[-1])
        return torch.where(
            inputs < self.knots[0],
            left,
            torch.where(inputs > self.knots[-1], right, inside),
        )


def _screen_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    screen = config.get("mechanism_screen", config)
    if not isinstance(screen, Mapping):
        raise ValueError("mechanism screen config must be a mapping")
    return screen


def _chebyshev_grid(samples: int, minimum: float, maximum: float) -> Tensor:
    index = torch.arange(samples, dtype=torch.float64)
    unit = torch.cos(math.pi * (2 * index + 1) / (2 * samples)).flip(0)
    return minimum + 0.5 * (unit + 1.0) * (maximum - minimum)


def _fit(model: nn.Module, inputs: Tensor, updates: int, learning_rate: float) -> bool:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    target = torch.tanh(inputs)
    denominator_gradient_nonzero = not isinstance(model, SafeRationalActivation)
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(inputs) - target).square())
        loss.backward()
        if isinstance(model, SafeRationalActivation):
            gradient = model.denominator.grad
            denominator_gradient_nonzero = denominator_gradient_nonzero or bool(
                gradient is not None and torch.any(gradient != 0).item()
            )
        optimizer.step()
    return denominator_gradient_nonzero


def _alias_attenuation(
    model: nn.Module,
    *,
    dft_samples: int,
    frames: int,
    k0_values: list[int],
    amplitudes: list[float],
) -> float:
    index = np.arange(dft_samples, dtype=np.float64)
    values: list[float] = []
    with torch.no_grad():
        for k0 in k0_values:
            for amplitude in amplitudes:
                frame = amplitude * np.sin(2.0 * np.pi * k0 * index / dft_samples)
                probe = np.tile(frame, frames)
                candidate = model(torch.from_numpy(probe)).cpu().numpy()
                reference = np.tanh(probe)
                residual = known_reference_alias_residual(
                    candidate,
                    reference,
                    k0=k0,
                    floor_db=-300.0,
                    dft_samples=dft_samples,
                )
                values.append(-float(residual["db"]))
    return float(np.median(values))


def _approximant_axis(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    minimum, maximum = (float(value) for value in config["interval"])
    train_samples = int(config.get("train_samples", 8193))
    evaluation_samples = int(config.get("evaluation_samples", 65_537))
    updates = int(config.get("updates", 2000))
    learning_rate = float(config.get("learning_rate", 0.002))
    knots = int(config.get("knots", 17))
    torch.manual_seed(seed)
    train = _chebyshev_grid(train_samples, minimum, maximum)
    generator = np.random.default_rng(seed)
    evaluation = torch.from_numpy(
        generator.uniform(minimum, maximum, evaluation_samples)
    ).to(torch.float64)
    models: dict[str, nn.Module] = {
        "cubic_hermite_c1": _CubicHermiteSpline(knots, minimum, maximum),
        "quintic_hermite_c2": QuinticHermiteSpline(knots, minimum, maximum).double(),
        "safe_rational_4_3": SafeRationalActivation().double(),
    }
    rows: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        denominator_gradient = _fit(model, train, updates, learning_rate)
        with torch.no_grad():
            output = model(evaluation)
            target = torch.tanh(evaluation)
            error = float(
                torch.sum((output - target).square()) / torch.sum(target.square())
            )
        finite = bool(torch.isfinite(output).all().item())
        if not finite or error <= 0.0 or not math.isfinite(error):
            raise RuntimeError(f"invalid approximant output: {name}")
        asr = _alias_attenuation(
            model,
            dft_samples=int(config.get("asr_dft_samples", 65_536)),
            frames=int(config.get("asr_frames", 6)),
            k0_values=list(config.get("asr_k0_values", [1705, 8191, 12287])),
            amplitudes=list(config.get("asr_amplitudes", [0.10, 0.25, 0.48])),
        )
        rows[name] = {
            "name": name,
            "finite_input_output": finite,
            "denominator_gradient_nonzero": denominator_gradient,
            "metrics": {"asr_db": asr, "approximation_error": error},
        }
    return {
        "other_axes_at_control": True,
        "cost_measurement_available": False,
        "cost_unavailable_action": "no_cost_based_promotion",
        "control": rows["cubic_hermite_c1"],
        "candidates": [
            rows["quintic_hermite_c2"],
            rows["safe_rational_4_3"],
        ],
    }


def _expand_zero_order(frames: Tensor, frame_size: int) -> Tensor:
    return frames.repeat_interleave(frame_size, dim=-1)


def _parasite_db(error: Tensor) -> float:
    spectrum = torch.fft.rfft(error.flatten())
    energy = spectrum.abs().square()
    split = max(1, 3 * len(energy) // 4)
    ratio = float(energy[split:].sum() / energy.sum().clamp_min(1.0e-30))
    return 10.0 * math.log10(max(ratio, 1.0e-30))


def _slow_axis(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    frame_size = int(config.get("frame_samples", config.get("frame_size", 64)))
    frame_count = int(config.get("frames", 64))
    fixtures = list(
        config.get(
            "fixtures",
            ["slow_sag", "attack_release", "blocking_distortion", "level_transition"],
        )
    )
    coefficient = float(config.get("exponential_coefficient", 0.25))
    errors: dict[str, list[Tensor]] = {
        "causal_zero_order_hold": [],
        "causal_exponential_hold": [],
        "causal_slope_limited_hold": [],
    }
    targets: list[Tensor] = []
    for fixture_index, fixture in enumerate(fixtures):
        samples = max(4096, (frame_count + 1) * frame_size)
        source = generate_mechanism_source(seed + fixture_index, samples)
        target = torch.from_numpy(
            apply_arch_fixture(fixture, source).astype(np.float64)
        )
        target = target[: (len(target) // frame_size) * frame_size]
        blocks = target.reshape(-1, frame_size)
        completed = blocks.mean(dim=-1)
        frames = completed[:-1].reshape(1, 1, -1)
        desired = blocks[1 : 1 + frames.shape[-1]].reshape(1, 1, -1)
        targets.append(desired.flatten())
        reconstructions = {
            "causal_zero_order_hold": _expand_zero_order(frames, frame_size),
        }
        for name, mode in (
            ("causal_exponential_hold", "exponential"),
            ("causal_slope_limited_hold", "slope_limited"),
        ):
            reconstructor = CausalControlReconstructor(1, mode, coefficient).double()
            reconstructor.reset_state()
            reconstructions[name] = reconstructor(frames, frame_size)
        for name, reconstruction in reconstructions.items():
            errors[name].append((reconstruction - desired).flatten())
    rows: dict[str, dict[str, Any]] = {}
    target_energy = float(torch.cat(targets).square().sum())
    if target_energy <= 0.0:
        raise RuntimeError("slow-control screen target has no energy")
    for name, chunks in errors.items():
        error = torch.cat(chunks)
        esr = float(error.square().sum()) / target_energy
        rows[name] = {
            "name": name,
            "causal": True,
            "metrics": {
                "dynamic_esr": max(esr, 1.0e-30),
                "parasite_db": _parasite_db(error),
                "lookahead_samples": 0,
            },
        }
    return {
        "other_axes_at_control": True,
        "control": rows["causal_zero_order_hold"],
        "candidates": [
            rows["causal_exponential_hold"],
            rows["causal_slope_limited_hold"],
        ],
    }


def _stopband_attenuation(coefficients: np.ndarray, bins: int) -> float:
    frequencies, response = freqz(coefficients, worN=bins, fs=96_000.0)
    stopband = np.abs(response[frequencies >= 30_000.0])
    return float(-20.0 * np.log10(max(float(np.max(stopband)), 1.0e-15)))


def _resampler_axis(config: Mapping[str, Any]) -> dict[str, Any]:
    bins = int(config.get("response_bins", 65_536))
    control = firwin(65, 0.5, window=("kaiser", 8.6), scale=True)
    candidate = design_equiripple_halfband(49).double().numpy()
    return {
        "other_axes_at_control": True,
        "cost_measurement_available": False,
        "cost_unavailable_action": "no_cost_based_promotion",
        "control": {
            "name": "kaiser_windowed_sinc",
            "metrics": {
                "stopband_attenuation_db": _stopband_attenuation(control, bins)
            },
        },
        "candidates": [
            {
                "name": "equiripple_halfband_polyphase",
                "metrics": {
                    "stopband_attenuation_db": _stopband_attenuation(candidate, bins)
                },
            }
        ],
    }


def run_synthetic_mechanism_screen(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the deterministic, physical-audio-free v1.1 mechanism screen."""
    screen = _screen_config(config)
    seed = int(screen.get("seed", 20_260_830))
    return {
        "campaign_version": CAMPAIGN_VERSION,
        "evidence_tier": "SYNTHETIC",
        "seed": seed,
        "physical_audio_samples_read": 0,
        "axes": {
            "approximant": _approximant_axis(screen["approximant"], seed),
            "slow_control": _slow_axis(screen["slow_control"], seed),
            "resampler": _resampler_axis(screen["resampler"]),
        },
    }


__all__ = ["run_synthetic_mechanism_screen"]
