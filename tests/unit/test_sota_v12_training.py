from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from fssr_nam.models.sota_v12 import V12_CANDIDATE
from fssr_nam.training.sota_v12 import (
    SotaV12ResourceLimitError,
    SotaV12StabilityError,
    evaluate_sota_v12_model,
    train_sota_v12_trajectory,
)


class TinyCausalModel(nn.Module):
    latency_samples = 2

    def __init__(self, *, invalid: bool = False) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.8))
        self.invalid = invalid
        self.register_buffer("_state", torch.empty(0), persistent=False)

    @property
    def dry_gain(self) -> torch.Tensor:
        return self.gain

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.gain.abs()

    def reset_state(self) -> None:
        self._state = self._state.new_empty(0)

    def detach_stream_state(self) -> None:
        self._state = self._state.detach()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        output = functional.pad(signal, (self.latency_samples, 0))[
            ..., : -self.latency_samples
        ]
        return output * (torch.nan if self.invalid else self.gain)

    def stream(self, signal: torch.Tensor) -> torch.Tensor:
        if self._state.numel() == 0:
            self._state = signal.new_zeros((len(signal), self.latency_samples))
        joined = torch.cat((self._state, signal), dim=-1)
        output = joined[..., : signal.shape[-1]]
        self._state = joined[..., -self.latency_samples :]
        return output * (torch.nan if self.invalid else self.gain)


def _episodes(samples: int = 5_000) -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.arange(samples, dtype=torch.float32)
    source = torch.stack(
        (0.25 * torch.sin(0.031 * time), 0.31 * torch.sin(0.019 * time))
    )
    return source, torch.tanh(2.0 * source)


def test_v12_evaluation_applies_preroll_after_declared_latency() -> None:
    source, _ = _episodes()
    model = TinyCausalModel()
    with torch.no_grad():
        model.gain.fill_(1.0)
    result = evaluate_sota_v12_model(
        model,
        source,
        source,
        common_preroll_samples_after_alignment=64,
        include_spectral=False,
    )
    assert result["aggregate"]["esr"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["aggregate"]["scored_samples"] == 2 * (5_000 - 64 - 2)
    assert all(row["scored_samples"] == 4_934 for row in result["sources"])


def test_v12_training_is_finite_checkpointed_and_source_preserving() -> None:
    source, target = _episodes()
    _, result, states = train_sota_v12_trajectory(
        family=V12_CANDIDATE,
        system="unit_system",
        train_inputs=source,
        train_targets=target,
        internal_dev_inputs=source.flip(0),
        internal_dev_targets=target.flip(0),
        snapshot_updates=(1, 2),
        evaluation_updates=(1, 2),
        common_preroll_samples_after_alignment=64,
        chunk_samples=128,
        learning_rate=1.0e-3,
        projection_gain_weight=0.05,
        gradient_clip_norm=1.0,
        maximum_elapsed_seconds=60.0,
        device=torch.device("cpu"),
        seed=0,
        model_builder=lambda _family: TinyCausalModel(),
    )
    assert result.scored_start == 66
    assert result.scored_samples_per_episode == 4_934
    assert result.total_training_samples == 256
    assert set(states) == {1, 2}
    assert [row["update"] for row in result.checkpoints] == [1, 2]
    first, final = result.checkpoints
    assert "log_mel" not in first["internal_dev"]["source_median"]
    assert set(final["internal_dev"]["source_median"]) >= {
        "esr",
        "mae",
        "log_mel",
        "mrstft",
    }
    assert len(final["internal_dev"]["sources"]) == 2
    assert np.isfinite(list(final["internal_dev"]["source_median"].values())).all()


def test_v12_nonfinite_output_is_a_scientific_stability_failure() -> None:
    source, target = _episodes()
    with pytest.raises(SotaV12StabilityError, match="pre-roll"):
        train_sota_v12_trajectory(
            family=V12_CANDIDATE,
            system="unit_system",
            train_inputs=source,
            train_targets=target,
            internal_dev_inputs=source,
            internal_dev_targets=target,
            snapshot_updates=(1,),
            evaluation_updates=(1,),
            common_preroll_samples_after_alignment=64,
            chunk_samples=128,
            learning_rate=1.0e-3,
            projection_gain_weight=0.05,
            gradient_clip_norm=1.0,
            maximum_elapsed_seconds=60.0,
            device=torch.device("cpu"),
            seed=0,
            model_builder=lambda _family: TinyCausalModel(invalid=True),
        )


def test_v12_wall_clock_limit_stops_before_first_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _episodes()
    readings = iter((0.0, 1.0))
    monkeypatch.setattr(
        "fssr_nam.training.sota_v12.time.perf_counter", lambda: next(readings)
    )
    with pytest.raises(SotaV12ResourceLimitError, match="wall-clock"):
        train_sota_v12_trajectory(
            family=V12_CANDIDATE,
            system="unit_system",
            train_inputs=source,
            train_targets=target,
            internal_dev_inputs=source,
            internal_dev_targets=target,
            snapshot_updates=(1,),
            evaluation_updates=(1,),
            common_preroll_samples_after_alignment=64,
            chunk_samples=128,
            learning_rate=1.0e-3,
            projection_gain_weight=0.05,
            gradient_clip_norm=1.0,
            maximum_elapsed_seconds=0.5,
            device=torch.device("cpu"),
            seed=0,
            model_builder=lambda _family: TinyCausalModel(),
        )
