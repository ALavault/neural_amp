from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import fssr_nam.training.quality_teacher as quality_teacher
from fssr_nam.training.quality_teacher import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_CURRICULUM,
    ContiguousMicrochunkCursor,
    CurriculumPhase,
    TeacherLoss,
    TeacherLossComponents,
    TrainingSource,
    _set_phase_trainability,
    delay_audio,
    fit_lower_quartile_fir,
    train_teacher_trajectory,
)


class RecordingSpectralLoss(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.shapes: list[tuple[torch.Size, torch.Size]] = []

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.shapes.append((output.shape, target.shape))
        return output.new_tensor(self.value)


class TinySquaredLoss(nn.Module):
    def components(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> TeacherLossComponents:
        squared = (output - target).square().mean()
        zero = squared * 0.0
        return TeacherLossComponents(squared, squared, zero, zero, zero)


class TinyObserver(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.25))


class TinyQualityTeacher(nn.Module):
    family = "tiny_quality_teacher"
    latency_samples = 0

    def __init__(self, *, slow_modulation_enabled: bool = True) -> None:
        super().__init__()
        self.fast_gain = nn.Parameter(torch.tensor(0.50))
        self.observer = TinyObserver()
        self.slow_modulation_enabled = slow_modulation_enabled
        self.training_stream_calls = 0
        self.training_reset_calls = 0
        self.detach_calls = 0

    def reset_state(self) -> None:
        if self.training:
            self.training_reset_calls += 1

    def detach_stream_state(self) -> None:
        self.detach_calls += 1

    def stream(self, signal: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.training_stream_calls += 1
        gain = self.fast_gain
        if self.slow_modulation_enabled:
            gain = gain + self.observer.gain
        return signal * gain


def _tiny_source(source_id: str = "train") -> TrainingSource:
    dry = torch.tensor((0.25, -0.50, 0.75, -0.25), dtype=torch.float32)
    return TrainingSource(source_id, dry, 1.5 * dry)


def test_teacher_loss_uses_frozen_weighting_and_preemphasis() -> None:
    spectral = RecordingSpectralLoss(2.5)
    loss = TeacherLoss(spectral_loss=spectral)
    output = torch.tensor(((0.2, -0.1, 0.4, 0.3),), dtype=torch.float32)
    target = torch.tensor(((0.1, 0.2, 0.5, -0.2),), dtype=torch.float32)

    components = loss.components(output, target)

    emphasized_output = output - 0.95 * torch.nn.functional.pad(
        output[..., :-1], (1, 0)
    )
    emphasized_target = target - 0.95 * torch.nn.functional.pad(
        target[..., :-1], (1, 0)
    )
    expected_esr = (emphasized_target - emphasized_output).square().mean() / (
        emphasized_target.square().mean() + 1.0e-5
    )
    projection = (output.mul(target).sum() / (target.square().sum() + 1.0e-8) - 1) ** 2
    expected = (
        10.0 * torch.nn.functional.l1_loss(output, target)
        + 2.5
        + expected_esr
        + 0.05 * projection
    )

    assert loss.preemphasis == 0.95
    assert spectral.shapes == [(torch.Size((1, 1, 4)), torch.Size((1, 1, 4)))]
    torch.testing.assert_close(components.mrstft, torch.tensor(2.5))
    torch.testing.assert_close(components.preemphasized_esr, expected_esr)
    torch.testing.assert_close(components.projection_gain, projection)
    torch.testing.assert_close(components.total, expected)
    torch.testing.assert_close(loss(output, target), expected)


def test_lower_quartile_fir_fit_is_deterministic_and_uses_only_quiet_passage() -> None:
    generator = np.random.default_rng(123)
    causal_coefficients = np.asarray((0.7, -0.2, 0.1), dtype=np.float64)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for scale in (0.1, 0.3, 0.6, 1.0):
        dry = scale * generator.standard_normal(256)
        wet = np.convolve(dry, causal_coefficients, mode="full")[: len(dry)]
        pairs.append((dry.astype(np.float32), wet.astype(np.float32)))

    first = fit_lower_quartile_fir(pairs, taps=3, passage_samples=256, ridge=1.0e-8)
    second = fit_lower_quartile_fir(pairs, taps=3, passage_samples=256, ridge=1.0e-8)

    assert first.passages_total == 4
    assert first.passages_selected == 1
    assert first.passage_samples == 256
    assert first.lower_quartile_rms == pytest.approx(second.lower_quartile_rms)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_allclose(
        first.coefficients,
        causal_coefficients[::-1],
        atol=4.0e-4,
        rtol=0.0,
    )


def test_cursor_uses_one_second_chunks_padding_and_source_resets() -> None:
    first_dry = torch.arange(48_002, dtype=torch.float32)
    first_target = first_dry + 100.0
    second_dry = torch.tensor((3.0, 4.0, 5.0), dtype=torch.float32)
    second_target = second_dry + 200.0
    sources = (
        TrainingSource("first", first_dry, first_target),
        TrainingSource("second", second_dry, second_target),
    )
    cursor = ContiguousMicrochunkCursor(
        sources, chunk_samples=48_000, latency_samples=2
    )

    chunks = (cursor.next(), cursor.next(), cursor.next())

    assert [chunk.source_id for chunk in chunks] == ["first", "first", "second"]
    assert [chunk.valid_samples for chunk in chunks] == [48_000, 2, 3]
    assert [chunk.reset_before for chunk in chunks] == [True, False, True]
    assert torch.count_nonzero(chunks[1].dry[2:]) == 0
    assert torch.count_nonzero(chunks[1].delayed_target[2:]) == 0
    assert torch.count_nonzero(chunks[2].dry[3:]) == 0
    visited_dry = torch.cat(tuple(chunk.dry[: chunk.valid_samples] for chunk in chunks))
    visited_target = torch.cat(
        tuple(chunk.delayed_target[: chunk.valid_samples] for chunk in chunks)
    )
    torch.testing.assert_close(visited_dry, torch.cat((first_dry, second_dry)))
    torch.testing.assert_close(
        visited_target,
        torch.cat(
            (
                delay_audio(first_target[None], 2)[0],
                delay_audio(second_target[None], 2)[0],
            )
        ),
    )
    wrapped = cursor.next()
    assert wrapped.source_id == "first"
    assert wrapped.reset_before is True


def test_curriculum_and_phase_freezing_match_frozen_protocol() -> None:
    assert DEFAULT_CURRICULUM == (
        CurriculumPhase("fast_fir", 2_500, True, False),
        CurriculumPhase("s4_film", 2_500, False, True),
        CurriculumPhase("joint", 2_500, True, True),
    )
    assert DEFAULT_CHECKPOINTS == tuple(range(500, 7_501, 500))
    model = TinyQualityTeacher()

    fast = _set_phase_trainability(model, DEFAULT_CURRICULUM[0])
    assert fast == (model.fast_gain,)
    assert model.fast_gain.requires_grad
    assert not model.observer.gain.requires_grad

    slow = _set_phase_trainability(model, DEFAULT_CURRICULUM[1])
    assert slow == (model.observer.gain,)
    assert not model.fast_gain.requires_grad
    assert model.observer.gain.requires_grad

    joint = _set_phase_trainability(model, DEFAULT_CURRICULUM[2])
    assert joint == (model.fast_gain, model.observer.gain)
    assert all(parameter.requires_grad for parameter in joint)

    control = TinyQualityTeacher(slow_modulation_enabled=False)
    assert _set_phase_trainability(control, DEFAULT_CURRICULUM[1]) == ()
    assert not control.fast_gain.requires_grad
    assert not control.observer.gain.requires_grad


def test_reduced_training_uses_three_microchunks_per_update() -> None:
    model = TinyQualityTeacher()
    source = _tiny_source()
    fast_before = model.fast_gain.detach().clone()
    slow_before = model.observer.gain.detach().clone()

    result, states = train_teacher_trajectory(
        model,
        train_sources=(source,),
        validation_sources=(source,),
        seed=7,
        loss_module=TinySquaredLoss(),
        curriculum=(CurriculumPhase("fast_fir", 500, True, False),),
        checkpoints=(500,),
        chunk_samples=4,
        microchunks_per_update=3,
        learning_rate=1.0e-3,
    )

    assert result.updates == 500
    assert result.selected_update == 500
    assert len(result.history) == 500
    assert set(states) == {500}
    assert model.training_stream_calls == 1_500
    assert model.training_reset_calls == 1_500
    assert model.detach_calls == 1_500
    assert {row["samples"] for row in result.history} == {12}
    assert {row["phase"] for row in result.history} == {"fast_fir"}
    assert not torch.equal(model.fast_gain.detach(), fast_before)
    torch.testing.assert_close(model.observer.gain.detach(), slow_before)


def test_fast_only_s4_phase_is_noop_and_selection_uses_validation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_totals = iter((2.0, 1.0))

    def fake_validation(*_args: object, **_kwargs: object) -> dict[str, float]:
        total = next(validation_totals)
        return {
            "total": total,
            "l1": 0.0,
            "mrstft": 0.0,
            "preemphasized_esr": 0.0,
            "projection_gain": 0.0,
        }

    monkeypatch.setattr(quality_teacher, "evaluate_validation_loss", fake_validation)
    model = TinyQualityTeacher(slow_modulation_enabled=False)
    source = _tiny_source()
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    result, states = train_teacher_trajectory(
        model,
        train_sources=(source,),
        validation_sources=(source,),
        seed=11,
        loss_module=TinySquaredLoss(),
        curriculum=(CurriculumPhase("s4_film", 1_000, False, True),),
        checkpoints=(500, 1_000),
        chunk_samples=4,
        microchunks_per_update=3,
    )

    assert result.selected_update == 1_000
    assert [row["validation"]["total"] for row in result.checkpoints] == [2.0, 1.0]
    assert set(states) == {500, 1_000}
    assert model.training_stream_calls == 3_000
    assert model.detach_calls == 3_000
    assert {row["gradient_norm"] for row in result.history} == {0.0}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0.0, rtol=0.0)


def test_training_rejects_a_gap_in_every_500_update_checkpoints() -> None:
    model = TinyQualityTeacher(slow_modulation_enabled=False)
    source = _tiny_source()

    with pytest.raises(ValueError, match="checkpoints"):
        train_teacher_trajectory(
            model,
            train_sources=(source,),
            validation_sources=(source,),
            seed=0,
            loss_module=TinySquaredLoss(),
            curriculum=(CurriculumPhase("s4_film", 1_000, False, True),),
            checkpoints=(1_000,),
            chunk_samples=4,
        )
