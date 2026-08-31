from __future__ import annotations

import inspect

import torch

from fssr_nam.models.quality_teacher import (
    QUALITY_TEACHER_FAMILY,
    QUALITY_TEACHER_FAST_CONTROL,
    WAVENET_DILATIONS,
    DenseGatedWaveNetResidual,
    QualityTeacherAmplifier,
    QualityTeacherOutput,
)
from fssr_nam.models.quality_teacher_comparators import DenseWaveNet16x18


def _small_teacher(
    family: str = QUALITY_TEACHER_FAMILY,
) -> QualityTeacherAmplifier:
    torch.manual_seed(1_729)
    return QualityTeacherAmplifier(
        family=family,
        fir_taps=5,
        wavenet_channels=2,
        wavenet_dilations=(1, 2, 4),
        observer_channels=3,
        observer_state_dim=2,
        observer_blocks=2,
        observer_evaluation_block_samples=8,
    )


def _activate_fast_and_slow_paths(model: QualityTeacherAmplifier) -> None:
    with torch.no_grad():
        model.branch.output_projection.weight.fill_(0.025)
        model.branch.output_projection.bias.fill_(0.005)
        model.observer.film_projection.weight.fill_(0.01)
        model.observer.film_projection.bias.fill_(0.02)


def _stream_components(
    model: QualityTeacherAmplifier,
    signal: torch.Tensor,
    chunk_sizes: tuple[int, ...],
) -> QualityTeacherOutput:
    parts: list[QualityTeacherOutput] = []
    offset = 0
    for size in chunk_sizes:
        parts.append(model.stream_components(signal[..., offset : offset + size]))
        offset += size
    assert offset == signal.shape[-1]
    return QualityTeacherOutput(
        **{
            name: torch.cat([getattr(part, name) for part in parts], dim=-1)
            for name in ("audio", "fast", "slow", "fir")
        }
    )


def _state_tensors(model: torch.nn.Module) -> list[torch.Tensor]:
    states: list[torch.Tensor] = []
    for module in model.modules():
        for name in ("_state", "_stream_state"):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                states.append(value)
    return states


def test_quality_teacher_default_contract_and_receptive_field_are_frozen() -> None:
    parameters = inspect.signature(QualityTeacherAmplifier).parameters
    assert parameters["fir_taps"].default == 257
    assert parameters["wavenet_channels"].default == 64
    assert parameters["observer_channels"].default == 64
    assert parameters["observer_state_dim"].default == 64
    assert parameters["observer_blocks"].default == 8
    assert WAVENET_DILATIONS == tuple(2**index for _ in range(2) for index in range(12))

    branch = DenseGatedWaveNetResidual(channels=1)
    assert len(branch.blocks) == 24
    assert branch.receptive_field_internal_samples == 16_381
    assert branch.receptive_field_base_samples == 8_191
    assert QualityTeacherAmplifier.sample_rate_hz == 48_000
    assert QualityTeacherAmplifier.internal_sample_rate_hz == 96_000
    assert QualityTeacherAmplifier.latency_samples == 32
    assert QualityTeacherAmplifier.precision == "float32"
    assert QualityTeacherAmplifier.aa_mode == "full_island_x2"


def test_quality_teacher_components_are_finite_and_exactly_delayed() -> None:
    model = _small_teacher()
    signal = torch.zeros(2, 96, dtype=torch.float32)
    signal[0, 7] = 1.0
    signal[1, 11] = -0.5

    result = model.forward_components(signal)
    expected = torch.nn.functional.pad(signal, (model.latency_samples, 0))[
        ..., : -model.latency_samples
    ]

    for component in (result.audio, result.fast, result.slow, result.fir):
        assert component.shape == signal.shape
        assert torch.isfinite(component).all()
    torch.testing.assert_close(result.audio, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(result.fir, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        result.fast, torch.zeros_like(signal), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        result.slow, torch.zeros_like(signal), atol=0.0, rtol=0.0
    )
    assert torch.nonzero(result.audio[0], as_tuple=True)[0].tolist() == [39]
    assert torch.nonzero(result.audio[1], as_tuple=True)[0].tolist() == [43]


def test_quality_teacher_audio_and_diagnostics_are_causal() -> None:
    model = _small_teacher()
    _activate_fast_and_slow_paths(model)
    torch.manual_seed(1_730)
    shared_prefix = torch.randn(1, 37)
    first = torch.cat((shared_prefix, torch.zeros(1, 42)), dim=-1)
    second = torch.cat((shared_prefix, torch.randn(1, 42)), dim=-1)

    first_result = model.forward_components(first)
    second_result = model.forward_components(second)

    for name in ("audio", "fast", "slow", "fir"):
        torch.testing.assert_close(
            getattr(first_result, name)[..., :37],
            getattr(second_result, name)[..., :37],
            atol=2.0e-6,
            rtol=2.0e-6,
        )


def test_quality_teacher_irregular_stream_matches_offline_and_reset_replays() -> None:
    model = _small_teacher()
    _activate_fast_and_slow_paths(model)
    torch.manual_seed(1_731)
    signal = torch.randn(2, 73)
    expected = model.forward_components(signal)

    model.reset_state()
    actual = _stream_components(model, signal, (1, 7, 3, 29, 2, 31))
    for name in ("audio", "fast", "slow", "fir"):
        torch.testing.assert_close(
            getattr(actual, name),
            getattr(expected, name),
            atol=3.0e-6,
            rtol=3.0e-6,
        )

    model.reset()
    replay = model.stream_components(signal)
    for name in ("audio", "fast", "slow", "fir"):
        torch.testing.assert_close(
            getattr(replay, name),
            getattr(expected, name),
            atol=3.0e-6,
            rtol=3.0e-6,
        )
    model.reset_state()
    torch.testing.assert_close(model.stream(signal), expected.audio)
    torch.testing.assert_close(model(signal), expected.audio)


def test_quality_teacher_detach_supports_gradients_on_successive_chunks() -> None:
    model = _small_teacher()
    _activate_fast_and_slow_paths(model)
    torch.manual_seed(1_732)

    first_signal = torch.randn(2, 41, requires_grad=True)
    first = model.stream_components(first_signal)
    first_loss = first.audio.square().mean() + 0.05 * first.slow.mean()
    first_loss.backward()
    assert first_signal.grad is not None
    assert torch.isfinite(first_signal.grad).all()
    states_before_detach = _state_tensors(model)
    assert states_before_detach
    assert any(state.grad_fn is not None for state in states_before_detach)

    model.detach_stream_state()
    assert all(
        state.grad_fn is None and not state.requires_grad
        for state in _state_tensors(model)
    )

    model.zero_grad(set_to_none=True)
    second_signal = torch.randn(2, 41, requires_grad=True)
    second = model.stream_components(second_signal)
    second_loss = second.audio.square().mean() + 0.05 * second.slow.mean()
    second_loss.backward()

    assert second_signal.grad is not None
    assert torch.isfinite(second_signal.grad).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.branch.output_projection.weight.grad is not None
    assert model.observer.film_projection.weight.grad is not None


def test_fast_control_equals_candidate_with_zero_modulation() -> None:
    candidate = _small_teacher()
    control = _small_teacher(QUALITY_TEACHER_FAST_CONTROL)
    _activate_fast_and_slow_paths(candidate)
    _activate_fast_and_slow_paths(control)
    control.load_state_dict(candidate.state_dict())
    candidate.set_slow_modulation_enabled(False)
    assert not any(
        parameter.requires_grad for parameter in control.observer.parameters()
    )

    torch.manual_seed(1_733)
    signal = torch.randn(2, 61)
    candidate_result = candidate.forward_components(signal)
    control_result = control.forward_components(signal)
    for name in ("audio", "fast", "slow", "fir"):
        torch.testing.assert_close(
            getattr(candidate_result, name),
            getattr(control_result, name),
            atol=0.0,
            rtol=0.0,
        )


def test_dense_wavenet_comparator_parameter_count_and_stream_parity() -> None:
    torch.manual_seed(1_734)
    model = DenseWaveNet16x18()
    assert sum(parameter.numel() for parameter in model.parameters()) == 21_913
    assert model.channels == 16
    assert len(model.dilations) == 18
    assert model.receptive_field == 2_045

    signal = torch.randn(1, 37)
    expected = model(signal)
    assert expected.shape == signal.shape
    assert torch.isfinite(expected).all()

    model.reset_state()
    actual = torch.cat(
        (
            model.stream(signal[..., :1]),
            model.stream(signal[..., 1:8]),
            model.stream(signal[..., 8:11]),
            model.stream(signal[..., 11:36]),
            model.stream(signal[..., 36:]),
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)

    model.reset_state()
    torch.testing.assert_close(model.stream(signal), expected, atol=2.0e-6, rtol=2.0e-6)
