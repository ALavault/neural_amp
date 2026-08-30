from __future__ import annotations

import torch

from fssr_nam.models.sota_v12 import (
    V12_CANDIDATE,
    V12_ZERO_MODULATION_CONTROL,
    build_sota_v12_model,
    trainable_parameter_count,
)


def _build_pair(seed: int = 37) -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(seed)
    candidate = build_sota_v12_model(V12_CANDIDATE).eval()
    torch.manual_seed(seed)
    control = build_sota_v12_model(V12_ZERO_MODULATION_CONTROL).eval()
    return candidate, control


def test_v12_control_has_the_exact_candidate_initialization() -> None:
    candidate, control = _build_pair()
    assert candidate.state_dict().keys() == control.state_dict().keys()
    for name, value in candidate.state_dict().items():
        torch.testing.assert_close(
            value, control.state_dict()[name], atol=0.0, rtol=0.0
        )
    assert trainable_parameter_count(control) < trainable_parameter_count(candidate)


def test_v12_pair_differs_only_when_slow_modulation_becomes_informative() -> None:
    candidate, control = _build_pair()
    source = torch.linspace(-0.7, 0.7, 257)[None]
    torch.testing.assert_close(candidate(source), control(source), atol=0.0, rtol=0.0)

    with torch.no_grad():
        for model in (candidate, control):
            model.branch.output_projection.weight.fill_(0.08)
        candidate.observer.modulation_projection.bias.fill_(0.25)
        control.observer.modulation_projection.bias.fill_(0.25)
    assert not torch.equal(candidate(source), control(source))


def test_v12_zero_modulation_control_has_exact_block_parity_and_reset() -> None:
    _, model = _build_pair()
    source = torch.randn(1, 513)
    expected = model(source)

    model.reset()
    chunks = [1, 7, 64, 3, 128, 310]
    position = 0
    streamed = []
    for samples in chunks:
        streamed.append(model.stream(source[:, position : position + samples]))
        position += samples
    actual = torch.cat(streamed, dim=-1)
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)

    model.reset()
    torch.testing.assert_close(model.stream(source), expected, atol=2.0e-6, rtol=2.0e-6)
