from __future__ import annotations

import numpy as np
import pytest
import torch

from fssr_nam.training.arch_v1 import (
    delay_audio,
    evaluate_mechanism_model,
    train_mechanism_model,
    wet_feature_targets,
)


def _episodes() -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.linspace(-0.7, 0.8, 256).repeat(2, 1)
    source[1] = torch.sin(torch.arange(256) * 0.071) * 0.5
    return source, torch.tanh(2.4 * source)


def test_delay_audio_and_wet_feature_targets_are_exact_and_finite() -> None:
    source, target = _episodes()
    delayed = delay_audio(source, 3)
    torch.testing.assert_close(delayed[:, :3], torch.zeros_like(delayed[:, :3]))
    torch.testing.assert_close(delayed[:, 3:], source[:, :-3])
    features = wet_feature_targets(target, decimation=16)
    assert features.shape == (2, 6, 256)
    assert torch.isfinite(features).all()
    with pytest.raises(ValueError, match="shape"):
        delay_audio(source[0], 2)


@pytest.mark.parametrize(
    ("family", "auxiliary_weight"),
    [("micro_tcn_x2", 0.0), ("phys_s6_tcn_x2", 0.01)],
)
def test_mechanism_training_smoke_is_finite(
    family: str, auxiliary_weight: float
) -> None:
    source, target = _episodes()
    model, result = train_mechanism_model(
        family=family,
        system="unit_fixture",
        train_inputs=source,
        train_targets=target,
        validation_inputs=source,
        validation_targets=target,
        profile="slim",
        updates=2,
        tail_samples=64,
        learning_rate=1.0e-3,
        auxiliary_weight=auxiliary_weight,
        device=torch.device("cpu"),
        seed=17,
    )
    assert result.updates == 2
    assert result.validation["esr"] >= 0.0
    assert np.isfinite(list(result.validation.values())).all()
    assert all(np.isfinite(list(row.values())).all() for row in result.history)
    metrics = evaluate_mechanism_model(model, source, target, tail_samples=64)
    assert np.isfinite(list(metrics.values())).all()


def test_mechanism_training_rejects_invalid_controls() -> None:
    source, target = _episodes()
    with pytest.raises(ValueError, match="positive"):
        train_mechanism_model(
            family="micro_tcn_x2",
            system="unit_fixture",
            train_inputs=source,
            train_targets=target,
            validation_inputs=source,
            validation_targets=target,
            profile="slim",
            updates=0,
            tail_samples=64,
            learning_rate=1.0e-3,
            auxiliary_weight=0.0,
            device=torch.device("cpu"),
            seed=17,
        )
