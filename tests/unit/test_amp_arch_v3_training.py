from __future__ import annotations

import numpy as np
import pytest
import torch

from fssr_nam.training.arch_v3 import (
    evaluate_arch_v3_model,
    projection_gain_loss,
    train_arch_v3_trajectory,
)


def _episodes(samples: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.arange(samples, dtype=torch.float32)
    source = torch.stack(
        (
            0.35 * torch.sin(0.071 * time),
            0.25 * torch.sin(0.043 * time) + 0.1 * torch.sin(0.13 * time),
        )
    )
    return source, torch.tanh(2.4 * source)


def test_projection_gain_loss_detects_under_gain_and_has_finite_gradients() -> None:
    target = torch.linspace(-0.8, 0.9, 257)
    output = (0.8 * target).requires_grad_()
    loss = projection_gain_loss(output, target)
    assert float(loss.detach()) == pytest.approx(0.04, abs=1.0e-6)
    loss.backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()
    assert float(projection_gain_loss(target, target)) == pytest.approx(0.0, abs=1.0e-7)


def test_v3_training_carries_state_and_reports_aggregate_checkpoints() -> None:
    source, target = _episodes()
    model, result, states = train_arch_v3_trajectory(
        family="gainhead_micro_tcn_x2",
        system="unit_fixture",
        train_inputs=source,
        train_targets=target,
        internal_dev_inputs=source.flip(0),
        internal_dev_targets=target.flip(0),
        profile="slim",
        initial_residual_scale=0.75,
        checkpoints=(1, 2),
        scored_start=64,
        chunk_samples=64,
        learning_rate=1.0e-3,
        projection_gain_weight=0.05,
        device=torch.device("cpu"),
        seed=19,
    )
    assert result.updates == 2
    assert [row["update"] for row in result.checkpoints] == [1, 2]
    assert set(states) == {1, 2}
    for row in result.checkpoints:
        assert set(row) == {"update", "training", "internal_dev"}
        assert np.isfinite(list(row["training"].values())).all()
        assert np.isfinite(list(row["internal_dev"].values())).all()
    metrics = evaluate_arch_v3_model(model, source, target, scored_start=64)
    assert np.isfinite(list(metrics.values())).all()


def test_v3_training_updates_have_equal_samples_across_episode_boundaries() -> None:
    source, target = _episodes(samples=250)
    _, result, _ = train_arch_v3_trajectory(
        family="gainhead_micro_tcn_x2",
        system="unit_fixture",
        train_inputs=source,
        train_targets=target,
        internal_dev_inputs=source,
        internal_dev_targets=target,
        profile="slim",
        initial_residual_scale=0.75,
        checkpoints=(1, 2, 3, 4),
        scored_start=64,
        chunk_samples=64,
        learning_rate=1.0e-3,
        projection_gain_weight=0.05,
        device=torch.device("cpu"),
        seed=23,
    )
    assert result.chunk_samples == 64
    assert result.total_training_samples == 256
    assert [row["last_chunk"]["samples"] for row in result.history] == [64] * 4
    assert [row["last_chunk"]["episode_segments"] for row in result.history] == [
        1,
        1,
        2,
        1,
    ]


def test_v3_training_is_exactly_deterministic_for_a_fixed_seed() -> None:
    source, target = _episodes(samples=130)

    def train() -> tuple[object, dict[int, dict[str, torch.Tensor]]]:
        _, result, states = train_arch_v3_trajectory(
            family="slow_state_micro_tcn_x2",
            system="unit_fixture",
            train_inputs=source,
            train_targets=target,
            internal_dev_inputs=source.flip(0),
            internal_dev_targets=target.flip(0),
            profile="slim",
            initial_residual_scale=0.75,
            checkpoints=(1, 2),
            scored_start=32,
            chunk_samples=64,
            learning_rate=1.0e-3,
            projection_gain_weight=0.05,
            device=torch.device("cpu"),
            seed=29,
        )
        return result, states

    first_result, first_states = train()
    second_result, second_states = train()
    assert first_result == second_result
    assert first_states.keys() == second_states.keys()
    for update in first_states:
        assert first_states[update].keys() == second_states[update].keys()
        for name in first_states[update]:
            torch.testing.assert_close(
                first_states[update][name],
                second_states[update][name],
                atol=0.0,
                rtol=0.0,
            )


@pytest.mark.parametrize("checkpoints", [(), (2, 1), (1, 1)])
def test_v3_training_rejects_noncanonical_checkpoints(
    checkpoints: tuple[int, ...],
) -> None:
    source, target = _episodes()
    with pytest.raises(ValueError, match="checkpoints"):
        train_arch_v3_trajectory(
            family="gainhead_micro_tcn_x2",
            system="unit_fixture",
            train_inputs=source,
            train_targets=target,
            internal_dev_inputs=source,
            internal_dev_targets=target,
            profile="slim",
            initial_residual_scale=0.75,
            checkpoints=checkpoints,
            scored_start=64,
            chunk_samples=64,
            learning_rate=1.0e-3,
            projection_gain_weight=0.05,
            device=torch.device("cpu"),
            seed=19,
        )
