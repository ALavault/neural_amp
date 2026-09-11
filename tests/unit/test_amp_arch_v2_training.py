from __future__ import annotations

import numpy as np
import pytest
import torch

from fssr_nam.training.arch_v2 import train_checkpointed_trajectory


def _episodes() -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.linspace(-0.7, 0.8, 256).repeat(2, 1)
    source[1] = torch.sin(torch.arange(256) * 0.071) * 0.5
    return source, torch.tanh(2.4 * source)


def test_checkpointed_training_is_continuous_finite_and_complete() -> None:
    source, target = _episodes()
    _, result, states = train_checkpointed_trajectory(
        family="micro_tcn_x2",
        system="unit_fixture",
        train_inputs=source,
        train_targets=target,
        validation_inputs=source,
        validation_targets=target,
        profile="slim",
        checkpoints=(1, 2),
        tail_samples=64,
        learning_rate=1.0e-3,
        auxiliary_weight=0.0,
        device=torch.device("cpu"),
        seed=17,
    )
    assert result.updates == 2
    assert [row["update"] for row in result.checkpoints] == [1, 2]
    assert set(states) == {1, 2}
    assert all(
        np.isfinite(list(row["validation"].values())).all()
        for row in result.checkpoints
    )


@pytest.mark.parametrize("checkpoints", [(), (2, 1), (1, 1)])
def test_checkpointed_training_rejects_noncanonical_checkpoints(
    checkpoints: tuple[int, ...],
) -> None:
    source, target = _episodes()
    with pytest.raises(ValueError, match="checkpoints"):
        train_checkpointed_trajectory(
            family="micro_tcn_x2",
            system="unit_fixture",
            train_inputs=source,
            train_targets=target,
            validation_inputs=source,
            validation_targets=target,
            profile="slim",
            checkpoints=checkpoints,
            tail_samples=64,
            learning_rate=1.0e-3,
            auxiliary_weight=0.0,
            device=torch.device("cpu"),
            seed=17,
        )
