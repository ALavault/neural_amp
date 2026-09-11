from __future__ import annotations

import pytest
import torch
from torch import nn

from fssr_nam.losses import WrightLoss
from fssr_nam.training.r2 import (
    M4PhysicalLoss,
    distillation_loss,
    loss_factory,
    required_checkpoints,
    select_fssr_core,
    training_updates,
)


class _FixedSpectralLoss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert output.ndim == target.ndim == 3
        return output.new_tensor(2.0)


def test_existing_loss_definitions_keep_their_weights() -> None:
    m4 = loss_factory("m4")
    assert isinstance(m4, M4PhysicalLoss)
    assert (m4.mse_weight, m4.mrstft_weight) == (1.0, 5.0e-4)
    assert (
        m4.spline_curvature_weight,
        m4.normalized_residual_energy_weight,
    ) == (1.0e-5, 1.0e-3)
    wright = loss_factory("wright")
    assert isinstance(wright, WrightLoss)
    assert (wright.preemphasis, wright.esr_weight, wright.dc_weight) == (
        0.85,
        0.75,
        0.25,
    )
    with pytest.raises(ValueError, match="R2 loss"):
        loss_factory("invented")


def test_m4_physical_loss_combines_auditable_components() -> None:
    objective = M4PhysicalLoss()
    objective.mrstft = _FixedSpectralLoss()
    target = torch.ones(2, 8)
    output = torch.zeros_like(target)
    components = objective.components(
        output,
        target,
        spline_curvature=output.new_tensor(3.0),
        residual=output.new_full(output.shape, 0.5),
    )
    expected = 1.0 + 5.0e-4 * 2.0 + 1.0e-5 * 3.0 + 1.0e-3 * 0.25
    torch.testing.assert_close(components.total, output.new_tensor(expected))


def test_topology_and_checkpoint_controls_are_boundary_exact() -> None:
    assert select_fssr_core(0.499999) == "mono"
    assert select_fssr_core(0.50) == "cascade"
    assert training_updates(loss_name="m4", promoted_loss="wright") == 5_000
    assert training_updates(loss_name="wright", promoted_loss="wright") == 15_000
    assert required_checkpoints(5_000) == (200, 1_000, 5_000)
    assert required_checkpoints(15_000) == (200, 1_000, 5_000, 15_000)
    with pytest.raises(ValueError, match="5k or 15k"):
        required_checkpoints(1_000)


def test_distillation_adds_quarter_preemphasized_esr_and_detaches_teacher() -> None:
    student = torch.tensor([[0.0, 0.2, -0.1, 0.4]], requires_grad=True)
    teacher = torch.tensor([[0.1, 0.1, -0.2, 0.3]], requires_grad=True)
    physical = student.new_tensor(0.7, requires_grad=True)
    loss = distillation_loss(physical, student, teacher)
    assert loss > physical
    loss.backward()
    assert student.grad is not None
    assert physical.grad is not None
    assert teacher.grad is None
