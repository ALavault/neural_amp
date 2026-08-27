import torch

from fssr_nam.models.r1 import R1Mono
from fssr_nam.training.r1 import (
    PhaseScheduler,
    ResidualPenaltyRamp,
    WrightLoss,
    residual_penalty_weight,
    wright_dc,
    wright_esr,
    wright_preemphasis,
)


def test_wright_loss_matches_core_audio_ml_formula_and_has_finite_gradient() -> None:
    output = torch.tensor([[0.1, 0.0, -0.2], [0.2, 0.3, 0.1]], requires_grad=True)
    target = torch.tensor([[0.0, 0.1, -0.1], [0.1, 0.2, 0.4]])
    emphasized_target = torch.tensor([[0.0, 0.1, -0.185], [0.1, 0.115, 0.23]])
    torch.testing.assert_close(wright_preemphasis(target), emphasized_target)
    emphasized_output = wright_preemphasis(output)
    expected_esr = (emphasized_target - emphasized_output).square().mean() / (
        emphasized_target.square().mean() + 1.0e-5
    )
    expected_dc = (target.mean(dim=-1) - output.mean(dim=-1)).square().mean() / (
        target.square().mean() + 1.0e-5
    )
    torch.testing.assert_close(wright_esr(output, target), expected_esr)
    torch.testing.assert_close(wright_dc(output, target), expected_dc)
    loss = WrightLoss()(output, target)
    torch.testing.assert_close(loss, 0.75 * expected_esr + 0.25 * expected_dc)
    loss.backward()
    assert output.grad is not None
    assert torch.isfinite(output.grad).all()
    assert WrightLoss()(target, target) == 0.0


def _phase_config() -> list[dict]:
    return [
        {
            "name": "filters_and_gains",
            "start_step": 1,
            "end_step": 500,
            "trainable": ["filters", "gains"],
        },
        {
            "name": "core_and_spline",
            "start_step": 501,
            "end_step": 1500,
            "trainable": ["core"],
        },
        {
            "name": "residual_on_frozen_core_error",
            "start_step": 1501,
            "end_step": 3500,
            "trainable": ["residual"],
        },
        {
            "name": "joint_fine_tuning",
            "start_step": 3501,
            "end_step": 5000,
            "trainable": ["all"],
        },
    ]


def test_phase_scheduler_applies_exact_freeze_masks_and_finite_gradients() -> None:
    model = R1Mono(taps=5, num_knots=9, receptive_field=31)
    schedule = PhaseScheduler.from_config(_phase_config())
    assert schedule.phase_at(500).name == "filters_and_gains"
    assert schedule.phase_at(501).name == "core_and_spline"

    schedule.apply(model, 1)
    groups = model.r1_parameter_groups()
    assert all(parameter.requires_grad for parameter in groups["filters"])
    assert all(parameter.requires_grad for parameter in groups["gains"])
    assert all(not parameter.requires_grad for parameter in groups["splines"])
    assert all(not parameter.requires_grad for parameter in groups["slow"])
    assert all(not parameter.requires_grad for parameter in groups["residual"])

    schedule.apply(model, 1501)
    assert all(not parameter.requires_grad for parameter in groups["core"])
    assert all(parameter.requires_grad for parameter in groups["residual"])
    signal = 0.1 * torch.randn(2, 61)
    model(signal).square().mean().backward()
    assert all(parameter.grad is None for parameter in groups["core"])
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in groups["residual"]
    )

    schedule.apply(model, 3501)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_residual_penalty_ramp_is_linear_inclusive_and_clamped() -> None:
    ramp = ResidualPenaltyRamp()
    assert ramp(0) == 1.0e-2
    assert ramp(1) == 1.0e-2
    assert ramp(5000) == 1.0e-3
    assert ramp(6000) == 1.0e-3
    expected = 1.0e-2 + (2500 - 1) / (5000 - 1) * (1.0e-3 - 1.0e-2)
    assert ramp(2500) == expected
    assert residual_penalty_weight(2500) == expected
