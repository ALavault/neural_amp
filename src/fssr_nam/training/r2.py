"""Frozen model, loss, checkpoint, and distillation controls for FSSR-R2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss
from torch import Tensor, nn

from fssr_nam.losses import WrightLoss, esr_loss, preemphasize
from fssr_nam.models import AAFSSR, AANAM
from fssr_nam.models.residual import normalized_residual_penalty
from fssr_nam.training.m4 import model_factory as m4_model_factory

R2_CHECKPOINTS = (200, 1_000, 5_000, 15_000)
R2_LOSSES = ("m4", "wright")


@dataclass(frozen=True)
class M4LossComponents:
    """Auditable components of the existing M4 objective."""

    total: Tensor
    mse: Tensor
    mrstft: Tensor
    spline_curvature: Tensor
    normalized_residual_energy: Tensor


class M4PhysicalLoss(nn.Module):
    """The existing M4 loss with its original, unchanged weights."""

    mse_weight = 1.0
    mrstft_weight = 5.0e-4
    spline_curvature_weight = 1.0e-5
    normalized_residual_energy_weight = 1.0e-3

    def __init__(self) -> None:
        super().__init__()
        self.mrstft = MultiResolutionSTFTLoss()

    def components(
        self,
        output: Tensor,
        target: Tensor,
        *,
        spline_curvature: Tensor | None = None,
        residual: Tensor | None = None,
    ) -> M4LossComponents:
        if output.shape != target.shape:
            raise ValueError("output and target shapes differ")
        mse = torch.nn.functional.mse_loss(output, target)
        mrstft = self.mrstft(output[:, None], target[:, None])
        curvature = (
            output.new_zeros(()) if spline_curvature is None else spline_curvature
        )
        residual_energy = (
            output.new_zeros(())
            if residual is None
            else normalized_residual_penalty(residual, target)
        )
        scalars = (mse, mrstft, curvature, residual_energy)
        if any(value.ndim != 0 or not torch.isfinite(value) for value in scalars):
            raise ValueError("M4 loss components must be finite scalars")
        total = (
            self.mse_weight * mse
            + self.mrstft_weight * mrstft
            + self.spline_curvature_weight * curvature
            + self.normalized_residual_energy_weight * residual_energy
        )
        return M4LossComponents(total, mse, mrstft, curvature, residual_energy)

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        *,
        spline_curvature: Tensor | None = None,
        residual: Tensor | None = None,
    ) -> Tensor:
        return self.components(
            output,
            target,
            spline_curvature=spline_curvature,
            residual=residual,
        ).total


def loss_factory(loss_name: str) -> nn.Module:
    """Construct exactly one of the two preregistered physical losses."""
    if loss_name == "m4":
        return M4PhysicalLoss()
    if loss_name == "wright":
        return WrightLoss(
            preemphasis=0.85,
            esr_weight=0.75,
            dc_weight=0.25,
            epsilon=1.0e-5,
        )
    raise ValueError(f"R2 loss must be one of {R2_LOSSES}")


def select_fssr_core(two_clipper_relative_esr_improvement: float) -> str:
    """Apply the preregistered two-clipper topology fixture rule."""
    if not torch.isfinite(torch.tensor(two_clipper_relative_esr_improvement)):
        raise ValueError("two-clipper improvement must be finite")
    return "cascade" if two_clipper_relative_esr_improvement >= 0.50 else "mono"


def model_factory(
    family: Literal["a2", "aa-nam", "aa-fssr"],
    *,
    aa_mode: str,
    root: Path,
    core_kind: str | None = None,
) -> nn.Module:
    """Build a pinned R2 training topology without tuning hidden defaults."""
    if family == "a2":
        if aa_mode != "off" or core_kind is not None:
            raise ValueError("A2 baseline must use aa_mode=off and no FSSR core")
        return m4_model_factory("B0", root=root, model_config={})
    if family == "aa-nam":
        if core_kind is not None:
            raise ValueError("AA-NAM cannot declare an FSSR core")
        return AANAM(aa_mode=aa_mode, root=root)
    if family == "aa-fssr":
        if core_kind not in {"mono", "cascade"}:
            raise ValueError("AA-FSSR requires a frozen mono or cascade core")
        return AAFSSR(core_kind=core_kind, aa_mode=aa_mode)
    raise ValueError("unsupported R2 family")


def training_updates(*, loss_name: str, promoted_loss: str) -> int:
    """Stop the non-promoted loss at 5k and the promoted loss at 15k."""
    if loss_name not in R2_LOSSES or promoted_loss not in R2_LOSSES:
        raise ValueError(f"R2 loss must be one of {R2_LOSSES}")
    return 15_000 if loss_name == promoted_loss else 5_000


def required_checkpoints(updates: int) -> tuple[int, ...]:
    """Return every common checkpoint that must exist for a trajectory."""
    if updates not in {5_000, 15_000}:
        raise ValueError("R2 trajectories must stop at 5k or 15k updates")
    return tuple(step for step in R2_CHECKPOINTS if step <= updates)


def distillation_loss(
    physical_loss: Tensor,
    student_output: Tensor,
    teacher_downsampled: Tensor,
    *,
    preemphasis: float = 0.85,
) -> Tensor:
    """Add the frozen 0.25 teacher ESR term to the selected physical loss."""
    if physical_loss.ndim != 0 or not torch.isfinite(physical_loss):
        raise ValueError("physical loss must be a finite scalar")
    if student_output.shape != teacher_downsampled.shape:
        raise ValueError("student and downsampled teacher shapes differ")
    teacher = teacher_downsampled.detach()
    teacher_esr = esr_loss(
        preemphasize(student_output, preemphasis),
        preemphasize(teacher, preemphasis),
    )
    return physical_loss + 0.25 * teacher_esr
