"""Training configuration and execution support."""

from .arch_v3 import (
    ArchV3TrainingResult,
    evaluate_arch_v3_model,
    projection_gain_loss,
    train_arch_v3_trajectory,
)
from .quality_teacher import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_CURRICULUM,
    TeacherLoss,
    TrainingSource,
    build_initialized_teacher,
    fit_lower_quartile_fir,
    train_teacher_trajectory,
)
from .r2 import M4PhysicalLoss, distillation_loss, loss_factory

__all__ = [
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_CURRICULUM",
    "ArchV3TrainingResult",
    "M4PhysicalLoss",
    "TeacherLoss",
    "TrainingSource",
    "build_initialized_teacher",
    "distillation_loss",
    "evaluate_arch_v3_model",
    "fit_lower_quartile_fir",
    "loss_factory",
    "projection_gain_loss",
    "train_arch_v3_trajectory",
    "train_teacher_trajectory",
]
