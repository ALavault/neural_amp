"""Statistical analyses for frozen scientific campaigns."""

from .quality_teacher import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    hierarchical_confirmation_bootstrap,
)

__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "hierarchical_confirmation_bootstrap",
]
