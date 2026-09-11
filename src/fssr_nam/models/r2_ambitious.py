"""Deliberately out-of-gamut R2 candidates for architecture exploration.

The official R2 screen remains limited to the preregistered ``aa-nam`` and
``aa-fssr`` families.  This module provides a separately named, larger
AA-FSSR hypothesis so that an attempt to exceed R1 is reproducible without
silently changing the frozen screen matrix.  It is not eligible for a gate
until a versioned protocol amendment explicitly admits it.
"""

from __future__ import annotations

from torch import nn

from .r2 import AAFSSR


class AAFSSRXL(AAFSSR):
    """High-capacity cascade with longer FIRs and finer Hermite grids.

    Relative to the R1 cascade controls (17-tap/17-knot/8-channel), this
    candidate uses 65-tap filters, 129-knot splines, a 16-unit slow state and
    a 16-channel RF2047 residual.  The larger core is intentional: resource
    gates, rather than an a-priori size cap, decide whether it is deployable.
    """

    candidate_id = "aa-fssr-xl"
    exploratory_only = True
    core_taps = 65
    spline_knots = 129

    def __init__(self, *, aa_mode: str) -> None:
        super().__init__(
            core_kind="cascade",
            aa_mode=aa_mode,
            taps=self.core_taps,
            num_knots=self.spline_knots,
            residual_channels=16,
            slow_hidden_size=16,
            slow_decimation=64,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_ambitious_candidate(*, aa_mode: str) -> nn.Module:
    """Build the exploratory candidate without changing frozen R2 factory IDs."""
    return AAFSSRXL(aa_mode=aa_mode)
