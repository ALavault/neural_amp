"""Exact frozen NablaFX black-box loss without modifying the lock file."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

AURALOSS_COMMIT = "1576b0cd6e927abc002b23cf3bfc455b660f663c"
_PACKAGE_NAME = "_fssr_auraloss_040"


def _load_exact_mrstft_class() -> type[nn.Module]:
    """Load the pinned auraloss 0.4.0 source tree as an isolated package."""
    if _PACKAGE_NAME not in sys.modules:
        root = Path(__file__).resolve().parents[3]
        package = root / "third_party/auraloss/auraloss"
        init_path = package / "__init__.py"
        if not init_path.is_file():
            raise RuntimeError("pinned third_party/auraloss source is missing")
        spec = importlib.util.spec_from_file_location(
            _PACKAGE_NAME,
            init_path,
            submodule_search_locations=[str(package)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load pinned auraloss package")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE_NAME] = module
        spec.loader.exec_module(module)
    frequency = importlib.import_module(f"{_PACKAGE_NAME}.freq")
    return frequency.MultiResolutionSTFTLoss


@dataclass(frozen=True)
class NablafxLossComponents:
    total: Tensor
    l1: Tensor
    mrstft: Tensor


class NablafxLoss(nn.Module):
    """NablaFX's published 0.5 L1 + 0.5 default MR-STFT objective."""

    def __init__(self, l1_weight: float = 0.5, mrstft_weight: float = 0.5) -> None:
        super().__init__()
        if l1_weight < 0.0 or mrstft_weight < 0.0:
            raise ValueError("NablaFX loss weights must be non-negative")
        if l1_weight + mrstft_weight <= 0.0:
            raise ValueError("at least one NablaFX loss weight must be positive")
        self.l1_weight = float(l1_weight)
        self.mrstft_weight = float(mrstft_weight)
        self.mrstft = _load_exact_mrstft_class()()

    def components(self, output: Tensor, target: Tensor) -> NablafxLossComponents:
        if output.shape != target.shape or output.ndim != 2:
            raise ValueError("NablaFX loss expects paired (batch,time) tensors")
        l1 = torch.nn.functional.l1_loss(output, target)
        spectral = self.mrstft(output[:, None], target[:, None])
        total = self.l1_weight * l1 + self.mrstft_weight * spectral
        return NablafxLossComponents(total=total, l1=l1, mrstft=spectral)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return self.components(output, target).total
