"""Same-weight transfer across the four R2 antialiasing modes."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def _canonical(name: str) -> str:
    return name.replace("processor.branch.", "processor.").replace(
        ".activation.spline.", ".spline."
    )


def _expanded(source: Tensor, target: Tensor, factor: int, name: str) -> Tensor:
    if source.shape == target.shape:
        return source
    same_leading = source.shape[:-1] == target.shape[:-1]
    scaled_tail = target.shape[-1] - 1 == factor * (source.shape[-1] - 1)
    if source.ndim >= 1 and same_leading and scaled_tail:
        result = torch.zeros_like(target)
        result[..., ::factor] = source
        return result
    raise ValueError(f"cannot preserve same weights for scaled tensor {name}")


def synchronize_aa_weights(
    reference_off: nn.Module, variants: Iterable[nn.Module]
) -> None:
    """Copy one AA-off parameterization into ADAA, x2, and x4 variants.

    Temporal FIR tensors are zero-insertion dilated. Fixed resampling filters are
    intentionally not copied because they are protocol constants, not learned
    weights. Every target trainable parameter must be accounted for.
    """
    if getattr(reference_off, "aa_mode", None) != "off":
        raise ValueError("same-weight reference must use aa_mode=off")
    source_state = {
        _canonical(name): value.detach()
        for name, value in reference_off.state_dict().items()
    }
    source_parameters = {
        _canonical(name) for name, _ in reference_off.named_parameters()
    }
    for target in variants:
        factor = int(getattr(target, "dilation_scale", 0))
        if factor not in {1, 2, 4}:
            raise ValueError("target has an invalid R2 dilation scale")
        target_parameters = {
            _canonical(name): name for name, _ in target.named_parameters()
        }
        if set(target_parameters) != source_parameters:
            missing = sorted(source_parameters - set(target_parameters))
            extra = sorted(set(target_parameters) - source_parameters)
            raise ValueError(
                "same-weight trainable mapping mismatch; "
                f"missing={missing}, extra={extra}"
            )
        resolved = target.state_dict()
        with torch.no_grad():
            for target_name, target_value in resolved.items():
                source = source_state.get(_canonical(target_name))
                if source is None:
                    if (
                        "upsample_filter" in target_name
                        or "downsample_filter" in target_name
                    ):
                        continue
                    raise ValueError(f"unmapped same-weight state: {target_name}")
                target_value.copy_(_expanded(source, target_value, factor, target_name))
        target.load_state_dict(resolved, strict=True)
