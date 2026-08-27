"""Time-domain metrics with explicit numerical conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def error_to_signal_ratio(
    prediction: ArrayLike, target: ArrayLike, *, epsilon: float = 1.0e-12
) -> float:
    """Return sum squared error divided by target energy.

    No alignment, gain correction, or DC correction is applied implicitly.
    """
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must have identical shapes")
    if not np.all(np.isfinite(prediction_array)) or not np.all(
        np.isfinite(target_array)
    ):
        raise ValueError("prediction and target must be finite")
    numerator = np.sum(np.square(prediction_array - target_array), dtype=np.float64)
    denominator = np.sum(np.square(target_array), dtype=np.float64)
    return float(numerator / (denominator + epsilon))
