from __future__ import annotations

import numpy as np
import pytest
import torch

from fssr_nam.data import R2_FIXTURES, apply_r2_fixture, residual_energy_ratio
from fssr_nam.models import AAFSSR, synchronize_aa_weights


@pytest.mark.parametrize("name", R2_FIXTURES)
def test_r2_fixture_set_is_finite_deterministic_and_causal(name: str) -> None:
    generator = np.random.default_rng(20260828)
    signal = generator.normal(0.0, 0.2, 8192).astype(np.float32)
    first = apply_r2_fixture(name, signal, 48_000)
    second = apply_r2_fixture(name, signal, 48_000)
    np.testing.assert_array_equal(first.output, second.output)
    assert np.isfinite(first.output).all()
    changed = signal.copy()
    changed[7000] += 1.0
    changed_output = apply_r2_fixture(name, changed, 48_000).output
    np.testing.assert_array_equal(first.output[:7000], changed_output[:7000])


def test_rf2047_fixture_has_exact_horizon_and_residual_energy_guard() -> None:
    generator = np.random.default_rng(17)
    signal = generator.normal(0.0, 0.25, 32_768).astype(np.float32)
    result = apply_r2_fixture("rf2047_residual", signal, 48_000)
    assert result.receptive_field == 2047
    assert residual_energy_ratio(result) >= 0.10
    impulse = np.zeros(4096, dtype=np.float32)
    impulse[0] = 0.25
    response = apply_r2_fixture("rf2047_residual", impulse, 48_000).residual
    assert response[2046] != 0.0
    assert np.count_nonzero(response[2047:]) == 0


def test_same_weight_transfer_maps_adaa_and_dilates_all_trainable_firs() -> None:
    torch.manual_seed(20260828)
    source = AAFSSR(core_kind="cascade", aa_mode="off")
    targets = [
        AAFSSR(core_kind="cascade", aa_mode="adaa1"),
        AAFSSR(core_kind="cascade", aa_mode="full_island_x2"),
        AAFSSR(core_kind="cascade", aa_mode="teacher_x4"),
    ]
    synchronize_aa_weights(source, targets)
    source_values = source.processor.core.shapers[0].spline.values
    target_values = targets[0].processor.core.shapers[0].activation.spline.values
    torch.testing.assert_close(target_values, source_values)
    source_fir = source.processor.core.filters[0].coefficients
    for factor, target in zip((2, 4), targets[1:], strict=True):
        target_fir = target.processor.branch.core.filters[0].coefficients
        torch.testing.assert_close(target_fir[::factor], source_fir)
        assert (
            torch.count_nonzero(
                torch.cat([target_fir[offset::factor] for offset in range(1, factor)])
            )
            == 0
        )
