import copy

import pytest

from fssr_nam.statistics.amp_sota import (
    AmpSotaStatisticsError,
    hierarchical_confirmation_bootstrap,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "device": device,
            "seed": seed,
            "source": source,
            "baseline": {"esr": 1.0},
            "candidate": {"esr": 0.8},
        }
        for device in ("blackstar", "ua1176")
        for seed in range(5)
        for source in ("source_a", "source_b")
    ]


def test_bootstrap_is_deterministic_paired_and_uses_no_windows() -> None:
    first = hierarchical_confirmation_bootstrap(_rows(), replicates=200, seed=7)
    second = hierarchical_confirmation_bootstrap(_rows(), replicates=200, seed=7)
    assert first == second
    assert first["esr_relative_improvement"] == pytest.approx(0.2)
    assert first["esr_relative_improvement_lower_95_bound"] == pytest.approx(0.2)
    assert first["windows_resampled"] is False
    assert first["device_esr_relative_improvement"] == {
        "blackstar": pytest.approx(0.2),
        "ua1176": pytest.approx(0.2),
    }


def test_bootstrap_rejects_missing_seed_duplicate_source_and_nonfinite() -> None:
    missing = [row for row in _rows() if row["seed"] != 4]
    with pytest.raises(AmpSotaStatisticsError, match="seeds"):
        hierarchical_confirmation_bootstrap(missing, replicates=2)

    duplicate = _rows()
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(AmpSotaStatisticsError, match="duplicate"):
        hierarchical_confirmation_bootstrap(duplicate, replicates=2)

    invalid = _rows()
    invalid[0]["candidate"]["esr"] = float("nan")
    with pytest.raises(AmpSotaStatisticsError, match="finite"):
        hierarchical_confirmation_bootstrap(invalid, replicates=2)
