from __future__ import annotations

import pytest

from fssr_nam.statistics.r1 import (
    CostObservation,
    EsrObservation,
    decide_confirmation,
    paired_hierarchical_bootstrap,
)

DEVICES = ("fulltone", "bigmuff", "blackstar", "ua1176")


def esr_rows(candidate_ratio: float) -> list[EsrObservation]:
    rows = []
    for device_index, device in enumerate(DEVICES):
        for seed in range(5):
            for source in ("source-a", "source-b"):
                baseline = 0.2 + 0.01 * device_index + 0.001 * seed
                rows.append(EsrObservation(device, source, seed, "a2", baseline))
                rows.append(
                    EsrObservation(
                        device,
                        source,
                        seed,
                        "r1_final",
                        baseline * candidate_ratio,
                    )
                )
    return rows


def cost_rows(candidate_ratio: float) -> list[CostObservation]:
    rows = []
    for device in DEVICES:
        for seed in range(5):
            rows.append(CostObservation(device, seed, "a2", 64, 100.0, 0))
            rows.append(
                CostObservation(
                    device, seed, "r1_final", 64, 100.0 * candidate_ratio, 0
                )
            )
    return rows


def test_h1_decision_uses_paired_seed_and_source_units() -> None:
    decision = decide_confirmation(
        esr_rows(0.90), cost_rows(1.0), replicates=100, seed=20260827
    )
    assert decision.verdict == "GO-A"
    assert decision.h1_passed
    assert decision.devices_won == sorted(DEVICES)


def test_h2_noninferiority_and_cpu_reduction() -> None:
    decision = decide_confirmation(
        esr_rows(1.0), cost_rows(0.70), replicates=100, seed=20260827
    )
    assert decision.verdict == "GO-B"
    assert decision.h2_passed
    assert decision.esr_ratio.one_sided_upper_95 == pytest.approx(1.0)


def test_incomplete_pairing_is_rejected() -> None:
    rows = esr_rows(0.9)
    rows.pop()
    with pytest.raises(ValueError, match="pairing is incomplete"):
        paired_hierarchical_bootstrap(rows, replicates=5)


def test_windows_cannot_be_passed_as_independent_units() -> None:
    with pytest.raises(TypeError):
        EsrObservation(  # type: ignore[call-arg]
            device="fulltone",
            source="source-a",
            seed=0,
            family="a2",
            esr=0.1,
            window=3,
        )
