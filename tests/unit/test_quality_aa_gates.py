from __future__ import annotations

from fssr_nam.campaign.quality_aa_v2_gates import evaluate_mechanism_gate
from fssr_nam.data.r2_fixtures import R2_FIXTURES


def _evidence(*, fail_x2: bool) -> dict:
    rows = []
    modes = ("off", "full_island_x2", "teacher_x4")
    for fixture_index, fixture in enumerate(R2_FIXTURES):
        for mode in modes:
            conditions = []
            for condition_index, (k0, amplitude) in enumerate(
                (923 + fixture_index * 9 + index, 0.1 + 0.01 * index)
                for index in range(9)
            ):
                rank = 1.0 + fixture_index * 9 + condition_index
                conditions.append(
                    {
                        "k0": k0,
                        "amplitude": amplitude,
                        "off_asr_floor_censored": False,
                        "paired_asr_gain_db": 0.0 if mode == "off" else 12.0,
                        "asr_linear": rank * 1.0e-6,
                        "known_reference_alias_residual_linear": rank * 2.0e-6,
                    }
                )
            rows.append(
                {
                    "fixture": fixture,
                    "mode": mode,
                    "latency_samples": 0 if mode == "off" else 32,
                    "guard_passed": not (
                        fail_x2 and mode == "full_island_x2" and fixture == "tanh"
                    ),
                    "known_reference_alias_residual": {
                        "median": -50.0 if mode == "full_island_x2" else -60.0
                    },
                    "residual_energy_ratio": (
                        0.2 if fixture == "rf2047_residual" else 0.0
                    ),
                    "conditions": conditions,
                }
            )
    return {
        "campaign_version": "FSSR-QUALITY-AA-v2",
        "reference_kind": "direct_synthetic_x8_x16",
        "physical_audio_samples_read": 0,
        "same_weights_across_modes": True,
        "rows": rows,
    }


def test_route_failure_does_not_poison_the_other_route() -> None:
    result = evaluate_mechanism_gate(_evidence(fail_x2=True))
    assert result["routes"]["full_island_x2"]["passed"] is False
    assert result["routes"]["teacher_x4"]["passed"] is True
    assert result["passing_routes"] == ["teacher_x4"]
    assert result["route_decisions_independent"] is True
