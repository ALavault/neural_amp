from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from fssr_nam.campaign.amp_sota_prototype_v1_2 import (
    CAMPAIGN_VERSION,
    SotaV12AuthorizationError,
    SotaV12ConfigError,
    confirmation_was_opened,
    load_protocol,
    make_run_id,
    validate_cuda_device_properties,
    validate_protocol_config,
    validate_repository_state,
    validate_stage_authorization,
)
from fssr_nam.campaign.quality_aa_provenance import digest_array_bundle

ROOT = Path(__file__).resolve().parents[2]


def _protocol() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/amp_sota_prototype_v1_2/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_v12_protocol_and_lock_are_exact_and_predecessors_remain_closed() -> None:
    protocol = load_protocol(ROOT)
    lock = yaml.safe_load(
        (ROOT / ".codex_campaign/amp_sota_prototype_v1_2/PROTOCOL_LOCK.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert protocol == lock
    assert validate_repository_state(ROOT) == protocol
    assert confirmation_was_opened(ROOT) is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("architecture", "candidate", "activation"), "safe_rational_4_3", "candidate"),
        (("optimization", "scored_start_samples"), 14_400, "scored_start"),
        (("sequential_competence", "correlation_strictly_greater_than"), 0.89, "corr"),
        (("slow_value", "paired_esr_improvement_median_minimum"), 0.049, "slow ESR"),
        (("confirmation", "median_esr_improvement_minimum"), 0.09, "final ESR"),
        (("physical_development", "aliasing_regression_allowed"), True, "fingerprint"),
    ],
)
def test_v12_protocol_rejects_decision_bearing_drift(
    path: tuple[str, ...], value: object, message: str
) -> None:
    protocol = copy.deepcopy(_protocol())
    target = protocol
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(SotaV12ConfigError, match=message):
        validate_protocol_config(protocol)


def test_v12_stage_authorization_is_strictly_sequential() -> None:
    validate_stage_authorization("preflight", {})
    with pytest.raises(SotaV12AuthorizationError, match="preflight"):
        validate_stage_authorization("competence_dynamic", {})
    decisions = {"preflight": "passed", "competence_dynamic": "passed"}
    validate_stage_authorization("competence_two_clippers", decisions)
    with pytest.raises(SotaV12AuthorizationError, match="competence_two_clippers"):
        validate_stage_authorization("competence_static", decisions)


def test_v12_run_ids_are_new_and_family_explicit() -> None:
    assert (
        make_run_id(
            stage="competence",
            system="dynamic_primary",
            family="slow_long_tcn_x2",
            seed=0,
        )
        == "sota_v1_2_competence_dynamic_primary_slow_long_tcn_x2_seed0_v1"
    )
    with pytest.raises(ValueError, match="system or family"):
        make_run_id(stage="competence", system="dynamic_primary", family="tanh", seed=0)
    assert make_run_id(
        stage="slow_value_eval",
        system="static_primary",
        family="slow_long_tcn_x2",
        seed=2,
    ).startswith("sota_v1_2_slow_value_eval_static_primary")
    assert CAMPAIGN_VERSION == "AMP-SOTA-PROTOTYPE-v1.2"


def test_v12_cuda_properties_are_exactly_bounded() -> None:
    protocol = _protocol()
    validate_cuda_device_properties(
        protocol,
        name="NVIDIA RTX PRO 4000 Blackwell",
        total_memory_bytes=25_149_898_752,
    )
    with pytest.raises(SotaV12AuthorizationError, match="device changed"):
        validate_cuda_device_properties(
            protocol, name="Other GPU", total_memory_bytes=25_149_898_752
        )
    with pytest.raises(SotaV12AuthorizationError, match="below"):
        validate_cuda_device_properties(
            protocol,
            name="NVIDIA RTX PRO 4000 Blackwell",
            total_memory_bytes=24_999_999_999,
        )


def test_v12_data_identity_covers_names_shapes_dtypes_and_values() -> None:
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    reference = digest_array_bundle({"input": value})
    assert digest_array_bundle({"input": value.copy()}) == reference
    assert digest_array_bundle({"target": value}) != reference
    assert digest_array_bundle({"input": value.astype(np.float64)}) != reference
    changed = value.copy()
    changed[0, 0] = 1.0
    assert digest_array_bundle({"input": changed}) != reference
