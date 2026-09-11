from __future__ import annotations

import pytest

from fssr_nam.campaign.r2_48k import (
    R248KAuthorizationError,
    open_internal_validation_tests,
    unlock_internal_validation,
)


def _freeze() -> dict:
    return {
        "campaign_version": "FSSR-R2-48K-v1",
        "development_tests_previously_observed": True,
        "external_report_only_locked": True,
        "external_results_accessed": False,
        "external_retest_authorized": False,
        "internal_validation_outputs_locked": True,
        "internal_validation_tests_locked": True,
        "internal_validation_test_open_count": 0,
        "internal_validation_tests_opened": False,
        "internal_validation_train_validation_locked": True,
        "schema_version": 1,
    }


def test_confirmatory_lock_releases_validation_but_not_tests() -> None:
    unlocked = unlock_internal_validation(_freeze())
    assert unlocked["internal_validation_train_validation_locked"] is False
    assert unlocked["internal_validation_tests_locked"] is True
    assert unlocked["internal_validation_outputs_locked"] is True
    assert unlocked["internal_validation_test_open_count"] == 0

    opened = open_internal_validation_tests(unlocked)
    assert opened["internal_validation_tests_locked"] is False
    assert opened["internal_validation_outputs_locked"] is False
    assert opened["internal_validation_tests_opened"] is True
    assert opened["internal_validation_test_open_count"] == 1
    with pytest.raises(R248KAuthorizationError, match="already opened"):
        open_internal_validation_tests(opened)
