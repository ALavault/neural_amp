from __future__ import annotations

from fssr_nam.metrics.validation import validate_metric_suite


def test_all_controlled_metric_checks_pass() -> None:
    validation = validate_metric_suite()
    failed = [name for name, passed in validation["checks"].items() if not passed]
    assert not failed, failed


def test_alias_label_is_explicitly_limited_to_known_reference_case() -> None:
    validation = validate_metric_suite()
    qualification = validation["aliasing_qualification"]
    assert "Only controlled_aliasing" in qualification
    assert "192 kHz" in qualification
