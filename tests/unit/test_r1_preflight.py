from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fssr_nam.reporting.r1_preflight import (
    protocol_digest,
    validate_protocol_counts,
    validate_stage_configs,
)

ROOT = Path(__file__).resolve().parents[2]


def test_r1_configs_expose_required_resolved_fields() -> None:
    stages = validate_stage_configs(ROOT)
    assert set(stages) == {
        "competence",
        "factorial",
        "horizon",
        "cascade",
        "confirm",
    }


def test_r1_protocol_caps_are_exact() -> None:
    counts = validate_protocol_counts(ROOT)
    assert counts["diagnostic_total"] == 19
    assert counts["confirmatory_total"] == 76


def test_protocol_digest_is_ordered_and_content_sensitive(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("one", encoding="utf-8")
    (tmp_path / "b").write_text("two", encoding="utf-8")
    first = protocol_digest(tmp_path, ("a", "b"))
    assert first == protocol_digest(tmp_path, ("a", "b"))
    assert first != protocol_digest(tmp_path, ("b", "a"))
    (tmp_path / "b").write_text("changed", encoding="utf-8")
    assert first != protocol_digest(tmp_path, ("a", "b"))


def test_malformed_confirmation_count_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "configs/r1"
    config_path.mkdir(parents=True)
    protocol = yaml.safe_load((ROOT / "configs/r1/protocol.yaml").read_text())
    protocol["confirmation"]["extension_seeds"] = [3]
    (config_path / "protocol.yaml").write_text(yaml.safe_dump(protocol))
    with pytest.raises(ValueError, match="76"):
        validate_protocol_counts(tmp_path)
