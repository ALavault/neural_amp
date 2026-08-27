from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/data/r1_physical.yaml").read_text(encoding="utf-8")
    )


def test_ua1176_sources_match_preregistered_disjoint_splits() -> None:
    ua = load_config()["devices"]["ua1176"]
    source_ids = {
        split: {source["source_id"] for source in sources}
        for split, sources in ua["splits"].items()
    }
    assert source_ids == {
        "train": {"bass1", "bass3", "bass4", "gtr6", "gtr8"},
        "validation": {"bass5", "gtr9"},
        "test": {"bass2", "gtr7"},
    }
    assert not source_ids["train"] & source_ids["validation"]
    assert not source_ids["train"] & source_ids["test"]
    assert not source_ids["validation"] & source_ids["test"]


def test_confirmation_audio_duration_is_equal_by_device_and_split() -> None:
    config = load_config()
    for device in config["devices"].values():
        for split, expected in config["equal_total_seconds"].items():
            observed = sum(
                float(source["maximum_seconds"]) for source in device["splits"][split]
            )
            assert observed == float(expected)


def test_r1_data_roles_and_exclusions_are_frozen() -> None:
    config = load_config()
    roles = {
        device: specification["r1_tier"]
        for device, specification in config["devices"].items()
    }
    assert roles == {
        "fulltone": "INTERNAL_DEV",
        "bigmuff": "INTERNAL_DEV",
        "blackstar": "INTERNAL_VALIDATION",
        "ua1176": "INTERNAL_VALIDATION",
    }
    assert config["excluded_from_primary_confirmation"] == [
        {
            "device": "marshall_jvm410h",
            "reason": (
                "no_fixed_source_disjoint_splits_suitable_for_one_model_per_device"
            ),
        }
    ]
    assert config["external_report_only"] == {"locked": True, "resources": []}
