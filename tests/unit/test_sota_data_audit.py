from pathlib import Path

from fssr_nam.data.sota_audit import audit_sota_data_metadata

ROOT = Path(__file__).resolve().parents[2]


def test_sota_data_audit_is_metadata_only_and_source_disjoint() -> None:
    result = audit_sota_data_metadata(ROOT)
    assert result["passed"] is True
    assert result["audio_files_read"] == 0
    assert result["commercial_claim_supported"] is False
    assert {row["device"] for row in result["devices"]} == {
        "fulltone",
        "bigmuff",
        "blackstar",
        "ua1176",
    }
