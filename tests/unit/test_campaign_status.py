from __future__ import annotations

import json
from pathlib import Path

from fssr_nam.campaign.status import format_status


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_format_status_supports_lineage_specific_freeze_schema(tmp_path: Path) -> None:
    root = tmp_path / ".codex_campaign"
    root.mkdir()
    (root / "ACTIVE_CAMPAIGN").write_text("candidate_v1\n", encoding="utf-8")
    _write_json(
        root / "candidate_v1/MATURITY.json",
        {"current_stage": "round_1_invalid", "status": "terminal_invalid"},
    )
    _write_json(
        root / "candidate_v1/EXTERNAL_FREEZE.json",
        {
            "external_report_only_locked": True,
            "blackstar_accessed": False,
            "ua1176_accessed": False,
        },
    )

    assert format_status(root) == (
        "campaign=candidate_v1 maturity=round_1_invalid status=terminal_invalid "
        "external_report_only_locked=true blackstar_accessed=false "
        "ua1176_accessed=false"
    )
