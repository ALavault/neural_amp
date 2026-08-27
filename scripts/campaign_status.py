"""Print the persisted campaign maturity and external freeze state."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(".codex_campaign")
    active_path = root / "ACTIVE_CAMPAIGN"
    if active_path.exists():
        active = active_path.read_text(encoding="utf-8").strip()
        active_root = root / active
        maturity = json.loads(
            (active_root / "MATURITY.json").read_text(encoding="utf-8")
        )
        freeze = json.loads(
            (active_root / "EXTERNAL_FREEZE.json").read_text(encoding="utf-8")
        )
        print(
            f"campaign={active} "
            f"maturity="
            f"{maturity.get('current_level', maturity.get('current_stage'))} "
            f"status={maturity['status']} "
            f"external_retest_authorized={freeze['external_retest_authorized']} "
            f"internal_validation_outputs_locked="
            f"{freeze['internal_validation_outputs_locked']}"
        )
        return
    maturity = json.loads((root / "MATURITY.json").read_text(encoding="utf-8"))
    freeze = json.loads((root / "EXTERNAL_FREEZE.json").read_text(encoding="utf-8"))
    print(
        f"maturity={maturity['current_level']} "
        f"status={maturity['status']} "
        f"external_retest_authorized={freeze['external_retest_authorized']}"
    )


if __name__ == "__main__":
    main()
