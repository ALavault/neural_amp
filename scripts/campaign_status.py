"""Print the persisted campaign maturity and external freeze state."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(".codex_campaign")
    maturity = json.loads((root / "MATURITY.json").read_text(encoding="utf-8"))
    freeze = json.loads((root / "EXTERNAL_FREEZE.json").read_text(encoding="utf-8"))
    print(
        f"maturity={maturity['current_level']} "
        f"status={maturity['status']} "
        f"external_retest_authorized={freeze['external_retest_authorized']}"
    )


if __name__ == "__main__":
    main()
