"""Print the persisted campaign maturity and external freeze state."""

from pathlib import Path

from fssr_nam.campaign.status import format_status


def main() -> None:
    print(format_status(Path(".codex_campaign")))


if __name__ == "__main__":
    main()
