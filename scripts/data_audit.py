"""Validate the pre-download dataset inventory without accessing result data."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REQUIRED_FIELDS = {
    "name",
    "source",
    "version",
    "license",
    "paired_input_output",
    "allowed_usage",
    "assigned_tier",
    "known_limitations",
}
ALLOWED_TIERS = {
    "SYNTHETIC",
    "INTERNAL_DEV",
    "INTERNAL_VALIDATION",
    "EXTERNAL_REPORT_ONLY",
    "UNASSIGNED",
}


def audit_catalog(path: Path) -> list[str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    resources = document.get("resources", []) if isinstance(document, dict) else []
    if not resources:
        errors.append("catalog has no resources")
    for index, resource in enumerate(resources):
        missing = sorted(REQUIRED_FIELDS - set(resource))
        if missing:
            errors.append(f"resource {index} missing fields: {', '.join(missing)}")
        if resource.get("assigned_tier") not in ALLOWED_TIERS:
            errors.append(f"resource {index} has invalid assigned_tier")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    args = parser.parse_args()
    errors = audit_catalog(args.catalog)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"dataset catalog valid: {args.catalog}")


if __name__ == "__main__":
    main()
