"""Shared source-control provenance for generated scientific artifacts."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence


def git_state(*, ignored_generated_paths: Sequence[str] = ()) -> dict[str, object]:
    """Return HEAD and whether non-generated tracked state differs from it."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pathspecs = [".", *(f":(exclude){path}" for path in ignored_generated_paths)]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *pathspecs],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status)}
