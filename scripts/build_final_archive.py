#!/usr/bin/env python3
"""Build and verify a redistribution-safe archive from tracked files only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROHIBITED_PARTS = (
    "datasets/raw/",
    "/checkpoints/",
    "/predictions/",
    "/stdout.log",
    "/stderr.log",
    ".venv/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fssr-final-") as temporary:
        tar_path = Path(temporary) / "repository.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", "--output", str(tar_path), "HEAD"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            ["zstd", "-19", "--force", str(tar_path), "-o", str(output)],
            cwd=ROOT,
            check=True,
        )
    listing = subprocess.run(
        ["tar", "--zstd", "-tf", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    prohibited = [
        member for member in listing if any(part in member for part in PROHIBITED_PARTS)
    ]
    if prohibited:
        raise RuntimeError(f"archive contains prohibited paths: {prohibited[:5]}")
    archive_hash = sha256(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{archive_hash}  {output.name}\n", encoding="utf-8")
    verification = {
        "schema_version": 1,
        "archive": str(output.relative_to(ROOT)),
        "sha256": archive_hash,
        "tracked_members": len(listing),
        "prohibited_members": prohibited,
        "source": "git archive HEAD",
        "raw_audio_included": False,
        "checkpoints_included": False,
        "prediction_audio_included": False,
    }
    output.with_suffix(output.suffix + ".verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
