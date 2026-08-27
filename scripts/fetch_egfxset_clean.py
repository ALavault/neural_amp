"""Fetch and verify the one EGFxSet archive authorized for INTERNAL_DEV."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/api/records/7044411/files/Clean.zip/content"
EXPECTED_MD5 = "cdb1b401960f56becc8640387910e78a"
EXPECTED_SHA256 = "baa01d1d040c044eef81453e2c96c560722ed8e26736aff52a9490bfd94d3aaa"
SELECTED_MEMBER = "Clean/Neck/6-22.wav"
ROOT = Path("datasets/raw/egfxset")
ARCHIVE = ROOT / "Clean.zip"
SELECTED = ROOT / "selected/6-22.wav"


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_archive() -> None:
    if _digest(ARCHIVE, "md5") != EXPECTED_MD5:
        raise RuntimeError("EGFxSet Clean.zip MD5 does not match the Zenodo record")
    if _digest(ARCHIVE, "sha256") != EXPECTED_SHA256:
        raise RuntimeError("EGFxSet Clean.zip SHA-256 does not match the campaign pin")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        partial = ARCHIVE.with_suffix(".zip.partial")
        if partial.exists():
            raise RuntimeError(f"refusing to overwrite partial download: {partial}")
        urllib.request.urlretrieve(URL, partial)
        os.replace(partial, ARCHIVE)
    _verify_archive()

    if not SELECTED.exists():
        SELECTED.parent.mkdir(parents=True, exist_ok=True)
        partial = SELECTED.with_suffix(".wav.partial")
        if partial.exists():
            raise RuntimeError(f"refusing to overwrite partial extraction: {partial}")
        with (
            zipfile.ZipFile(ARCHIVE) as archive,
            archive.open(SELECTED_MEMBER) as source,
        ):
            with partial.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        os.replace(partial, SELECTED)

    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "archive_sha256": EXPECTED_SHA256,
                "license": "CC-BY-4.0",
                "selected": str(SELECTED),
                "selected_sha256": _digest(SELECTED, "sha256"),
                "tier": "INTERNAL_DEV",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
