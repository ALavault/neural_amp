#!/usr/bin/env python3
"""Manifest of the data and of the estimation / reading splits (NOTE.md).

Writes manifest.json: for each source, its path, rate, licence note, which part is
estimation (A) and which is reading (B), whether A and B are independent sources, and
the sha256 of the input files (VS100R takes are listed by file name and pass start,
not hashed: about 19 GB). CPU, a few seconds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import soundfile as sf

from fssr_nam.product import defect_bench as db
from fssr_nam.product import vs100r

HERE = Path(__file__).resolve().parent
JVM = vs100r.ROOT / "datasets/raw/external/tone_twist_param/extracted"
JVM = JVM / "Marshall-JVM410H-ChOD1-trimmed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def section_entry(name: str) -> dict:
    entry = vs100r.section(name)
    return {"start": entry["start"], "samples": entry["samples"], "rate": vs100r.RATE}


def synthetic() -> dict:
    return {
        "kind": "synthetic DSP reference and known-defect variants (defect_bench.py)",
        "rate_hz": 48_000,
        "families": {
            family: {
                "level_kind": kind,
                "levels_db": list(db.LEVELS_DB[kind]),
                "estimation_seeds": "s (dev 100-102, test 200-202)",
                "reading_seeds": "s + 50",
            }
            for family, kind in db.LEVEL_KIND.items()
            if family != "di"
        },
        "di": {
            "source": db.DI_SOURCE,
            "sha256": sha256(vs100r.ROOT / db.DI_SOURCE),
            "estimation": {"section": "di_train", **section_entry("di_train")},
            "reading": {"section": "di_test", **section_entry("di_test")},
            "independent_sources": True,
            "note": "di_train: idmt-gtr2, idmt-gtr4-sg, nam, prvt-gtr, yt-bass; "
            "di_test: idmt-bass, idmt-gtr4-ib (files never in di_train)",
        },
        "two_tone_note": "tone pair from seed % 50: shared by A (s) and B (s + 50)",
    }


def real_vs100r() -> dict:
    takes = [
        {
            "order": t["order"],
            "setting": t["setting"],
            "loopback": t["pass"]["file"],
            "pass_start": t["pass"]["start"],
            "slipped": t["slipped"],
            "low_signal": t["low_signal"],
        }
        for t in vs100r.sheet()
        if t["channel"] == "OD2" and t["power_dimension"] == "off"
    ]
    return {
        "kind": "VS100R FX send, own recording 2026-09-26 (raw audio not committed)",
        "media": str(vs100r.MEDIA.relative_to(vs100r.ROOT)),
        "analysis_rate_hz": 48_000,
        "estimation": {"section": "di_train", **section_entry("di_train")},
        "reading": [
            {"section": s, **section_entry(s)}
            for s in ("di_test", "pink_noise_-30dBFS_rms")
        ],
        "harmonic": [
            section_entry(s) | {"section": s}
            for s in ("sweep_-20dBFS", "sweep_-10dBFS", "sweep_-3dBFS")
        ],
        "independent_sources": True,
        "takes": takes,
    }


def real_jvm() -> dict:
    dry = JVM / "DRY/trainval/trainval.input.wav"
    rate = sf.info(dry).samplerate
    return {
        "kind": "Marshall JVM410H OD1, ToneTwist AFx (CC BY-NC 4.0)",
        "dry": str(dry.relative_to(vs100r.ROOT)),
        "sha256": sha256(dry),
        "rate_hz": rate,
        "estimation_s": [0, 150],
        "reading_s": [215, 300],
        "independent_sources": False,
        "note": "two disjoint segments of one input file whose source boundaries are "
        "not documented: A/B independence NOT established",
    }


def main() -> None:
    manifest = {
        "synthetic": synthetic(),
        "vs100r": real_vs100r(),
        "jvm410h": real_jvm(),
        "not_measurable": [
            "BLA in the strict sense on real data (no periodic multisine, no repeats)",
            "synchronized-sweep harmonic phase on VS100R (sweeps not synchronized)",
            "levels other than the three sweep levels; stepped sines",
            "day-to-day repeatability; cold / warm / hot takes; volt calibration",
            "content above 20 kHz (sweeps) or 24 kHz (DI)",
            "VS100R output after the tone stack",
        ],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
