from __future__ import annotations

import numpy as np

from fssr_nam.data.corpus import build_corpus_manifest


def test_complete_synthetic_corpus_manifest_is_deterministic() -> None:
    config = {
        "master_sample_rate": 192_000,
        "duration_seconds": 0.01,
        "seed": 3,
        "excitations": ["two_tone", "level_changes"],
        "systems": ["polynomial", "wiener_hammerstein", "slow_sag"],
    }
    external_di = np.linspace(-0.25, 0.25, 480, dtype=np.float32)
    first = build_corpus_manifest(config, external_di_48k=external_di)
    second = build_corpus_manifest(config, external_di_48k=external_di)
    assert first == second
    assert first["excitation_count"] == 2
    assert first["system_count"] == 3
    assert first["pair_count"] == 6
    assert set(first["external_di_diagnostic"]["targets"]) == set(config["systems"])
    assert all(
        len(summary["sha256"]) == 64
        for summary in first["excitations"].values()
        for summary in summary.values()
    )
