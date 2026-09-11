from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from fssr_nam.data.r2_capture import CaptureAuditError, _verify_derived_triplets
from fssr_nam.data.r2_resampling import (
    FILTER_TAPS,
    FrozenDecimator2,
    derive_capture_rates,
    frozen_decimation_fir,
)


def test_frozen_capture_fir_is_unity_dc_symmetric_and_65_taps() -> None:
    coefficients = frozen_decimation_fir()
    assert coefficients.shape == (FILTER_TAPS,)
    np.testing.assert_allclose(coefficients, coefficients[::-1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(coefficients.sum(), 1.0, atol=1.0e-15)


def test_streaming_capture_derivation_matches_whole_and_never_normalizes() -> None:
    generator = np.random.default_rng(20260828)
    signal = generator.normal(0.0, 0.2, 12_001).astype(np.float32)
    expected_96, expected_48 = derive_capture_rates(signal)
    first = FrozenDecimator2()
    second = FrozenDecimator2()
    chunks_96 = []
    chunks_48 = []
    position = 0
    for size in (1, 7, 1024, 3, 4095, 6871):
        block = signal[position : position + size]
        if not len(block):
            break
        rate_96 = first.process(block)
        chunks_96.append(rate_96)
        chunks_48.append(second.process(rate_96))
        position += len(block)
    np.testing.assert_allclose(np.concatenate(chunks_96), expected_96, atol=1.0e-8)
    np.testing.assert_allclose(np.concatenate(chunks_48), expected_48, atol=1.0e-8)
    assert np.max(np.abs(expected_96)) < np.max(np.abs(signal))


def test_capture_audit_streams_and_rejects_a_modified_derivative(tmp_path) -> None:
    signal = 0.2 * np.sin(2.0 * np.pi * np.arange(10_001) * 313.0 / 192_000)
    rate_96, rate_48 = derive_capture_rates(signal)
    declarations = {
        "native_192000": {},
        "derived_96000": {},
        "derived_48000": {},
    }
    for role in ("dry", "hardware", "loopback"):
        for key, rate, samples in (
            ("native_192000", 192_000, signal),
            ("derived_96000", 96_000, rate_96),
            ("derived_48000", 48_000, rate_48),
        ):
            path = tmp_path / f"{role}-{key}.wav"
            sf.write(path, samples, rate, subtype="FLOAT")
            declarations[key][role] = str(path)
    _verify_derived_triplets(declarations, root=tmp_path, label="fixture")
    corrupted = rate_48.copy()
    corrupted[-1] += 0.01
    sf.write(
        declarations["derived_48000"]["hardware"],
        corrupted,
        48_000,
        subtype="FLOAT",
    )
    with pytest.raises(CaptureAuditError, match="derived FIR mismatch"):
        _verify_derived_triplets(declarations, root=tmp_path, label="fixture")
