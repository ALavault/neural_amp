"""Reamp signal, alignment and quality control for product captures.

The layout is fixed by the constants below, so the quality control only ever
needs the reference signal it sent and the capture it got back. It never sees
the latency, the gain or the clean signal of the device under test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 48_000
BLIP_MILLISECONDS = 2.0
BLIP_HZ = 1_000.0
BLIP_AMPLITUDE = 0.5
# Silence around each blip, so the onset is unambiguous at any playing level.
GAP_SECONDS = 0.5
# Program material, in the order the reamp signal carries it.
SEGMENT_SECONDS = {"train": 120.0, "validation": 30.0, "test": 30.0}

# The first blip starts one gap into the signal; the search window is one second.
MARKER_POSITION = int(GAP_SECONDS * SAMPLE_RATE)
SEARCH_SAMPLES = SAMPLE_RATE
BASELINE_SAMPLES = 4_800
CLIP_LEVEL = 0.999
CLIP_RUN = 3
SILENCE_DBFS = -60.0


class ClippedCapture(Exception):
    """The capture hit the converter ceiling: the take must be redone lower."""


class SilentCapture(Exception):
    """The capture carries no usable program: check routing and levels."""


def _blip() -> NDArray[np.float32]:
    """A burst that starts at full amplitude, so its onset is its first sample."""
    length = int(BLIP_MILLISECONDS * SAMPLE_RATE / 1000)
    index = np.arange(length)
    decay = np.cos(0.5 * np.pi * index / length)
    tone = np.cos(2.0 * np.pi * BLIP_HZ * index / SAMPLE_RATE)
    return (BLIP_AMPLITUDE * tone * decay).astype(np.float32)


def _markers() -> NDArray[np.float32]:
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([gap, _blip(), gap, _blip(), gap])


def marker_samples() -> int:
    return len(_markers())


def reamp_signal(program: dict[str, NDArray[np.float32]]) -> NDArray[np.float32]:
    """Build the reamp signal: calibration markers, program material, markers."""
    parts = [_markers()]
    for split, seconds in SEGMENT_SECONDS.items():
        expected = int(seconds * SAMPLE_RATE)
        segment = np.asarray(program[split], dtype=np.float32)
        if segment.shape != (expected,):
            raise ValueError(f"{split} must be {expected} samples, got {segment.shape}")
        parts.append(segment)
    parts.append(_markers())
    return np.concatenate(parts).astype(np.float32)


def _program_region(signal: NDArray[np.float32]) -> NDArray[np.float32]:
    start = marker_samples()
    stop = start + int(sum(SEGMENT_SECONDS.values()) * SAMPLE_RATE)
    return signal[start:stop]


def estimate_latency(capture: NDArray[np.float32]) -> int:
    """Return the capture's latency in samples, from the first calibration blip."""
    window = capture[MARKER_POSITION : MARKER_POSITION + SEARCH_SAMPLES]
    baseline = capture[MARKER_POSITION - BASELINE_SAMPLES : MARKER_POSITION]
    # A trained model carries a DC offset, so the reference is the median, not zero.
    threshold = max(10.0 * float(np.std(baseline)), 1e-3)
    onset = np.nonzero(np.abs(window - np.median(baseline)) > threshold)[0]
    if onset.size == 0:
        raise SilentCapture(
            "capture silencieuse : aucun blip de calibration trouvé dans la "
            f"fenêtre de recherche (seuil {threshold:.2e})"
        )
    return int(onset[0])


def quality_control(
    reference: NDArray[np.float32], capture: NDArray[np.float32]
) -> dict[str, float]:
    """Align the capture on the reference and reject unusable takes."""
    latency = estimate_latency(capture)
    aligned = capture[latency : latency + len(reference)]
    if aligned.shape != reference.shape:
        raise SilentCapture(
            f"capture trop courte : {aligned.size} échantillons utiles après "
            f"alignement, {reference.size} attendus"
        )
    program = _program_region(aligned)
    hot = np.abs(program) >= CLIP_LEVEL
    runs = np.convolve(hot.astype(np.int32), np.ones(CLIP_RUN, dtype=np.int32), "valid")
    if np.any(runs >= CLIP_RUN):
        raise ClippedCapture(
            f"capture écrêtée : {int(np.count_nonzero(hot))} échantillons à "
            f"|x| >= {CLIP_LEVEL}, dont au moins {CLIP_RUN} consécutifs"
        )
    level_dbfs = 20.0 * np.log10(float(np.sqrt(np.mean(np.square(program)))))
    if level_dbfs < SILENCE_DBFS:
        raise SilentCapture(
            f"capture silencieuse : programme à {level_dbfs:.1f} dBFS, "
            f"seuil {SILENCE_DBFS:.1f} dBFS"
        )
    return {
        "latency_samples": float(latency),
        "level_dbfs": level_dbfs,
        "peak": float(np.max(np.abs(program))),
    }


def split_pairs(
    reference: NDArray[np.float32], capture: NDArray[np.float32], latency: int
) -> dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]]:
    """Cut the aligned reamp pass into the train, validation and test pairs."""
    aligned = capture[latency : latency + len(reference)]
    position = marker_samples()
    pairs = {}
    for split, seconds in SEGMENT_SECONDS.items():
        span = slice(position, position + int(seconds * SAMPLE_RATE))
        pairs[split] = (reference[span], aligned[span])
        position = span.stop
    return pairs
