#!/usr/bin/env python3
"""Prove the capture chain end to end, without an audio interface.

A trained `.nam` model plays the part of the unknown device: the reamp signal
goes through it, a random latency and gain are applied, and the quality control
has to recover a usable dataset from the capture alone. The chain is then run
all the way to a retrained model and its test ESR.
"""

from __future__ import annotations

import json
import secrets
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from product_report import MANIFEST, ROOT, RUNS, build_tools, run_native
from product_train import run_training

from fssr_nam.product.capture import (
    SAMPLE_RATE,
    SEGMENT_SECONDS,
    ClippedCapture,
    SilentCapture,
    marker_samples,
    quality_control,
    reamp_signal,
    split_pairs,
)
from fssr_nam.product.data import device_pairs

OUT_DIR = ROOT / "demo/capture"
STAND_IN = "fulltone_full_drive_2"
RUN_ID = "selftest_capture_chain"
ESR_LIMIT = 0.02
MAX_EPOCHS = 100
MAX_LATENCY = 4_800  # 100 ms, above any real interface round trip


def dbfs(signal: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(signal)))))


def main() -> int:
    seed = secrets.randbits(32)
    rng = np.random.default_rng(seed)
    latency = int(rng.integers(0, MAX_LATENCY))
    print("=== Auto-test de la chaîne de capture ===")
    print(f"graine tirée (secrets.randbits): {seed}")
    print(f"latence injectée : {latency} échantillons")

    _, runner = build_tools()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Génération du signal de reamp")
    pairs = device_pairs(MANIFEST, STAND_IN, root=ROOT)
    program = {}
    for split, seconds in SEGMENT_SECONDS.items():
        audio, rate = sf.read(pairs[split][0], dtype="float32")
        if rate != SAMPLE_RATE:
            raise RuntimeError(f"{split} program is at {rate} Hz")
        program[split] = audio[: int(seconds * SAMPLE_RATE)]
    reference = reamp_signal(program)
    reference_path = OUT_DIR / "reamp_reference.wav"
    sf.write(reference_path, reference, SAMPLE_RATE, subtype="FLOAT")
    print(
        f"  reamp FSSR v0 (disposition inspirée de NAM v3) : {len(reference)} "
        f"échantillons, {len(reference) / SAMPLE_RATE:.1f} s, "
        f"{marker_samples()} échantillons de marqueurs à chaque bout"
    )

    print("\n[2/4] Rendu à travers l'appareil fictif")
    model_path = ROOT / "demo/runs" / RUNS[STAND_IN] / "model_full.nam"
    print(f"  appareil fictif : {model_path.relative_to(ROOT)}")
    clean, _ = run_native(runner, model_path, reference, "64")
    # An operator sets the trim so the take does not hit the ceiling, so the
    # injected gain is drawn inside the real headroom. Clipping is exercised
    # deliberately in the rejection step below.
    headroom_db = float(20.0 * np.log10(0.95 / np.max(np.abs(clean))))
    gain_db = float(rng.uniform(-6.0, min(6.0, headroom_db)))
    gain = 10.0 ** (gain_db / 20.0)
    print(f"  marge avant saturation : {headroom_db:+.2f} dB")
    print(f"  gain injecté           : {gain_db:+.4f} dB")
    fed = np.concatenate(
        [
            np.zeros(latency, dtype=np.float32),
            reference,
            np.zeros(SAMPLE_RATE, dtype=np.float32),
        ]
    )
    captured, _ = run_native(runner, model_path, fed, "64")
    capture = (captured * gain).astype(np.float32)
    capture_path = OUT_DIR / "reamp_capture.wav"
    sf.write(capture_path, capture, SAMPLE_RATE, subtype="FLOAT")

    print("\n[3/4] Contrôle qualité (référence + capture uniquement)")
    report = quality_control(reference, capture)
    print("  " + json.dumps(report))
    estimated = int(report["latency_samples"])
    latency_error = estimated - latency
    # The QC cannot separate device gain from interface trim behind a
    # nonlinearity, so it reports a level; the injected gain is recovered here,
    # where the clean render is known.
    program_start = marker_samples()
    program_stop = program_start + int(sum(SEGMENT_SECONDS.values()) * SAMPLE_RATE)
    clean_program = clean[program_start:program_stop]
    level_error = report["level_dbfs"] - dbfs(clean_program) - gain_db
    print(
        f"  latence estimée {estimated}, injectée {latency} -> écart "
        f"{latency_error} échantillon(s)"
    )
    print(
        f"  niveau QC {report['level_dbfs']:.4f} dBFS, niveau propre "
        f"{dbfs(clean_program):.4f} dBFS, gain injecté {gain_db:+.4f} dB -> "
        f"écart {level_error:+.4f} dB"
    )
    aligned = capture[estimated : estimated + len(reference)]
    residual = float(np.max(np.abs(aligned / np.float32(gain) - clean)))
    print(f"  max|capture alignée / gain - rendu propre| = {residual:.3e}")
    if latency_error != 0 or abs(level_error) >= 0.1:
        print("ÉCHEC : alignement ou niveau hors tolérance")
        return 1

    print("\n[3b/4] Rejets attendus")
    for name, broken, expected in (
        (
            "capture écrêtée",
            np.clip(capture * 12.0, -1.0, 1.0).astype(np.float32),
            ClippedCapture,
        ),
        ("capture silencieuse", np.zeros_like(capture), SilentCapture),
    ):
        try:
            quality_control(reference, broken)
        except expected as error:
            print(f"  {name} -> {type(error).__name__}: {error}")
        else:
            print(f"ÉCHEC : {name} acceptée par le contrôle qualité")
            return 1

    print("\n[4/4] Réentraînement sur la capture alignée")
    dataset = OUT_DIR / "dataset"
    shutil.rmtree(dataset, ignore_errors=True)
    dataset.mkdir(parents=True)
    trained_pairs = {}
    for split, (sent, got) in split_pairs(reference, capture, estimated).items():
        sent_path = dataset / f"{split}_input.wav"
        got_path = dataset / f"{split}_target.wav"
        sf.write(sent_path, sent, SAMPLE_RATE, subtype="FLOAT")
        sf.write(got_path, got, SAMPLE_RATE, subtype="FLOAT")
        trained_pairs[split] = (sent_path, got_path)
    record = run_training(
        trained_pairs,
        run_id=RUN_ID,
        device="SELFTEST_FICTIVE_DEVICE_not_a_real_capture",
        seed=0,
        max_epochs=MAX_EPOCHS,
        progress_bar=False,
    )
    esr = record["test_esr"]["full"]
    print("  " + json.dumps(record))
    print(f"  ESR de test (A2 full) = {esr:.5f}, limite {ESR_LIMIT}")
    if esr >= ESR_LIMIT:
        print("ÉCHEC : ESR au-dessus de la limite")
        return 1

    print("\n=== Chaîne de capture validée ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
