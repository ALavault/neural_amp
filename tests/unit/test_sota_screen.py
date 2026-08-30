import copy
import math
from pathlib import Path

import yaml

from fssr_nam.campaign.amp_sota_gates import (
    evaluate_mechanism_promotion_gate,
)
from fssr_nam.campaign.amp_sota_prototype_v1 import load_protocol
from fssr_nam.metrics.sota_screen import run_synthetic_mechanism_screen

ROOT = Path(__file__).resolve().parents[2]


def test_small_synthetic_screen_is_deterministic_and_gate_shaped() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/amp_sota_prototype_v1_1/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    screen = config["mechanism_screen"]
    screen["approximant"].update(
        {
            "train_samples": 257,
            "evaluation_samples": 256,
            "updates": 2,
            "asr_dft_samples": 256,
            "asr_frames": 2,
            "asr_k0_values": [17],
            "asr_amplitudes": [0.25],
        }
    )
    screen["slow_control"].update({"frames": 8, "frame_samples": 8})
    screen["resampler"]["response_bins"] = 1024
    first = run_synthetic_mechanism_screen(config)
    second = run_synthetic_mechanism_screen(config)
    assert first == second
    assert first["physical_audio_samples_read"] == 0
    assert set(first["axes"]) == {"approximant", "slow_control", "resampler"}


def test_frozen_screen_produces_finite_gate_evidence_with_reduced_fixture() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/amp_sota_prototype_v1_1/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = copy.deepcopy(config)
    screen = config["mechanism_screen"]
    screen["approximant"].update(
        {
            "train_samples": 257,
            "evaluation_samples": 256,
            "updates": 2,
            "asr_dft_samples": 256,
            "asr_frames": 2,
            "asr_k0_values": [17],
            "asr_amplitudes": [0.25],
        }
    )
    screen["slow_control"].update({"frames": 8, "frame_samples": 8})
    screen["resampler"]["response_bins"] = 1024
    evidence = run_synthetic_mechanism_screen(config)
    gate = evaluate_mechanism_promotion_gate(evidence, load_protocol(ROOT))
    assert gate["valid"] is True
    assert gate["axis_results"]["resampler"]["passed"] is False
    assert gate["axis_results"]["resampler"]["cost_measurement_available"] is False
    for axis in evidence["axes"].values():
        for row in [axis["control"], *axis["candidates"]]:
            assert all(
                not isinstance(value, float) or math.isfinite(value)
                for value in row["metrics"].values()
            )
