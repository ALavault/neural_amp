from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from fssr_nam.inference import (
    R2NativeReference,
    build_r2_native_payload,
    export_r2_native_model,
    load_r2_native_payload,
    validate_r2_native_payload,
)
from fssr_nam.models import AAFSSR, AAFSSRXL, AANAM


def _make_nontrivial(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters()):
            positions = torch.arange(parameter.numel(), dtype=parameter.dtype)
            values = 0.025 * torch.sin(0.31 * positions + 0.17 * parameter_index)
            parameter.copy_(values.reshape_as(parameter))


@pytest.mark.parametrize(
    ("mode", "factor", "latency"),
    [
        ("off", 1, 0),
        ("adaa1", 1, 1),
        ("full_island_x2", 2, 16),
        ("teacher_x4", 4, 16),
    ],
)
def test_r2_export_reference_matches_torch_for_all_aa_modes(
    tmp_path: Path, mode: str, factor: int, latency: int
) -> None:
    torch.manual_seed(20260828)
    model = AAFSSR(core_kind="cascade", aa_mode=mode)
    _make_nontrivial(model)
    signal = torch.linspace(-0.4, 0.5, 193) + 0.03 * torch.sin(torch.arange(193) * 0.27)
    model.reset_state()
    with torch.no_grad():
        expected = model.stream(signal).numpy()
    destination = export_r2_native_model(model, tmp_path / f"{mode}.json")
    payload = load_r2_native_payload(destination)
    assert payload["aa_mode"] == mode
    assert payload["internal_sample_rate"] == 48_000 * factor
    assert payload["dilation_scale"] == factor
    assert payload["latency_samples"] == latency
    assert payload["sizes"]["parameters"] > 0
    assert payload["sizes"]["weight_bytes"] > 0
    assert payload["sizes"]["persistent_state_bytes"] > 0
    assert payload["sizes"]["scratch_bytes"] > 0
    reference = R2NativeReference(payload)
    actual = reference.process(signal.numpy())
    np.testing.assert_allclose(actual, expected, atol=2.0e-5, rtol=2.0e-5)
    reference.reset()
    chunks = []
    start = 0
    for size in (1, 7, 31, 3, 64, 87):
        stop = min(len(signal), start + size)
        chunks.append(reference.process(signal[start:stop].numpy()))
        start = stop
        if start == len(signal):
            break
    irregular = np.concatenate(chunks)
    np.testing.assert_allclose(irregular, expected, atol=2.0e-5, rtol=2.0e-5)


def test_r2_export_validation_rejects_metadata_or_adaa_drift() -> None:
    payload = build_r2_native_payload(AAFSSR(core_kind="mono", aa_mode="adaa1"))
    invalid_rate = deepcopy(payload)
    invalid_rate["internal_sample_rate"] = 96_000
    with pytest.raises(ValueError, match="internal_sample_rate"):
        validate_r2_native_payload(invalid_rate)
    invalid_limit = deepcopy(payload)
    invalid_limit["core"]["shapers"][0]["difference_limit_threshold"] = 2.0e-4
    with pytest.raises(ValueError, match="difference limit"):
        validate_r2_native_payload(invalid_limit)


@pytest.mark.parametrize(
    ("mode", "factor"),
    [("full_island_x2", 2), ("teacher_x4", 4)],
)
def test_quality_aa_latency32_export_matches_torch(
    tmp_path: Path, mode: str, factor: int
) -> None:
    torch.manual_seed(20260828)
    model = AAFSSR(core_kind="cascade", aa_mode=mode, aa_latency_samples=32)
    _make_nontrivial(model)
    signal = torch.linspace(-0.4, 0.5, 193)
    model.reset_state()
    with torch.no_grad():
        expected = model.stream(signal).numpy()
    path = export_r2_native_model(model, tmp_path / f"quality-{mode}.json")
    payload = load_r2_native_payload(path)
    assert payload["format"] == "fssr-quality-aa-native-v1"
    assert payload["campaign_version"] == "FSSR-QUALITY-AA-v2"
    assert payload["latency_samples"] == 32
    assert payload["resampling"]["linear_delay_samples"] == 32
    assert len(payload["resampling"]["upsample_coefficients"]) == factor * 32 + 1
    actual = R2NativeReference(payload).process(signal.numpy())
    np.testing.assert_allclose(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_aa_nam_native_reference_matches_padded_torch_reset(tmp_path: Path) -> None:
    torch.manual_seed(20260828)
    model = AANAM(aa_mode="off")
    signal = torch.linspace(-0.3, 0.4, 33)
    model.reset_state()
    with torch.inference_mode():
        expected = model.stream(signal).numpy()
    path = export_r2_native_model(model, tmp_path / "aa-nam-off.json")
    payload = load_r2_native_payload(path)
    assert payload["family"] == "aa-nam"
    assert payload["wavenet"]["layer_count"] == 23
    reference = R2NativeReference(payload)
    actual = reference.process(signal.numpy())
    np.testing.assert_allclose(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_ambitious_xl_candidate_is_larger_than_r1_and_exportable(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260828)
    model = AAFSSRXL(aa_mode="full_island_x2")
    assert model.exploratory_only is True
    assert model.candidate_id == "aa-fssr-xl"
    assert model.core_receptive_field == 1 + 3 * ((65 - 1) * 2)
    assert model.processor.branch.core.shapers[0].spline.knots.numel() == 129
    assert model.processor.branch.residual.channels == 16
    # R1 cascade is 17-tap/17-knot with an 8-channel residual; XL is
    # deliberately outside that shape before any deployment gate is applied.
    assert model.parameter_count > 1_537
    signal = torch.linspace(-0.3, 0.4, 97)
    perturbed = signal.clone()
    perturbed[64:] += 0.2
    with torch.inference_mode():
        baseline_prefix = model(signal)[:64]
        perturbed_prefix = model(perturbed)[:64]
    torch.testing.assert_close(baseline_prefix, perturbed_prefix)
    model.reset_state()
    with torch.inference_mode():
        expected = model.stream(signal).numpy()
    path = export_r2_native_model(model, tmp_path / "aa-fssr-xl.json")
    payload = load_r2_native_payload(path)
    assert payload["family"] == "aa-fssr"
    assert len(payload["core"]["filters"][0]["coefficients"]) == 129
    assert len(payload["core"]["shapers"][0]["knots"]) == 129
    reference = R2NativeReference(payload)
    actual = reference.process(signal.numpy())
    np.testing.assert_allclose(actual, expected, atol=2.0e-5, rtol=2.0e-5)
