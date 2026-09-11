from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from fssr_nam.models import (
    AANAM,
    FirstOrderADAA,
    HermiteCustomTanh,
    synchronize_aa_weights,
)
from fssr_nam.models.aa_nam import NAMContextAdapter, prepare_a2_config


class _ThreeTap(nn.Module):
    receptive_field = 3

    def forward(self, signal: torch.Tensor, *, pad_start: bool) -> torch.Tensor:
        if pad_start:
            signal = torch.nn.functional.pad(signal, (2, 0))
        return torch.nn.functional.conv1d(
            signal[:, None], signal.new_tensor([[[0.2, 0.3, 0.5]]])
        )[:, 0]


def test_nam_context_adapter_matches_irregular_blocks_and_reset() -> None:
    generator = torch.Generator().manual_seed(20260828)
    signal = torch.randn(2, 41, generator=generator)
    adapter = NAMContextAdapter(_ThreeTap())
    expected = adapter(signal)
    actual = torch.cat(
        (
            adapter.stream(signal[:, :7]),
            adapter.stream(signal[:, 7:19]),
            adapter.stream(signal[:, 19:]),
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=1.0e-7, rtol=1.0e-7)
    adapter.reset_state()
    torch.testing.assert_close(adapter.stream(signal), expected)


def test_prepare_a2_config_forces_tanh_and_preserves_physical_rf() -> None:
    path = Path(
        "third_party/neural-amp-modeler/nam/train/_resources/config_model_packed.json"
    )
    source = json.loads(path.read_text(encoding="utf-8"))["net"]["config"]
    resolved = prepare_a2_config(source, 4)
    layer = resolved["submodels"][1]["config"]["layers_configs"][0]
    original = source["submodels"][1]["config"]["layers_configs"][0]
    assert layer["activation"] == "Tanh"
    assert original["activation"] == "LeakyReLU"
    assert layer["dilations"] == [4 * value for value in original["dilations"]]
    assert layer["head"]["kernel_size"] == 61


@pytest.mark.parametrize(
    ("mode", "factor", "latency", "activation_type"),
    (
        ("off", 1, 0, HermiteCustomTanh),
        ("adaa1", 1, 1, FirstOrderADAA),
        ("full_island_x2", 2, 16, HermiteCustomTanh),
        ("teacher_x4", 4, 16, HermiteCustomTanh),
    ),
)
def test_aa_nam_uses_pinned_a2_topology(
    mode: str,
    factor: int,
    latency: int,
    activation_type: type[nn.Module],
) -> None:
    model = AANAM(aa_mode=mode)
    assert model.activation_count == 23
    assert model.internal_sample_rate == 48_000 * factor
    assert model.topology_receptive_field == factor * (6347 - 1) + 1
    assert model.latency_samples == latency
    assert sum(isinstance(module, activation_type) for module in model.modules()) == 23
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_aa_nam_same_weight_transfer_dilates_output_head() -> None:
    torch.manual_seed(20260828)
    source = AANAM(aa_mode="off")
    target = AANAM(aa_mode="full_island_x2")
    synchronize_aa_weights(source, (target,))
    source_head = source.processor.model._net._layer_arrays[0]._head_rechannel.weight
    target_head = target.processor.branch.model._net._layer_arrays[
        0
    ]._head_rechannel.weight
    torch.testing.assert_close(target_head[..., ::2], source_head)
    assert torch.count_nonzero(target_head[..., 1::2]) == 0
