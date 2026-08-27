import json
from pathlib import Path

import pytest
import torch
from torch import nn

from fssr_nam.models.wright import WrightLSTM, load_wright_json

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_MODEL = (
    ROOT / "third_party/Automated-GuitarAmpModelling/Results/muff-muff2/model_best.json"
)


def _published_reference(payload: dict, signal: torch.Tensor) -> torch.Tensor:
    hidden_size = payload["model_data"]["hidden_size"]
    recurrent = nn.LSTM(1, hidden_size, batch_first=True)
    head = nn.Linear(hidden_size, 1)
    state = payload["state_dict"]
    recurrent.load_state_dict(
        {
            "weight_ih_l0": torch.tensor(state["rec.weight_ih_l0"]),
            "weight_hh_l0": torch.tensor(state["rec.weight_hh_l0"]),
            "bias_ih_l0": torch.tensor(state["rec.bias_ih_l0"]),
            "bias_hh_l0": torch.tensor(state["rec.bias_hh_l0"]),
        },
        strict=True,
    )
    head.load_state_dict(
        {
            "weight": torch.tensor(state["lin.weight"]),
            "bias": torch.tensor(state["lin.bias"]),
        },
        strict=True,
    )
    sequence = signal[None, :, None]
    hidden, _ = recurrent(sequence)
    return (head(hidden) + sequence)[0, :, 0]


def test_published_wright_lstm64_conversion_is_exact() -> None:
    payload = json.loads(PUBLISHED_MODEL.read_text(encoding="utf-8"))
    model = load_wright_json(PUBLISHED_MODEL)
    assert model.hidden_size == 64
    assert model.sample_rate == 44_100
    signal = torch.linspace(-0.25, 0.25, 311)
    expected = _published_reference(payload, signal)
    actual = model(signal)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.shape == signal.shape
    assert torch.isfinite(actual).all()


def test_published_wright_stream_is_causal_and_block_invariant() -> None:
    model = WrightLSTM.from_wright_json(PUBLISHED_MODEL)
    generator = torch.Generator().manual_seed(20260827)
    signal = 0.1 * torch.randn(277, generator=generator)
    expected = model(signal)
    blocks = (1, 17, 3, 64, 2, 89, 101)
    start = 0
    streamed = []
    for size in blocks:
        streamed.append(model.stream(signal[start : start + size]))
        start += size
    torch.testing.assert_close(torch.cat(streamed), expected, rtol=0.0, atol=0.0)

    model.reset_state()
    torch.testing.assert_close(model.stream(signal), expected, rtol=0.0, atol=0.0)
    changed = signal.clone()
    changed[173:] = torch.flip(changed[173:], dims=(0,))
    torch.testing.assert_close(model(changed)[:173], expected[:173], rtol=0.0, atol=0.0)


def test_wright_conversion_rejects_schema_drift() -> None:
    payload = json.loads(PUBLISHED_MODEL.read_text(encoding="utf-8"))
    payload["model_data"]["unit_type"] = "GRU"
    with pytest.raises(ValueError, match="unit_type"):
        load_wright_json(payload)

    payload = json.loads(PUBLISHED_MODEL.read_text(encoding="utf-8"))
    payload["state_dict"].pop("lin.bias")
    with pytest.raises(ValueError, match="state_dict schema"):
        load_wright_json(payload)
