import json

import pytest
import torch

from fssr_nam.losses import WrightLoss, dc_loss, esr_loss, preemphasize
from fssr_nam.models import WrightLSTM


def test_wright_loss_matches_hand_calculation_and_is_finite() -> None:
    output = torch.tensor([[0.1, 0.0, -0.2], [0.2, 0.3, 0.1]])
    target = torch.tensor([[0.0, 0.1, -0.1], [0.1, 0.2, 0.4]])
    expected_filter = torch.tensor([[0.0, 0.1, -0.185], [0.1, 0.115, 0.23]])
    torch.testing.assert_close(preemphasize(target), expected_filter)
    expected = 0.75 * esr_loss(preemphasize(output), expected_filter) + 0.25 * dc_loss(
        output, target
    )
    actual = WrightLoss()(output, target)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual)
    assert WrightLoss()(target, target) == 0.0


def test_wright_lstm_streaming_matches_full_sequence_and_reset() -> None:
    torch.manual_seed(4)
    model = WrightLSTM(hidden_size=8)
    signal = torch.randn(2, 129) * 0.1
    expected = model(signal)
    streamed = torch.cat(
        (
            model.stream(signal[:, :7]),
            model.stream(signal[:, 7:80]),
            model.stream(signal[:, 80:]),
        ),
        dim=-1,
    )
    torch.testing.assert_close(streamed, expected, atol=2.0e-6, rtol=2.0e-6)
    model.reset_state()
    torch.testing.assert_close(model.stream(signal), expected, atol=2.0e-6, rtol=2.0e-6)


def test_reference_json_loader_preserves_state(tmp_path) -> None:
    torch.manual_seed(8)
    reference = WrightLSTM()
    payload = {
        "model_data": {
            "model": "SimpleRNN",
            "input_size": 1,
            "skip": 1,
            "output_size": 1,
            "unit_type": "LSTM",
            "num_layers": 1,
            "hidden_size": 64,
            "bias_fl": True,
        },
        "state_dict": {
            "rec.weight_ih_l0": reference.lstm.weight_ih_l0.detach().tolist(),
            "rec.weight_hh_l0": reference.lstm.weight_hh_l0.detach().tolist(),
            "rec.bias_ih_l0": reference.lstm.bias_ih_l0.detach().tolist(),
            "rec.bias_hh_l0": reference.lstm.bias_hh_l0.detach().tolist(),
            "lin.weight": reference.head.weight.detach().tolist(),
            "lin.bias": reference.head.bias.detach().tolist(),
        },
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = WrightLSTM.from_wright_json(path)
    signal = torch.randn(257) * 0.1
    torch.testing.assert_close(loaded(signal), reference(signal))


def test_wright_interfaces_reject_invalid_shapes() -> None:
    model = WrightLSTM()
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1, 1, 1))
    with pytest.raises(ValueError, match="shapes differ"):
        esr_loss(torch.zeros(2), torch.zeros(3))
