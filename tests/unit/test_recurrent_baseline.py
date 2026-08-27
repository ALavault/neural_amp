import torch

from fssr_nam.models.recurrent import CausalGRUBaseline


def test_gru_full_matches_a2_parameter_budget() -> None:
    model = CausalGRUBaseline(hidden_size=62)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 12_153
    assert model.estimated_macs_per_sample == 11_780
    assert abs(parameters - 12_145) / 12_145 < 0.001


def test_gru_streaming_matches_complete_sequence_and_reset() -> None:
    torch.manual_seed(4)
    model = CausalGRUBaseline(hidden_size=8).eval()
    signal = torch.randn(257)
    with torch.inference_mode():
        complete = model(signal)
        model.reset_state()
        blocked = torch.cat(
            [
                model.stream(signal[:31]),
                model.stream(signal[31:96]),
                model.stream(signal[96:]),
            ]
        )
        model.reset_state()
        reset = model.stream(signal)
    assert torch.allclose(blocked, complete, atol=2.0e-7, rtol=1.0e-6)
    assert torch.equal(reset, complete)


def test_gru_is_causal() -> None:
    torch.manual_seed(5)
    model = CausalGRUBaseline(hidden_size=8).eval()
    signal = torch.randn(128)
    changed = signal.clone()
    changed[64:] = torch.randn(64)
    with torch.inference_mode():
        original = model(signal)
        altered = model(changed)
    assert torch.equal(original[:64], altered[:64])
