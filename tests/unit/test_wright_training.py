import numpy as np
import pytest
import torch

from fssr_nam.losses import WrightLoss
from fssr_nam.models import WrightLSTM
from fssr_nam.training.wright import (
    evaluate_prediction,
    frame_audio,
    predict_streaming,
    train_epoch,
)


def test_frame_audio_drops_only_incomplete_tail() -> None:
    framed = frame_audio(np.arange(11, dtype=np.float32), 5)
    assert framed.shape == (2, 5)
    torch.testing.assert_close(framed[-1], torch.arange(5, 10, dtype=torch.float32))


def test_tiny_wright_epoch_is_finite_and_updates_parameters() -> None:
    torch.manual_seed(3)
    signal = np.linspace(-0.2, 0.2, 64, dtype=np.float32)
    frames = frame_audio(signal, 16)
    model = WrightLSTM(hidden_size=4)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss, updates = train_epoch(
        model,
        frames,
        torch.tanh(2.0 * frames),
        WrightLoss(),
        optimizer,
        batch_size=2,
        warmup_samples=2,
        tbptt_samples=7,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(9),
    )
    assert np.isfinite(loss)
    assert updates == 4
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def test_stream_prediction_and_evaluation_are_block_invariant() -> None:
    rng = np.random.default_rng(10)
    signal = (rng.standard_normal(131) * 0.05).astype(np.float32)
    model = WrightLSTM(hidden_size=5)
    expected = predict_streaming(
        model, signal, device=torch.device("cpu"), chunk_samples=17
    )
    alternate = predict_streaming(
        model, signal, device=torch.device("cpu"), chunk_samples=43
    )
    np.testing.assert_allclose(alternate, expected, rtol=2.0e-6, atol=2.0e-6)
    reference_loss, esr = evaluate_prediction(expected, signal, WrightLoss())
    assert np.isfinite(reference_loss)
    assert np.isfinite(esr)


def test_long_uneven_prediction_is_invariant_at_frozen_chunk() -> None:
    signal = np.linspace(-0.1, 0.1, 2 * 32_768 + 17, dtype=np.float32)
    model = WrightLSTM(hidden_size=4)
    expected = predict_streaming(
        model, signal, device=torch.device("cpu"), chunk_samples=32_768
    )
    alternate = predict_streaming(
        model, signal, device=torch.device("cpu"), chunk_samples=4_093
    )
    assert expected.shape == signal.shape
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(alternate, expected, rtol=2.0e-5, atol=2.0e-5)


def test_stream_state_resets_when_prediction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = WrightLSTM(hidden_size=4)
    original_stream = model.stream
    calls = 0

    def fail_second_chunk(chunk: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected streaming failure")
        return original_stream(chunk)

    monkeypatch.setattr(model, "stream", fail_second_chunk)
    with pytest.raises(RuntimeError, match="injected streaming failure"):
        predict_streaming(
            model,
            np.linspace(-0.1, 0.1, 10, dtype=np.float32),
            device=torch.device("cpu"),
            chunk_samples=5,
        )
    assert model._stream_hidden.numel() == 0
    assert model._stream_cell.numel() == 0
