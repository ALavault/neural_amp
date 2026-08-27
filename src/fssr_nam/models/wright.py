"""Wright et al. causal LSTM baseline and published-model conversion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

_MODEL_KEYS = {
    "model",
    "input_size",
    "skip",
    "output_size",
    "unit_type",
    "num_layers",
    "hidden_size",
    "bias_fl",
}
_STATE_KEYS = {
    "rec.weight_ih_l0",
    "rec.weight_hh_l0",
    "rec.bias_ih_l0",
    "rec.bias_hh_l0",
    "lin.weight",
    "lin.bias",
}

LSTMState = tuple[Tensor, Tensor]


def _batch(signal: Tensor) -> tuple[Tensor, bool]:
    if signal.ndim == 1:
        return signal[None, :, None].contiguous(), True
    if signal.ndim == 2:
        return signal[:, :, None].contiguous(), False
    raise ValueError("signal must have shape (time,) or (batch, time)")


class WrightLSTM(nn.Module):
    """One-layer mono LSTM with the direct skip used by Wright et al.

    ``forward`` always starts from a zero recurrent state. ``stream`` keeps an
    explicit state until ``reset_state`` is called, so arbitrary causal block
    boundaries give the same result as full-sequence inference.
    """

    latency_samples = 0

    def __init__(self, hidden_size: int = 64, sample_rate: int = 44_100) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise TypeError("hidden_size must be an integer")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        self.hidden_size = hidden_size
        self.sample_rate = sample_rate
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1, bias=True)
        self.register_buffer("_stream_hidden", torch.empty(0), persistent=False)
        self.register_buffer("_stream_cell", torch.empty(0), persistent=False)

    @property
    def estimated_macs_per_sample(self) -> int:
        """Dense multiply-accumulates, excluding elementwise gate operations."""
        return 4 * (self.hidden_size + self.hidden_size**2) + self.hidden_size

    def _output(
        self, sequence: Tensor, state: LSTMState | None
    ) -> tuple[Tensor, LSTMState]:
        recurrent, next_state = self.lstm(sequence, state)
        return self.head(recurrent) + sequence, next_state

    def forward_with_state(
        self, signal: Tensor, state: LSTMState | None = None
    ) -> tuple[Tensor, LSTMState]:
        """Run a batch and return explicit state for faithful truncated BPTT."""
        sequence, squeeze = _batch(signal)
        output, next_state = self._output(sequence, state)
        mono = output.squeeze(-1)
        return (mono.squeeze(0) if squeeze else mono), next_state

    def forward(self, signal: Tensor) -> Tensor:
        return self.forward_with_state(signal)[0]

    def reset_state(self) -> None:
        self._stream_hidden = self._stream_hidden.new_empty(0)
        self._stream_cell = self._stream_cell.new_empty(0)

    def stream(self, signal: Tensor) -> Tensor:
        sequence, squeeze = _batch(signal)
        if self._stream_hidden.numel() == 0:
            state = None
        else:
            if self._stream_hidden.shape[1] != sequence.shape[0]:
                raise ValueError("stream batch size changed without reset")
            state = (self._stream_hidden, self._stream_cell)
        output, (hidden, cell) = self._output(sequence, state)
        self._stream_hidden = hidden.detach()
        self._stream_cell = cell.detach()
        mono = output.squeeze(-1)
        return mono.squeeze(0) if squeeze else mono

    @classmethod
    def from_wright_json(cls, source: str | Path | Mapping[str, Any]) -> WrightLSTM:
        """Convert an unmodified published CoreAudioML ``SimpleRNN`` JSON.

        The conversion is deliberately strict: unsupported architectures,
        missing or extra metadata/state keys, non-finite weights, and shape
        mismatches are rejected instead of being silently defaulted.
        """
        document = _read_document(source)
        if set(document) != {"model_data", "state_dict"}:
            raise ValueError("Wright JSON must contain only model_data and state_dict")
        metadata = document["model_data"]
        state = document["state_dict"]
        if not isinstance(metadata, Mapping) or set(metadata) != _MODEL_KEYS:
            raise ValueError("unexpected Wright model_data schema")
        if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
            raise ValueError("unexpected Wright state_dict schema")

        fixed_metadata = {
            "model": "SimpleRNN",
            "input_size": 1,
            "skip": 1,
            "output_size": 1,
            "unit_type": "LSTM",
            "num_layers": 1,
            "bias_fl": True,
        }
        for key, expected in fixed_metadata.items():
            if metadata[key] != expected or type(metadata[key]) is not type(expected):
                raise ValueError(f"unsupported Wright metadata {key}={metadata[key]!r}")
        hidden_size = metadata["hidden_size"]
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size < 1
        ):
            raise ValueError("Wright hidden_size must be a positive integer")

        model = cls(hidden_size=hidden_size, sample_rate=44_100)
        target_state = model.state_dict()
        source_to_target = {
            "rec.weight_ih_l0": "lstm.weight_ih_l0",
            "rec.weight_hh_l0": "lstm.weight_hh_l0",
            "rec.bias_ih_l0": "lstm.bias_ih_l0",
            "rec.bias_hh_l0": "lstm.bias_hh_l0",
            "lin.weight": "head.weight",
            "lin.bias": "head.bias",
        }
        converted: dict[str, Tensor] = {}
        for source_key, target_key in source_to_target.items():
            try:
                tensor = torch.tensor(state[source_key], dtype=torch.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid Wright tensor {source_key}") from error
            if tensor.shape != target_state[target_key].shape:
                raise ValueError(
                    f"Wright tensor {source_key} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(target_state[target_key].shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Wright tensor {source_key} must be finite")
            converted[target_key] = tensor
        model.load_state_dict(converted, strict=True)
        return model


def _read_document(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Wright JSON: {path}") from error
    if not isinstance(document, Mapping):
        raise ValueError("Wright JSON root must be an object")
    return document


def load_wright_json(source: str | Path | Mapping[str, Any]) -> WrightLSTM:
    """Functional alias for strict published-model conversion."""
    return WrightLSTM.from_wright_json(source)
