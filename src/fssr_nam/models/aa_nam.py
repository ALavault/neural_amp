"""Pinned NAM A2 topology with the prospective R2 antialiasing routes."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from nam.models.factory import init as init_nam_model
from torch import Tensor, nn

from .oversampling import FullRateIsland
from .r2 import (
    AA_MODES,
    replace_tanh_with_hermite,
    scale_a2_config_for_internal_rate,
)
from .structured import CausalDelay, _batch

_PACKED_CONFIG = Path(
    "third_party/neural-amp-modeler/nam/train/_resources/config_model_packed.json"
)
_A2_SUBMODEL_INDEX = 1
_A2_ACTIVATIONS = 23
_A2_RECEPTIVE_FIELD = 6347


def prepare_a2_config(config: dict[str, Any], factor: int) -> dict[str, Any]:
    """Resolve A2 with Tanh layers and an internal-rate-preserving horizon."""
    resolved = deepcopy(config)

    def set_tanh(value: Any) -> None:
        if isinstance(value, dict):
            if "activation" in value:
                value["activation"] = "Tanh"
            for child in value.values():
                set_tanh(child)
        elif isinstance(value, list):
            for child in value:
                set_tanh(child)

    set_tanh(resolved)
    return scale_a2_config_for_internal_rate(resolved, factor)


def load_pinned_a2_config(root: Path | None = None) -> dict[str, Any]:
    """Read the repository-pinned official packed A2 declaration."""
    path = (root or Path.cwd()) / _PACKED_CONFIG
    payload = json.loads(path.read_text(encoding="utf-8"))
    net = payload.get("net")
    if not isinstance(net, dict) or net.get("name") != "PackedWaveNet":
        raise ValueError("pinned A2 config is not the expected PackedWaveNet")
    return net


class NAMContextAdapter(nn.Module):
    """Give the stateless official WaveNet exact causal block state."""

    def __init__(self, model: nn.Module, *, extra_history: int = 0) -> None:
        super().__init__()
        receptive_field = int(getattr(model, "receptive_field", 0))
        if receptive_field < 1 or extra_history < 0:
            raise ValueError("NAM context declaration is invalid")
        self.model = model
        self.receptive_field = receptive_field + extra_history
        self._context: Tensor | None = None

    @property
    def history(self) -> int:
        return self.receptive_field - 1

    def reset_state(self) -> None:
        self._context = None
        for module in self.model.modules():
            if module is self.model:
                continue
            reset = getattr(module, "reset_state", None)
            if reset is not None:
                reset()

    def forward(self, signal: Tensor) -> Tensor:
        batched, scalar = _batch(signal)
        output = self.model(batched, pad_start=True)
        return output[0] if scalar else output

    def stream(self, signal: Tensor) -> Tensor:
        batched, scalar = _batch(signal)
        if self._context is None:
            self._context = batched.new_zeros((len(batched), self.history))
        if self._context.shape[0] != batched.shape[0]:
            raise ValueError("stream batch size changed without reset")
        joined = torch.cat((self._context, batched), dim=-1)
        output = self.model(joined, pad_start=False)[..., -batched.shape[-1] :]
        if output.shape != batched.shape:
            raise RuntimeError("official NAM topology did not preserve block shape")
        self._context = joined[..., -self.history :].detach()
        return output[0] if scalar else output


class AANAM(nn.Module):
    """AA-NAM using the pinned A2 Full topology and R2 Hermite activations."""

    sample_rate_hz = 48_000

    def __init__(self, *, aa_mode: str, root: Path | None = None) -> None:
        super().__init__()
        if aa_mode not in AA_MODES:
            raise ValueError(f"aa_mode must be one of {AA_MODES}")
        factor = {
            "off": 1,
            "adaa1": 1,
            "full_island_x2": 2,
            "teacher_x4": 4,
        }[aa_mode]
        declaration = load_pinned_a2_config(root)
        packed_config = prepare_a2_config(declaration["config"], factor)
        packed = init_nam_model(declaration["name"], kwargs={"config": packed_config})
        topology = packed.extract_submodel(_A2_SUBMODEL_INDEX)
        replaced = replace_tanh_with_hermite(topology, adaa=aa_mode == "adaa1")
        if replaced != _A2_ACTIVATIONS:
            raise RuntimeError(
                f"expected {_A2_ACTIVATIONS} A2 activations, replaced {replaced}"
            )
        expected_rf = factor * (_A2_RECEPTIVE_FIELD - 1) + 1
        if int(topology.receptive_field) != expected_rf:
            raise RuntimeError(
                "scaled A2 receptive field does not preserve its horizon"
            )
        extra_history = _A2_ACTIVATIONS if aa_mode == "adaa1" else 0
        branch = NAMContextAdapter(topology, extra_history=extra_history)
        self.processor = FullRateIsland(branch, factor=factor) if factor > 1 else branch
        self.output_delay = CausalDelay(1) if aa_mode == "adaa1" else None
        self.aa_mode = aa_mode
        self.internal_sample_rate = self.sample_rate_hz * factor
        self.dilation_scale = factor
        self.latency_samples = 16 if factor > 1 else int(aa_mode == "adaa1")
        self.activation_count = replaced
        self.topology_receptive_field = int(topology.receptive_field)

    def reset_state(self) -> None:
        self.processor.reset_state()
        if self.output_delay is not None:
            self.output_delay.reset_state()

    def forward(self, signal: Tensor) -> Tensor:
        output = self.processor(signal)
        return self.output_delay(output) if self.output_delay is not None else output

    def stream(self, signal: Tensor) -> Tensor:
        output = self.processor.stream(signal)
        return (
            self.output_delay.stream(output)
            if self.output_delay is not None
            else output
        )
