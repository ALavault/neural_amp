"""Versioned export and NumPy reference inference for FSSR-R2 models."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fssr_nam.inference.r1_native import (
    _array,
    _CausalFir,
    _extract_core,
    _extract_residual,
    _extract_slow,
    _finite_matrix,
    _finite_vector,
    _ResidualLayer,
    _shaper_payload,
    _SlowController,
)

FORMAT = "fssr-r2-native-v1"
CAMPAIGN_VERSION = "FSSR-R2-v1"
BASE_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
AA_FACTORS = {"off": 1, "adaa1": 1, "full_island_x2": 2, "teacher_x4": 4}


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _validate_core(core: object, aa_mode: str) -> None:
    if not isinstance(core, Mapping) or core.get("kind") not in {"mono", "cascade"}:
        raise ValueError("R2 core.kind must be mono or cascade")
    count = 1 if core["kind"] == "mono" else 2
    filters = core.get("filters")
    shapers = core.get("shapers")
    if not isinstance(filters, list) or len(filters) != count + 1:
        raise ValueError("R2 core filter count does not match its kind")
    if not isinstance(shapers, list) or len(shapers) != count:
        raise ValueError("R2 core shaper count does not match its kind")
    for index, fir in enumerate(filters):
        if not isinstance(fir, Mapping):
            raise ValueError(f"R2 core filter {index} must be an object")
        coefficients = _finite_vector(fir.get("coefficients"), "FIR coefficients")
        if not coefficients:
            raise ValueError("R2 core FIRs cannot be empty")
    for index, shaper in enumerate(shapers):
        if not isinstance(shaper, Mapping):
            raise ValueError(f"R2 shaper {index} must be an object")
        knots = _finite_vector(shaper.get("knots"), "spline knots")
        values = _finite_vector(shaper.get("values"), "spline values")
        slopes = _finite_vector(shaper.get("slopes"), "spline slopes")
        if len(knots) < 4 or len(values) != len(knots) or len(slopes) != len(knots):
            raise ValueError("R2 spline arrays have incompatible dimensions")
        if any(right <= left for left, right in pairwise(knots)):
            raise ValueError("R2 spline knots must be strictly increasing")
        for scalar in ("drive", "offset"):
            if not math.isfinite(float(shaper.get(scalar, math.nan))):
                raise ValueError(f"R2 shaper {scalar} must be finite")
        uses_adaa = shaper.get("adaa1") is True
        if uses_adaa != (aa_mode == "adaa1"):
            raise ValueError("R2 shaper ADAA declaration differs from aa_mode")
        if uses_adaa:
            threshold = float(shaper.get("difference_limit_threshold", math.nan))
            if threshold != 1.0e-4:
                raise ValueError("R2 ADAA difference limit must be exactly 1e-4")
    if not math.isfinite(float(core.get("output_gain", math.nan))):
        raise ValueError("R2 core output gain must be finite")


def _validate_slow(slow: object, core_kind: str, factor: int) -> None:
    if not isinstance(slow, Mapping):
        raise ValueError("R2 slow controller is required")
    if slow.get("kind") != "fssr-slow-gru-v1":
        raise ValueError("R2 slow controller kind is unsupported")
    if slow.get("hidden_size") != 16 or slow.get("decimation") != 64 * factor:
        raise ValueError("R2 slow GRU size/cadence differs from the protocol")
    expected_application = (
        "first-shaper-drive-offset-and-output-gain"
        if core_kind == "cascade"
        else "shaper-drive-offset-and-output-gain"
    )
    if slow.get("application") != expected_application:
        raise ValueError("R2 slow controller application is invalid")
    _finite_matrix(slow.get("weight_ih"), "slow weight_ih", 48, 3)
    _finite_matrix(slow.get("weight_hh"), "slow weight_hh", 48, 16)
    _finite_vector(slow.get("bias_ih"), "slow bias_ih", length=48)
    _finite_vector(slow.get("bias_hh"), "slow bias_hh", length=48)
    _finite_matrix(slow.get("projection_weight"), "slow projection", 3, 16)
    _finite_vector(slow.get("projection_bias"), "slow projection bias", length=3)


def _validate_residual(residual: object, factor: int) -> None:
    if not isinstance(residual, Mapping):
        raise ValueError("R2 RF2047 residual is required")
    if residual.get("kind") != "causal-depthwise-separable-tcn":
        raise ValueError("R2 residual must be depthwise separable")
    if residual.get("channels") != 16 or residual.get("kernel_size") != 3:
        raise ValueError("R2 residual must use 16 channels and kernel size three")
    dilations = tuple(residual.get("dilations", ()))
    expected = tuple(factor * value for value in BASE_DILATIONS)
    if dilations != expected:
        raise ValueError("R2 residual dilations do not preserve the physical RF")
    if residual.get("receptive_field") != 1 + 2 * sum(expected):
        raise ValueError("R2 residual receptive field is inconsistent")
    _finite_matrix(
        residual.get("input_projection", {}).get("weight"),
        "R2 residual input weight",
        16,
        2,
    )
    _finite_vector(
        residual.get("input_projection", {}).get("bias"),
        "R2 residual input bias",
        length=16,
    )
    layers = residual.get("layers")
    if not isinstance(layers, list) or len(layers) != len(expected):
        raise ValueError("R2 residual requires one layer per dilation")
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ValueError(f"R2 residual layer {index} must be an object")
        _finite_matrix(layer.get("depthwise_weight"), "depthwise weight", 16, 3)
        _finite_vector(layer.get("depthwise_bias"), "depthwise bias", length=16)
        _finite_matrix(layer.get("pointwise_weight"), "pointwise weight", 16, 16)
        _finite_vector(layer.get("pointwise_bias"), "pointwise bias", length=16)
    _finite_vector(
        residual.get("output_projection", {}).get("weight"),
        "R2 residual output weight",
        length=16,
    )
    output_bias = float(residual.get("output_projection", {}).get("bias", math.nan))
    if not math.isfinite(output_bias):
        raise ValueError("R2 residual output bias must be finite")


def validate_r2_native_payload(payload: Mapping[str, Any]) -> None:
    """Validate R2 topology, AA metadata, dimensions, RF, and memory sizes."""
    required = {
        "format": FORMAT,
        "version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "precision": "float32",
        "sample_rate_hz": 48_000,
    }
    for name, expected in required.items():
        if payload.get(name) != expected:
            raise ValueError(f"{name} must equal {expected!r}")
    aa_mode = payload.get("aa_mode")
    if aa_mode not in AA_FACTORS:
        raise ValueError("R2 aa_mode is unsupported")
    factor = AA_FACTORS[str(aa_mode)]
    if payload.get("internal_sample_rate") != 48_000 * factor:
        raise ValueError("R2 internal_sample_rate differs from aa_mode")
    if payload.get("dilation_scale") != factor:
        raise ValueError("R2 dilation_scale differs from aa_mode")
    expected_latency = 16 if factor > 1 else 1 if aa_mode == "adaa1" else 0
    if payload.get("latency_samples") != expected_latency:
        raise ValueError("R2 latency_samples differs from aa_mode")
    family = payload.get("family")
    if family == "aa-nam":
        _validate_aa_nam(payload.get("wavenet"), str(aa_mode), factor)
    elif family != "aa-fssr":
        raise ValueError("R2 family must be aa-nam or aa-fssr")
    if family == "aa-nam":
        if any(
            payload.get(name) is not None
            for name in ("core", "slow_controller", "residual")
        ):
            raise ValueError("AA-NAM cannot contain AA-FSSR topology fields")
    else:
        core = payload.get("core")
        _validate_core(core, str(aa_mode))
        assert isinstance(core, Mapping)
        _validate_slow(payload.get("slow_controller"), str(core["kind"]), factor)
        _validate_residual(payload.get("residual"), factor)
    resampling = payload.get("resampling")
    if factor == 1:
        if resampling is not None:
            raise ValueError("base-rate R2 modes cannot declare resampling")
    else:
        if not isinstance(resampling, Mapping) or resampling.get("factor") != factor:
            raise ValueError("R2 full island requires its resampling declaration")
        if resampling.get("linear_delay_samples") != 16:
            raise ValueError("R2 full island linear delay must be 16 samples")
        taps = factor * 16 + 1
        for name in ("upsample_coefficients", "downsample_coefficients"):
            _finite_vector(resampling.get(name), name, length=taps)
    sizes = payload.get("sizes")
    if not isinstance(sizes, Mapping):
        raise ValueError("R2 export sizes must be an object")
    for name in (
        "parameters",
        "weight_bytes",
        "persistent_state_bytes",
        "scratch_bytes",
    ):
        _integer(sizes.get(name), f"sizes.{name}")


def _validate_aa_nam(wavenet: object, aa_mode: str, factor: int) -> None:
    if not isinstance(wavenet, Mapping):
        raise ValueError("AA-NAM wavenet declaration is required")
    if wavenet.get("kind") != "pinned-a2-full-hermite-v1":
        raise ValueError("AA-NAM wavenet kind is unsupported")
    if wavenet.get("channels") != 8 or wavenet.get("layer_count") != 23:
        raise ValueError("AA-NAM must preserve A2 Full's 23x8 topology")
    expected_rf = factor * (6347 - 1) + 1
    if wavenet.get("receptive_field") != expected_rf:
        raise ValueError("AA-NAM receptive field does not preserve its horizon")
    _finite_matrix(wavenet.get("input_projection"), "AA-NAM input projection", 8, 1)
    layers = wavenet.get("layers")
    if not isinstance(layers, list) or len(layers) != 23:
        raise ValueError("AA-NAM requires 23 layers")
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ValueError(f"AA-NAM layer {index} must be an object")
        kernel_size = _integer(
            layer.get("kernel_size"), "AA-NAM kernel size", minimum=1
        )
        dilation = _integer(layer.get("dilation"), "AA-NAM dilation", minimum=1)
        if dilation % factor:
            raise ValueError("AA-NAM dilation must be scaled by the internal factor")
        _finite_matrix(
            layer.get("conv_weight"), "AA-NAM convolution", 8, 8 * kernel_size
        )
        _finite_vector(layer.get("conv_bias"), "AA-NAM convolution bias", length=8)
        _finite_vector(
            layer.get("condition_weight"), "AA-NAM condition weight", length=8
        )
        _finite_matrix(layer.get("residual_weight"), "AA-NAM residual weight", 8, 8)
        _finite_vector(layer.get("residual_bias"), "AA-NAM residual bias", length=8)
        activation = layer.get("activation")
        _validate_aa_nam_activation(activation, aa_mode)
    head = wavenet.get("head")
    if not isinstance(head, Mapping):
        raise ValueError("AA-NAM head is required")
    expected_head = factor * 15 + 1
    if head.get("kernel_size") != expected_head:
        raise ValueError("AA-NAM head horizon is inconsistent")
    _finite_matrix(head.get("weight"), "AA-NAM head weight", 8, expected_head)
    if not math.isfinite(float(head.get("bias", math.nan))):
        raise ValueError("AA-NAM head bias must be finite")
    if not math.isfinite(float(wavenet.get("head_scale", math.nan))):
        raise ValueError("AA-NAM head scale must be finite")


def _validate_aa_nam_activation(activation: object, aa_mode: str) -> None:
    if not isinstance(activation, Mapping):
        raise ValueError("AA-NAM Hermite activation is required")
    for name in ("knots", "values", "slopes"):
        values = _finite_vector(activation.get(name), f"AA-NAM activation {name}")
        if len(values) != 33:
            raise ValueError("AA-NAM activation must contain 33 Hermite knots")
    uses_adaa = activation.get("adaa1") is True
    if uses_adaa != (aa_mode == "adaa1"):
        raise ValueError("AA-NAM activation ADAA declaration differs from aa_mode")
    if uses_adaa and activation.get("difference_limit_threshold") != 1.0e-4:
        raise ValueError("AA-NAM ADAA difference limit must be exactly 1e-4")


def _model_sizes(model: Any, payload: Mapping[str, Any]) -> dict[str, int]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    weight_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    )
    pointer_bytes = np.dtype(np.uintp).itemsize
    factor = int(payload["dilation_scale"])
    if payload["family"] == "aa-nam":
        wavenet = payload["wavenet"]
        persistent = sum(
            8 * (int(layer["kernel_size"]) - 1) * int(layer["dilation"]) * 4
            + pointer_bytes
            for layer in wavenet["layers"]
        )
        persistent += 8 * (int(wavenet["head"]["kernel_size"]) - 1) * 4
        if payload["aa_mode"] == "adaa1":
            persistent += 23 * 8 * 4
            persistent += 4 + pointer_bytes
        if factor > 1:
            taps = factor * 16 + 1
            persistent += 2 * ((taps - 1) * 4 + pointer_bytes)
            persistent += 16 * 4 + pointer_bytes
        return {
            "parameters": int(parameters),
            "weight_bytes": int(weight_bytes),
            "persistent_state_bytes": int(persistent),
            "scratch_bytes": 6 * 8 * 4,
        }
    core = payload["core"]
    persistent = sum(
        (len(fir["coefficients"]) - 1) * 4 + pointer_bytes for fir in core["filters"]
    )
    persistent += 4 * sum(shaper["adaa1"] for shaper in core["shapers"])
    if payload["aa_mode"] == "adaa1":
        persistent += 4 + pointer_bytes
    persistent += (16 + 3) * 4 + pointer_bytes
    residual = payload["residual"]
    persistent += sum(
        16 * 2 * dilation * 4 + pointer_bytes for dilation in residual["dilations"]
    )
    if factor > 1:
        taps = factor * 16 + 1
        persistent += 2 * ((taps - 1) * 4 + pointer_bytes)
        persistent += 16 * 4 + pointer_bytes
    scratch = 16 * 4 + len(residual["layers"]) * 2 * 16 * 4
    return {
        "parameters": int(parameters),
        "weight_bytes": int(weight_bytes),
        "persistent_state_bytes": int(persistent),
        "scratch_bytes": int(scratch),
    }


def _resampling_payload(processor: Any, factor: int) -> dict[str, Any] | None:
    if factor == 1:
        return None
    return {
        "factor": factor,
        "linear_delay_samples": 16,
        "upsample_coefficients": _array(
            processor.upsample_filter.coefficients, "upsample coefficients"
        ).tolist(),
        "downsample_coefficients": _array(
            processor.downsample_filter.coefficients, "downsample coefficients"
        ).tolist(),
    }


def _extract_aa_nam(processor: Any, aa_mode: str) -> dict[str, Any]:
    branch = getattr(processor, "branch", processor)
    topology = branch.model
    net = topology._net
    arrays = list(net._layer_arrays)
    if len(arrays) != 1:
        raise TypeError("AA-NAM export requires the pinned single-array A2 topology")
    array = arrays[0]
    layers = []
    for index, layer in enumerate(array._layers):
        conv = _array(layer._conv.weight, f"AA-NAM layer {index} conv")
        activation = _shaper_payload(layer, layer._activation, index)
        activation["adaa1"] = aa_mode == "adaa1"
        activation["difference_limit_threshold"] = (
            1.0e-4 if aa_mode == "adaa1" else None
        )
        layers.append(
            {
                "kernel_size": int(layer._conv.kernel_size[0]),
                "dilation": int(layer._conv.dilation[0]),
                "conv_weight": conv.reshape(8, -1).tolist(),
                "conv_bias": _array(
                    layer._conv.bias, f"AA-NAM layer {index} conv bias"
                ).tolist(),
                "condition_weight": _array(
                    layer._input_mixer.weight,
                    f"AA-NAM layer {index} condition",
                )
                .reshape(8)
                .tolist(),
                "activation": activation,
                "residual_weight": _array(
                    layer._layer1x1.weight,
                    f"AA-NAM layer {index} residual",
                )
                .reshape(8, 8)
                .tolist(),
                "residual_bias": _array(
                    layer._layer1x1.bias,
                    f"AA-NAM layer {index} residual bias",
                ).tolist(),
            }
        )
    head = array._head_rechannel
    return {
        "kind": "pinned-a2-full-hermite-v1",
        "channels": 8,
        "layer_count": 23,
        "receptive_field": int(topology.receptive_field),
        "input_projection": _array(array._rechannel.weight, "AA-NAM input projection")
        .reshape(8, 1)
        .tolist(),
        "layers": layers,
        "head": {
            "kernel_size": int(head.kernel_size[0]),
            "weight": _array(head.weight, "AA-NAM head weight")
            .reshape(8, int(head.kernel_size[0]))
            .tolist(),
            "bias": float(_array(head.bias, "AA-NAM head bias").reshape(())),
        },
        "head_scale": float(net._head_scale),
    }


def build_r2_native_payload(model: Any) -> dict[str, Any]:
    """Convert an AA-FSSR or AA-NAM model to the frozen R2 native format."""
    aa_mode = getattr(model, "aa_mode", None)
    if aa_mode not in AA_FACTORS:
        raise TypeError("R2 native export requires an R2 AA model")
    factor = AA_FACTORS[aa_mode]
    processor = model.processor
    is_aa_nam = hasattr(getattr(processor, "branch", processor), "model")
    if is_aa_nam:
        payload: dict[str, Any] = {
            "format": FORMAT,
            "version": 1,
            "campaign_version": CAMPAIGN_VERSION,
            "precision": "float32",
            "sample_rate_hz": 48_000,
            "family": "aa-nam",
            "aa_mode": aa_mode,
            "internal_sample_rate": 48_000 * factor,
            "dilation_scale": factor,
            "latency_samples": int(model.latency_samples),
            "wavenet": _extract_aa_nam(processor, aa_mode),
            "core": None,
            "slow_controller": None,
            "residual": None,
            "resampling": _resampling_payload(processor, factor),
        }
        payload["sizes"] = _model_sizes(model, payload)
        validate_r2_native_payload(payload)
        return payload
    branch = getattr(processor, "branch", processor)
    core = _extract_core(branch)
    for shaper in core["shapers"]:
        shaper["adaa1"] = aa_mode == "adaa1"
        shaper["difference_limit_threshold"] = 1.0e-4 if aa_mode == "adaa1" else None
    payload: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "precision": "float32",
        "sample_rate_hz": 48_000,
        "family": "aa-fssr",
        "aa_mode": aa_mode,
        "internal_sample_rate": 48_000 * factor,
        "dilation_scale": factor,
        "latency_samples": int(model.latency_samples),
        "wavenet": None,
        "core": core,
        "slow_controller": _extract_slow(branch, core["kind"]),
        "residual": _extract_residual(branch, allowed_channels=(16,)),
        "resampling": _resampling_payload(processor, factor),
    }
    payload["sizes"] = _model_sizes(model, payload)
    validate_r2_native_payload(payload)
    return payload


def write_r2_native_payload(
    payload: Mapping[str, Any], destination: str | Path
) -> Path:
    validate_r2_native_payload(payload)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload, stream, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        stream.write("\n")
    return path


def export_r2_native_model(model: Any, destination: str | Path) -> Path:
    return write_r2_native_payload(build_r2_native_payload(model), destination)


def load_r2_native_payload(source: str | Path) -> dict[str, Any]:
    with Path(source).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("R2 native model root must be a JSON object")
    validate_r2_native_payload(payload)
    return payload


class _R2Spline:
    def __init__(self, payload: Mapping[str, Any]):
        self.knots = np.asarray(payload["knots"], dtype=np.float32)
        self.values = np.asarray(payload["values"], dtype=np.float32)
        self.slopes = np.asarray(payload["slopes"], dtype=np.float32)
        self.spacing = np.float32(
            (self.knots[-1] - self.knots[0]) / (len(self.knots) - 1)
        )
        self.drive = np.float32(payload["drive"])
        self.offset = np.float32(payload["offset"])
        self.adaa1 = bool(payload["adaa1"])
        self.threshold = np.float32(
            payload["difference_limit_threshold"] if self.adaa1 else 1.0e-4
        )
        segment_integrals = self.spacing * (
            np.float32(0.5) * self.values[:-1]
            + np.float32(0.5) * self.values[1:]
            + self.spacing * (self.slopes[:-1] - self.slopes[1:]) / np.float32(12.0)
        )
        self.cumulative = np.concatenate(
            (
                np.zeros(1, dtype=np.float32),
                np.cumsum(segment_integrals, dtype=np.float32),
            )
        )
        self.previous = np.float32(0.0)

    def reset(self) -> None:
        self.previous = np.float32(0.0)

    def _coordinate(self, value: np.float32) -> tuple[int, np.float32]:
        coordinate = np.float32((value - self.knots[0]) / self.spacing)
        index = min(max(int(np.floor(coordinate)), 0), len(self.knots) - 2)
        return index, np.float32(coordinate - index)

    def _plain(self, value: np.float32) -> np.float32:
        if value < self.knots[0]:
            return np.float32(self.values[0] + self.slopes[0] * (value - self.knots[0]))
        if value > self.knots[-1]:
            return np.float32(
                self.values[-1] + self.slopes[-1] * (value - self.knots[-1])
            )
        index, local = self._coordinate(value)
        local2 = np.float32(local * local)
        local3 = np.float32(local2 * local)
        h00 = np.float32(2.0 * local3 - 3.0 * local2 + 1.0)
        h10 = np.float32(local3 - 2.0 * local2 + local)
        h01 = np.float32(-2.0 * local3 + 3.0 * local2)
        h11 = np.float32(local3 - local2)
        return np.float32(
            h00 * self.values[index]
            + h10 * self.spacing * self.slopes[index]
            + h01 * self.values[index + 1]
            + h11 * self.spacing * self.slopes[index + 1]
        )

    def _primitive(self, value: np.float32) -> np.float32:
        if value < self.knots[0]:
            distance = np.float32(value - self.knots[0])
            return np.float32(
                self.values[0] * distance
                + np.float32(0.5) * self.slopes[0] * distance * distance
            )
        if value > self.knots[-1]:
            distance = np.float32(value - self.knots[-1])
            return np.float32(
                self.cumulative[-1]
                + self.values[-1] * distance
                + np.float32(0.5) * self.slopes[-1] * distance * distance
            )
        index, local = self._coordinate(value)
        local2 = np.float32(local * local)
        local3 = np.float32(local2 * local)
        local4 = np.float32(local2 * local2)
        integral = self.spacing * (
            (np.float32(0.5) * local4 - local3 + local) * self.values[index]
            + (
                np.float32(0.25) * local4
                - np.float32(2.0 / 3.0) * local3
                + np.float32(0.5) * local2
            )
            * self.spacing
            * self.slopes[index]
            + (-np.float32(0.5) * local4 + local3) * self.values[index + 1]
            + (np.float32(0.25) * local4 - np.float32(1.0 / 3.0) * local3)
            * self.spacing
            * self.slopes[index + 1]
        )
        return np.float32(self.cumulative[index] + integral)

    def sample(
        self,
        input_sample: np.float32,
        drive_factor: float = 1.0,
        offset_delta: float = 0.0,
    ) -> np.float32:
        value = np.float32(
            self.drive * drive_factor * input_sample + self.offset + offset_delta
        )
        if not self.adaa1:
            return self._plain(value)
        difference = np.float32(value - self.previous)
        if abs(difference) < self.threshold:
            output = self._plain(np.float32(0.5) * (value + self.previous))
        else:
            output = np.float32(
                (self._primitive(value) - self._primitive(self.previous)) / difference
            )
        self.previous = value
        return output


class _Delay:
    def __init__(self, samples: int):
        self.history = np.zeros(samples, dtype=np.float32)
        self.index = 0

    def reset(self) -> None:
        self.history.fill(0.0)
        self.index = 0

    def sample(self, value: np.float32) -> np.float32:
        if not self.history.size:
            return value
        output = self.history[self.index]
        self.history[self.index] = value
        self.index = (self.index + 1) % len(self.history)
        return output


class _AAFSSRReference:
    """Stateful float32 reference for off, ADAA, x2, and teacher-x4 exports."""

    def __init__(self, payload: Mapping[str, Any]):
        validate_r2_native_payload(payload)
        core = payload["core"]
        self.filters = [_CausalFir(item["coefficients"]) for item in core["filters"]]
        self.shapers = [_R2Spline(item) for item in core["shapers"]]
        self.output_gain = np.float32(core["output_gain"])
        self.slow = _SlowController(payload["slow_controller"])
        residual = payload["residual"]
        self.input_weight = np.asarray(
            residual["input_projection"]["weight"], dtype=np.float32
        )
        self.input_bias = np.asarray(
            residual["input_projection"]["bias"], dtype=np.float32
        )
        self.layers = [
            _ResidualLayer(layer, dilation, residual["kind"])
            for layer, dilation in zip(
                residual["layers"], residual["dilations"], strict=True
            )
        ]
        self.negative_slope = np.float32(residual["negative_slope"])
        self.scale = np.float32(residual["scale"])
        self.output_weight = np.asarray(
            residual["output_projection"]["weight"], dtype=np.float32
        )
        self.output_bias = np.float32(residual["output_projection"]["bias"])
        self.factor = int(payload["dilation_scale"])
        resampling = payload["resampling"]
        self.upsample = (
            None
            if resampling is None
            else _CausalFir(resampling["upsample_coefficients"])
        )
        self.downsample = (
            None
            if resampling is None
            else _CausalFir(resampling["downsample_coefficients"])
        )
        delay = int(payload["latency_samples"])
        self.delay = _Delay(delay)

    def reset(self) -> None:
        for fir in self.filters:
            fir.reset()
        for shaper in self.shapers:
            shaper.reset()
        for layer in self.layers:
            layer.reset()
        self.slow.reset()
        if self.upsample is not None:
            self.upsample.reset()
            assert self.downsample is not None
            self.downsample.reset()
        self.delay.reset()

    def _internal_sample(self, input_sample: np.float32) -> np.float32:
        modulation = self.slow.modulation()
        core = self.filters[0].sample(input_sample)
        for index, shaper in enumerate(self.shapers):
            shaped = (
                shaper.sample(core, modulation[0], modulation[1])
                if index == 0
                else shaper.sample(core)
            )
            core = self.filters[index + 1].sample(shaped)
        core = np.float32(self.output_gain * modulation[2] * core)
        self.slow.observe(input_sample)
        features = np.asarray((input_sample, core), dtype=np.float32)
        hidden = np.asarray(
            self.input_weight @ features + self.input_bias, dtype=np.float32
        )
        for layer in self.layers:
            update = layer.sample(hidden)
            activated = np.where(
                update >= 0.0, update, self.negative_slope * update
            ).astype(np.float32)
            hidden = np.asarray(hidden + activated, dtype=np.float32)
        raw = np.float32(np.dot(self.output_weight, hidden) + self.output_bias)
        return np.float32(core + self.scale * np.tanh(raw))

    def _sample(self, value: np.float32) -> np.float32:
        if self.factor == 1:
            return self.delay.sample(self._internal_sample(value))
        assert self.upsample is not None and self.downsample is not None
        selected = np.float32(0.0)
        for phase in range(self.factor):
            inserted = value if phase == 0 else np.float32(0.0)
            interpolated = self.upsample.sample(inserted)
            branch = self._internal_sample(interpolated)
            filtered = self.downsample.sample(np.float32(branch - interpolated))
            if phase == 0:
                selected = filtered
        return np.float32(self.delay.sample(value) + selected)

    def process(self, samples: Any) -> NDArray[np.float32]:
        signal = _array(samples, "samples").reshape(-1)
        output = np.empty_like(signal)
        for index, value in enumerate(signal):
            output[index] = self._sample(value)
        return output


class _VectorCausalConv:
    def __init__(
        self,
        weight: Any,
        bias: Any,
        *,
        input_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        self.weight = np.asarray(weight, dtype=np.float32).reshape(
            -1, input_channels, kernel_size
        )
        self.bias = np.asarray(bias, dtype=np.float32).reshape(-1)
        self.history = np.zeros(
            ((kernel_size - 1) * dilation, input_channels), dtype=np.float32
        )
        self.dilation = dilation
        self.index = 0

    def reset(self) -> None:
        self.history.fill(0.0)
        self.index = 0

    def sample(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        current = np.asarray(inputs, dtype=np.float32).reshape(-1)
        output = self.bias.copy()
        history_size = len(self.history)
        for tap in range(self.weight.shape[2]):
            lag = (self.weight.shape[2] - 1 - tap) * self.dilation
            source = (
                current
                if lag == 0
                else self.history[(self.index + history_size - lag) % history_size]
            )
            output = np.asarray(
                output + self.weight[:, :, tap] @ source, dtype=np.float32
            )
        if history_size:
            self.history[self.index] = current
            self.index = (self.index + 1) % history_size
        return output


class _AANAMReference:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        wavenet = payload["wavenet"]
        self.warmup_samples = int(wavenet["receptive_field"]) - 1
        if payload["aa_mode"] == "adaa1":
            self.warmup_samples += 23
        self.input_projection = np.asarray(
            wavenet["input_projection"], dtype=np.float32
        ).reshape(8)
        self.layers = []
        self.activations: list[list[_R2Spline]] = []
        for declaration in wavenet["layers"]:
            self.layers.append(
                (
                    _VectorCausalConv(
                        declaration["conv_weight"],
                        declaration["conv_bias"],
                        input_channels=8,
                        kernel_size=int(declaration["kernel_size"]),
                        dilation=int(declaration["dilation"]),
                    ),
                    np.asarray(declaration["condition_weight"], dtype=np.float32),
                    np.asarray(declaration["residual_weight"], dtype=np.float32),
                    np.asarray(declaration["residual_bias"], dtype=np.float32),
                )
            )
            self.activations.append(
                [_R2Spline(declaration["activation"]) for _ in range(8)]
            )
        head = wavenet["head"]
        self.head = _VectorCausalConv(
            head["weight"],
            [head["bias"]],
            input_channels=8,
            kernel_size=int(head["kernel_size"]),
            dilation=1,
        )
        self.head_scale = np.float32(wavenet["head_scale"])
        self.factor = int(payload["dilation_scale"])
        resampling = payload["resampling"]
        self.upsample = (
            None
            if resampling is None
            else _CausalFir(resampling["upsample_coefficients"])
        )
        self.downsample = (
            None
            if resampling is None
            else _CausalFir(resampling["downsample_coefficients"])
        )
        self.delay = _Delay(int(payload["latency_samples"]))
        self.reset()

    def reset(self) -> None:
        for (convolution, _, _, _), activations in zip(
            self.layers, self.activations, strict=True
        ):
            convolution.reset()
            for activation in activations:
                activation.reset()
        self.head.reset()
        if self.upsample is not None:
            self.upsample.reset()
            assert self.downsample is not None
            self.downsample.reset()
        self.delay.reset()
        for _ in range(self.warmup_samples):
            self._internal_sample(np.float32(0.0))

    def _internal_sample(self, input_sample: np.float32) -> np.float32:
        hidden = np.asarray(self.input_projection * input_sample, dtype=np.float32)
        head_sum = np.zeros(8, dtype=np.float32)
        for declaration, activations in zip(self.layers, self.activations, strict=True):
            convolution, condition, residual_weight, residual_bias = declaration
            shaped_input = np.asarray(
                convolution.sample(hidden) + condition * input_sample,
                dtype=np.float32,
            )
            activated = np.asarray(
                [
                    activation.sample(value)
                    for activation, value in zip(activations, shaped_input, strict=True)
                ],
                dtype=np.float32,
            )
            update = np.asarray(
                residual_weight @ activated + residual_bias, dtype=np.float32
            )
            hidden = np.asarray(hidden + update, dtype=np.float32)
            head_sum = np.asarray(head_sum + activated, dtype=np.float32)
        return np.float32(self.head_scale * self.head.sample(head_sum)[0])

    def _sample(self, value: np.float32) -> np.float32:
        if self.factor == 1:
            return self.delay.sample(self._internal_sample(value))
        assert self.upsample is not None and self.downsample is not None
        selected = np.float32(0.0)
        for phase in range(self.factor):
            inserted = value if phase == 0 else np.float32(0.0)
            interpolated = self.upsample.sample(inserted)
            branch = self._internal_sample(interpolated)
            filtered = self.downsample.sample(np.float32(branch - interpolated))
            if phase == 0:
                selected = filtered
        return np.float32(self.delay.sample(value) + selected)

    def process(self, samples: Any) -> NDArray[np.float32]:
        signal = _array(samples, "samples").reshape(-1)
        return np.asarray([self._sample(value) for value in signal], dtype=np.float32)


class R2NativeReference:
    """Dispatch the frozen float32 reference for either prospective family."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        validate_r2_native_payload(payload)
        implementation = (
            _AANAMReference if payload["family"] == "aa-nam" else _AAFSSRReference
        )
        self._implementation = implementation(payload)

    def reset(self) -> None:
        self._implementation.reset()

    def process(self, samples: Any) -> NDArray[np.float32]:
        return self._implementation.process(samples)
