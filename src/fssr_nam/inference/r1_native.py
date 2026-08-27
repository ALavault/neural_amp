"""Versioned export and NumPy reference inference for native FSSR-R1 models.

The v1 graph is an alternating causal-FIR/Hermite-spline core, the existing
decimated ``SlowStateController``, and an optional registered R1 residual.  No
other slow-controller type or modulation route is accepted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FORMAT = "fssr-r1-native-v1"
CAMPAIGN_VERSION = "FSSR-R1"
RF_DILATIONS = {
    31: (1, 2, 4, 8),
    2047: (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
}


def _value(member: Any) -> Any:
    """Detach a tensor-like value without importing the optional torch package."""
    if hasattr(member, "detach"):
        member = member.detach()
    if hasattr(member, "cpu"):
        member = member.cpu()
    if hasattr(member, "numpy"):
        member = member.numpy()
    return member


def _array(member: Any, name: str) -> NDArray[np.float32]:
    array = np.asarray(_value(member), dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite float32 values")
    return array


def _scalar(member: Any, name: str) -> float:
    array = _array(member, name)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    return float(array.reshape(()))


def _member(instance: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(instance, Mapping) and name in instance:
            return instance[name]
        if hasattr(instance, name):
            return getattr(instance, name)
    return default


def _required_member(instance: Any, name: str, *aliases: str) -> Any:
    missing = object()
    member = _member(instance, name, *aliases, default=missing)
    if member is missing:
        choices = ", ".join((name, *aliases))
        raise TypeError(f"native export requires one of: {choices}")
    return member


def _parameter_list(member: Any, name: str, ndim: int) -> list[Any]:
    array = _array(member, name)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    return array.tolist()


def _conv(instance: Any) -> Any:
    return _member(instance, "conv", default=instance)


def _conv_weight(instance: Any, name: str) -> NDArray[np.float32]:
    weight = _array(_required_member(_conv(instance), "weight"), f"{name}.weight")
    if weight.ndim != 3:
        raise ValueError(f"{name}.weight must be a Conv1d tensor")
    return weight


def _conv_bias(instance: Any, length: int, name: str) -> NDArray[np.float32]:
    bias = _member(_conv(instance), "bias")
    if bias is None:
        return np.zeros(length, dtype=np.float32)
    values = _array(bias, f"{name}.bias")
    if values.shape != (length,):
        raise ValueError(f"{name}.bias must have shape ({length},)")
    return values


def _fir_payload(fir: Any, name: str) -> dict[str, Any]:
    coefficients = _array(
        _required_member(fir, "coefficients", "weight"), f"{name}.coefficients"
    )
    coefficients = coefficients.reshape(-1)
    if coefficients.size < 1:
        raise ValueError(f"{name} must have at least one coefficient")
    return {"coefficients": coefficients.tolist()}


def _indexed_scalar(
    core: Any, shaper: Any, base: str, index: int, *, default: float
) -> float:
    direct = _member(shaper, base)
    if direct is not None:
        return _scalar(direct, f"shaper[{index}].{base}")
    for name in (
        f"{base}{index + 1}",
        f"{base}_{index + 1}",
        base if index == 0 else "",
    ):
        if name:
            value = _member(core, name)
            if value is not None:
                return _scalar(value, f"core.{name}")
    sequence = _member(core, f"{base}s")
    if sequence is not None:
        return _scalar(sequence[index], f"core.{base}s[{index}]")
    return default


def _shaper_payload(core: Any, shaper: Any, index: int) -> dict[str, Any]:
    knots = _array(_required_member(shaper, "knots"), f"shaper[{index}].knots")
    values = _array(_required_member(shaper, "values"), f"shaper[{index}].values")
    slopes = _array(_required_member(shaper, "slopes"), f"shaper[{index}].slopes")
    for name, array in (("knots", knots), ("values", values), ("slopes", slopes)):
        if array.ndim != 1:
            raise ValueError(f"shaper[{index}].{name} must be one-dimensional")
    return {
        "knots": knots.tolist(),
        "values": values.tolist(),
        "slopes": slopes.tolist(),
        "drive": _indexed_scalar(core, shaper, "drive", index, default=1.0),
        "offset": _indexed_scalar(core, shaper, "offset", index, default=0.0),
    }


def _as_sequence(value: Any, name: str) -> list[Any]:
    if isinstance(value, Sequence) or hasattr(value, "__iter__"):
        return list(value)
    raise TypeError(f"{name} must be a sequence")


def _extract_core(model: Any) -> dict[str, Any]:
    core = _member(model, "core", default=model)
    filters_member = _member(core, "filters")
    shapers_member = _member(core, "shapers")
    if filters_member is not None and shapers_member is not None:
        filters = _as_sequence(filters_member, "core.filters")
        shapers = _as_sequence(shapers_member, "core.shapers")
    elif all(hasattr(core, name) for name in ("pre", "shaper", "post")):
        filters = [core.pre, core.post]
        shapers = [core.shaper]
    elif all(hasattr(core, name) for name in ("h0", "spline1", "h1", "spline2", "h2")):
        filters = [core.h0, core.h1, core.h2]
        shapers = [core.spline1, core.spline2]
    else:
        raise TypeError(
            "unsupported R1 core: expected filters/shapers, pre/shaper/post, "
            "or h0/spline1/h1/spline2/h2"
        )
    if len(shapers) not in (1, 2) or len(filters) != len(shapers) + 1:
        raise ValueError("native v1 supports one or two shapers with N+1 FIR filters")
    kind = "mono" if len(shapers) == 1 else "cascade"
    output_gain = _member(core, "output_gain")
    if output_gain is None:
        output_gain = _member(model, "output_gain", default=1.0)
    return {
        "kind": kind,
        "filters": [
            _fir_payload(fir, f"core.filters[{index}]")
            for index, fir in enumerate(filters)
        ],
        "shapers": [
            _shaper_payload(core, shaper, index) for index, shaper in enumerate(shapers)
        ],
        "output_gain": _scalar(output_gain, "core.output_gain"),
    }


def _projection_payload(projection: Any) -> tuple[list[list[float]], list[float]]:
    weight = _conv_weight(projection, "residual.input_projection")
    if weight.shape[2] != 1:
        raise ValueError("residual input projection must use kernel size one")
    matrix = weight[:, :, 0]
    bias = _conv_bias(projection, matrix.shape[0], "residual.input_projection")
    return matrix.tolist(), bias.tolist()


def _layer_dilation(layer: Any, convolution: Any) -> int:
    dilation_value = _member(
        _conv(convolution), "dilation", default=_member(layer, "dilation")
    )
    if isinstance(dilation_value, Sequence):
        dilation_value = dilation_value[0]
    if dilation_value is None:
        raise TypeError("each residual layer must expose its dilation")
    return int(dilation_value)


def _layer_payload(layer: Any, index: int) -> tuple[dict[str, Any], int, str, int]:
    depthwise = _member(layer, "depthwise")
    pointwise = _member(layer, "pointwise")
    if depthwise is None or pointwise is None:
        convolution = _conv(layer)
        weight = _conv_weight(convolution, f"residual.layers[{index}]")
        channels, input_channels, kernel_size = weight.shape
        if input_channels != channels:
            raise ValueError(
                "full residual weights must have shape (channels,channels,k)"
            )
        bias = _conv_bias(convolution, channels, f"residual.layers[{index}]")
        return (
            {
                "convolution_weight": weight.tolist(),
                "convolution_bias": bias.tolist(),
            },
            _layer_dilation(layer, convolution),
            "causal-full-convolution-tcn",
            kernel_size,
        )
    depthwise_weight = _conv_weight(depthwise, f"residual.layers[{index}].depthwise")
    if depthwise_weight.shape[1] != 1:
        raise ValueError("depthwise weights must have shape (channels,1,kernel)")
    channels, _, kernel_size = depthwise_weight.shape
    pointwise_weight = _conv_weight(pointwise, f"residual.layers[{index}].pointwise")
    if pointwise_weight.shape != (channels, channels, 1):
        raise ValueError("pointwise weights must have shape (channels,channels,1)")
    depthwise_bias = _conv_bias(
        depthwise, channels, f"residual.layers[{index}].depthwise"
    )
    pointwise_bias = _conv_bias(
        pointwise, channels, f"residual.layers[{index}].pointwise"
    )
    return (
        {
            "depthwise_weight": depthwise_weight[:, 0, :].tolist(),
            "depthwise_bias": depthwise_bias.tolist(),
            "pointwise_weight": pointwise_weight[:, :, 0].tolist(),
            "pointwise_bias": pointwise_bias.tolist(),
        },
        _layer_dilation(layer, depthwise),
        "causal-depthwise-separable-tcn",
        kernel_size,
    )


def _extract_residual(model: Any) -> dict[str, Any] | None:
    residual = _member(model, "residual")
    if residual is None:
        return None
    projection = _required_member(residual, "input_projection")
    input_weight, input_bias = _projection_payload(projection)
    channels = len(input_weight)
    if channels != 8 or any(len(row) != 2 for row in input_weight):
        raise ValueError(
            "R1 native residual requires 8 channels and two input features"
        )
    layer_payloads: list[dict[str, Any]] = []
    dilations: list[int] = []
    layer_kinds: list[str] = []
    kernel_sizes: list[int] = []
    residual_layers = _as_sequence(_required_member(residual, "layers"), "layers")
    for index, layer in enumerate(residual_layers):
        payload, dilation, layer_kind, kernel_size = _layer_payload(layer, index)
        layer_payloads.append(payload)
        dilations.append(dilation)
        layer_kinds.append(layer_kind)
        kernel_sizes.append(kernel_size)
    if not layer_payloads or len(set(layer_kinds)) != 1 or len(set(kernel_sizes)) != 1:
        raise ValueError("residual layers must share one operator and kernel size")
    output = _required_member(residual, "output_projection")
    output_weight = _conv_weight(output, "residual.output_projection")
    if output_weight.shape != (1, channels, 1):
        raise ValueError("residual output projection must have shape (1,channels,1)")
    output_bias = _conv_bias(output, 1, "residual.output_projection")
    scale = _member(residual, "residual_scale", "scale")
    if scale is None:
        scale_logit = _scalar(
            _required_member(residual, "scale_logit"), "residual.scale_logit"
        )
        scale = 0.5 / (1.0 + math.exp(-scale_logit))
    else:
        scale = _scalar(scale, "residual.scale")
    negative_slope = _member(residual, "negative_slope", default=0.01)
    payload = {
        "kind": layer_kinds[0],
        "input_features": ["input", "core"],
        "channels": channels,
        "kernel_size": kernel_sizes[0],
        "dilations": dilations,
        "receptive_field": 1 + (kernel_sizes[0] - 1) * sum(dilations),
        "negative_slope": _scalar(negative_slope, "residual.negative_slope"),
        "scale": scale,
        "input_projection": {"weight": input_weight, "bias": input_bias},
        "layers": layer_payloads,
        "output_projection": {
            "weight": output_weight[0, :, 0].tolist(),
            "bias": float(output_bias[0]),
        },
    }
    return payload


def _extract_slow(model: Any, core_kind: str) -> dict[str, Any] | str:
    slow = _member(model, "slow", "slow_controller")
    if slow is None:
        return "none"
    hidden_size = int(_required_member(slow, "hidden_size"))
    decimation = int(_required_member(slow, "decimation"))
    gru = _required_member(slow, "gru")
    projection = _required_member(slow, "projection")
    weight_ih = _array(_required_member(gru, "weight_ih"), "slow.gru.weight_ih")
    weight_hh = _array(_required_member(gru, "weight_hh"), "slow.gru.weight_hh")
    bias_ih = _array(_required_member(gru, "bias_ih"), "slow.gru.bias_ih")
    bias_hh = _array(_required_member(gru, "bias_hh"), "slow.gru.bias_hh")
    projection_weight = _array(
        _required_member(projection, "weight"), "slow.projection.weight"
    )
    projection_bias = _array(
        _required_member(projection, "bias"), "slow.projection.bias"
    )
    expected_shapes = {
        "weight_ih": (3 * hidden_size, 3),
        "weight_hh": (3 * hidden_size, hidden_size),
        "bias_ih": (3 * hidden_size,),
        "bias_hh": (3 * hidden_size,),
        "projection_weight": (3, hidden_size),
        "projection_bias": (3,),
    }
    actual = {
        "weight_ih": weight_ih.shape,
        "weight_hh": weight_hh.shape,
        "bias_ih": bias_ih.shape,
        "bias_hh": bias_hh.shape,
        "projection_weight": projection_weight.shape,
        "projection_bias": projection_bias.shape,
    }
    if actual != expected_shapes:
        raise ValueError(f"slow controller shapes differ: {actual!r}")
    application = (
        "first-shaper-drive-offset-and-output-gain"
        if core_kind == "cascade"
        else "shaper-drive-offset-and-output-gain"
    )
    return {
        "kind": "fssr-slow-gru-v1",
        "hidden_size": hidden_size,
        "decimation": decimation,
        "feature_order": ["mean_abs", "mean_square", "mean"],
        "gate_order": ["reset", "update", "new"],
        "application": application,
        "drive_log_range": 0.25,
        "offset_range": 0.1,
        "gain_log_range": 0.25,
        "weight_ih": weight_ih.tolist(),
        "weight_hh": weight_hh.tolist(),
        "bias_ih": bias_ih.tolist(),
        "bias_hh": bias_hh.tolist(),
        "projection_weight": projection_weight.tolist(),
        "projection_bias": projection_bias.tolist(),
    }


def build_r1_native_payload(
    model: Any, *, sample_rate_hz: int = 48_000
) -> dict[str, Any]:
    """Convert a compatible mono/cascade R1 PyTorch model to native v1 data."""
    core = _extract_core(model)
    payload = {
        "format": FORMAT,
        "version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "precision": "float32",
        "sample_rate_hz": int(sample_rate_hz),
        "latency_samples": 0,
        "slow_controller": _extract_slow(model, core["kind"]),
        "core": core,
        "residual": _extract_residual(model),
    }
    validate_r1_native_payload(payload)
    return payload


def _finite_vector(value: Any, name: str, *, length: int | None = None) -> list[float]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        expected = "a list" if length is None else f"a list of length {length}"
        raise ValueError(f"{name} must be {expected}")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _finite_matrix(value: Any, name: str, rows: int, columns: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    return [
        _finite_vector(row, f"{name}[{index}]", length=columns)
        for index, row in enumerate(value)
    ]


def validate_r1_native_payload(payload: Mapping[str, Any]) -> None:
    """Validate topology, dimensions, finiteness, and the registered RF exactly."""
    required = {
        "format": FORMAT,
        "version": 1,
        "campaign_version": CAMPAIGN_VERSION,
        "precision": "float32",
        "latency_samples": 0,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"{key} must equal {expected!r}")
    sample_rate = payload.get("sample_rate_hz")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate_hz must be a positive integer")
    core = payload.get("core")
    if not isinstance(core, Mapping) or core.get("kind") not in {"mono", "cascade"}:
        raise ValueError("core.kind must be mono or cascade")
    shaper_count = 1 if core["kind"] == "mono" else 2
    filters = core.get("filters")
    shapers = core.get("shapers")
    if not isinstance(filters, list) or len(filters) != shaper_count + 1:
        raise ValueError("core filter count does not match its kind")
    if not isinstance(shapers, list) or len(shapers) != shaper_count:
        raise ValueError("core shaper count does not match its kind")
    for index, fir in enumerate(filters):
        if not isinstance(fir, Mapping):
            raise ValueError(f"core.filters[{index}] must be an object")
        coefficients = _finite_vector(
            fir.get("coefficients"), f"core.filters[{index}].coefficients"
        )
        if not coefficients:
            raise ValueError("FIR filters cannot be empty")
    for index, shaper in enumerate(shapers):
        if not isinstance(shaper, Mapping):
            raise ValueError(f"core.shapers[{index}] must be an object")
        knots = _finite_vector(shaper.get("knots"), f"core.shapers[{index}].knots")
        values = _finite_vector(
            shaper.get("values"), f"core.shapers[{index}].values", length=len(knots)
        )
        slopes = _finite_vector(
            shaper.get("slopes"), f"core.shapers[{index}].slopes", length=len(knots)
        )
        if len(knots) < 4 or not all(a < b for a, b in pairwise(knots)):
            raise ValueError(
                "spline knots must contain at least four increasing values"
            )
        uniform = np.linspace(knots[0], knots[-1], len(knots), dtype=np.float32)
        if not np.allclose(knots, uniform, atol=1.0e-6, rtol=1.0e-6):
            raise ValueError("spline knots must use the registered uniform grid")
        if len(values) != len(slopes):
            raise ValueError("spline vectors differ in length")
        for scalar in ("drive", "offset"):
            if not math.isfinite(float(shaper.get(scalar, math.nan))):
                raise ValueError(f"core.shapers[{index}].{scalar} must be finite")
    if not math.isfinite(float(core.get("output_gain", math.nan))):
        raise ValueError("core.output_gain must be finite")
    slow = payload.get("slow_controller")
    if slow != "none":
        if not isinstance(slow, Mapping) or slow.get("kind") != "fssr-slow-gru-v1":
            raise ValueError("slow_controller must be none or fssr-slow-gru-v1")
        hidden_size = slow.get("hidden_size")
        decimation = slow.get("decimation")
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("slow hidden_size must be a positive integer")
        if not isinstance(decimation, int) or decimation < 1:
            raise ValueError("slow decimation must be a positive integer")
        if slow.get("feature_order") != ["mean_abs", "mean_square", "mean"]:
            raise ValueError("slow feature order is unsupported")
        if slow.get("gate_order") != ["reset", "update", "new"]:
            raise ValueError("slow gate order must match torch GRUCell")
        expected_application = (
            "first-shaper-drive-offset-and-output-gain"
            if core["kind"] == "cascade"
            else "shaper-drive-offset-and-output-gain"
        )
        if slow.get("application") != expected_application:
            raise ValueError("slow modulation application does not match the core")
        constants = {
            "drive_log_range": 0.25,
            "offset_range": 0.1,
            "gain_log_range": 0.25,
        }
        for name, expected in constants.items():
            if float(slow.get(name, math.nan)) != expected:
                raise ValueError(f"slow {name} must equal {expected}")
        _finite_matrix(slow.get("weight_ih"), "slow weight_ih", 3 * hidden_size, 3)
        _finite_matrix(
            slow.get("weight_hh"),
            "slow weight_hh",
            3 * hidden_size,
            hidden_size,
        )
        _finite_vector(slow.get("bias_ih"), "slow bias_ih", length=3 * hidden_size)
        _finite_vector(slow.get("bias_hh"), "slow bias_hh", length=3 * hidden_size)
        _finite_matrix(
            slow.get("projection_weight"),
            "slow projection weight",
            3,
            hidden_size,
        )
        _finite_vector(slow.get("projection_bias"), "slow projection bias", length=3)
    residual = payload.get("residual")
    if residual is None:
        return
    if not isinstance(residual, Mapping):
        raise ValueError("residual must be null or an object")
    residual_kind = residual.get("kind")
    supported_kinds = {
        "causal-depthwise-separable-tcn",
        "causal-full-convolution-tcn",
    }
    if residual_kind not in supported_kinds:
        raise ValueError("unsupported residual kind")
    if residual.get("input_features") != ["input", "core"]:
        raise ValueError("residual input feature order must be [input, core]")
    channels = residual.get("channels")
    kernel_size = residual.get("kernel_size")
    dilations = residual.get("dilations")
    receptive_field = residual.get("receptive_field")
    if channels != 8 or kernel_size != 3:
        raise ValueError("R1 residual requires eight channels and kernel size three")
    if receptive_field not in RF_DILATIONS:
        raise ValueError("R1 native supports only RF31 and RF2047")
    if tuple(dilations or ()) != RF_DILATIONS[receptive_field]:
        raise ValueError("dilations do not exactly match the declared receptive field")
    if residual_kind == "causal-full-convolution-tcn" and receptive_field != 31:
        raise ValueError("the registered full-convolution ablation is RF31 only")
    negative_slope = float(residual.get("negative_slope", math.nan))
    scale = float(residual.get("scale", math.nan))
    if not math.isfinite(negative_slope) or negative_slope < 0.0:
        raise ValueError("residual negative_slope must be finite and non-negative")
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("residual scale must be finite and in [0,1]")
    input_projection = residual.get("input_projection", {})
    _finite_matrix(input_projection.get("weight"), "input weight", channels, 2)
    _finite_vector(input_projection.get("bias"), "input bias", length=channels)
    layers = residual.get("layers")
    if not isinstance(layers, list) or len(layers) != len(dilations):
        raise ValueError("one residual layer is required per dilation")
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ValueError(f"residual.layers[{index}] must be an object")
        if residual_kind == "causal-depthwise-separable-tcn":
            _finite_matrix(
                layer.get("depthwise_weight"),
                f"layer {index} depthwise weight",
                channels,
                kernel_size,
            )
            _finite_vector(
                layer.get("depthwise_bias"),
                f"layer {index} depthwise bias",
                length=channels,
            )
            _finite_matrix(
                layer.get("pointwise_weight"),
                f"layer {index} pointwise weight",
                channels,
                channels,
            )
            _finite_vector(
                layer.get("pointwise_bias"),
                f"layer {index} pointwise bias",
                length=channels,
            )
        else:
            weight = layer.get("convolution_weight")
            if not isinstance(weight, list) or len(weight) != channels:
                raise ValueError(f"layer {index} convolution weight has wrong shape")
            for output, matrix in enumerate(weight):
                _finite_matrix(
                    matrix,
                    f"layer {index} convolution weight[{output}]",
                    channels,
                    kernel_size,
                )
            _finite_vector(
                layer.get("convolution_bias"),
                f"layer {index} convolution bias",
                length=channels,
            )
    output_projection = residual.get("output_projection", {})
    _finite_vector(output_projection.get("weight"), "output weight", length=channels)
    if not math.isfinite(float(output_projection.get("bias", math.nan))):
        raise ValueError("output bias must be finite")


def write_r1_native_payload(
    payload: Mapping[str, Any], destination: str | Path
) -> Path:
    """Validate and write deterministic JSON without changing learned values."""
    validate_r1_native_payload(payload)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload, stream, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        stream.write("\n")
    return path


def export_r1_native_model(
    model: Any, destination: str | Path, *, sample_rate_hz: int = 48_000
) -> Path:
    """Export a compatible R1 model after converting every value to float32."""
    return write_r1_native_payload(
        build_r1_native_payload(model, sample_rate_hz=sample_rate_hz), destination
    )


def load_r1_native_payload(source: str | Path) -> dict[str, Any]:
    with Path(source).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("native model root must be a JSON object")
    validate_r1_native_payload(payload)
    return payload


class _CausalFir:
    def __init__(self, coefficients: Sequence[float]):
        self.coefficients = np.asarray(coefficients, dtype=np.float32)
        self.history = np.zeros(max(0, len(self.coefficients) - 1), dtype=np.float32)

    def reset(self) -> None:
        self.history.fill(0.0)

    def sample(self, value: np.float32) -> np.float32:
        joined = np.append(self.history, value).astype(np.float32, copy=False)
        result = np.float32(np.dot(self.coefficients, joined))
        if self.history.size:
            self.history[:] = joined[1:]
        return result


class _Spline:
    def __init__(self, payload: Mapping[str, Any]):
        self.knots = np.asarray(payload["knots"], dtype=np.float32)
        self.values = np.asarray(payload["values"], dtype=np.float32)
        self.slopes = np.asarray(payload["slopes"], dtype=np.float32)
        self.spacing = np.float32(
            (self.knots[-1] - self.knots[0]) / (len(self.knots) - 1)
        )
        self.drive = np.float32(payload["drive"])
        self.offset = np.float32(payload["offset"])

    def sample(
        self,
        value: np.float32,
        drive_factor: float = 1.0,
        offset_delta: float = 0.0,
    ) -> np.float32:
        value = np.float32(
            self.drive * drive_factor * value + self.offset + offset_delta
        )
        if value < self.knots[0]:
            return np.float32(self.values[0] + self.slopes[0] * (value - self.knots[0]))
        if value > self.knots[-1]:
            return np.float32(
                self.values[-1] + self.slopes[-1] * (value - self.knots[-1])
            )
        coordinate = np.float32((value - self.knots[0]) / self.spacing)
        index = int(np.floor(coordinate))
        index = min(max(index, 0), len(self.knots) - 2)
        spacing = self.spacing
        local = np.float32(coordinate - index)
        local2 = np.float32(local * local)
        local3 = np.float32(local2 * local)
        h00 = np.float32(2.0 * local3 - 3.0 * local2 + 1.0)
        h10 = np.float32(local3 - 2.0 * local2 + local)
        h01 = np.float32(-2.0 * local3 + 3.0 * local2)
        h11 = np.float32(local3 - local2)
        return np.float32(
            h00 * self.values[index]
            + h10 * spacing * self.slopes[index]
            + h01 * self.values[index + 1]
            + h11 * spacing * self.slopes[index + 1]
        )


class _ResidualLayer:
    def __init__(self, payload: Mapping[str, Any], dilation: int, kind: str):
        self.kind = kind
        if kind == "causal-depthwise-separable-tcn":
            self.depthwise_weight = np.asarray(
                payload["depthwise_weight"], dtype=np.float32
            )
            self.depthwise_bias = np.asarray(
                payload["depthwise_bias"], dtype=np.float32
            )
            self.pointwise_weight = np.asarray(
                payload["pointwise_weight"], dtype=np.float32
            )
            self.pointwise_bias = np.asarray(
                payload["pointwise_bias"], dtype=np.float32
            )
            channels = len(self.depthwise_weight)
        else:
            self.convolution_weight = np.asarray(
                payload["convolution_weight"], dtype=np.float32
            )
            self.convolution_bias = np.asarray(
                payload["convolution_bias"], dtype=np.float32
            )
            channels = len(self.convolution_weight)
        self.dilation = dilation
        self.history = np.zeros((channels, 2 * dilation), dtype=np.float32)

    def reset(self) -> None:
        self.history.fill(0.0)

    def sample(self, hidden: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.kind == "causal-depthwise-separable-tcn":
            depthwise = self.depthwise_bias.copy()
            depthwise += self.depthwise_weight[:, 0] * self.history[:, 0]
            depthwise += self.depthwise_weight[:, 1] * self.history[:, self.dilation]
            depthwise += self.depthwise_weight[:, 2] * hidden
            result = np.asarray(
                self.pointwise_weight @ depthwise + self.pointwise_bias,
                dtype=np.float32,
            )
        else:
            inputs = np.stack(
                (self.history[:, 0], self.history[:, self.dilation], hidden), axis=1
            )
            result = np.asarray(
                np.einsum(
                    "oik,ik->o",
                    self.convolution_weight,
                    inputs,
                    dtype=np.float32,
                )
                + self.convolution_bias,
                dtype=np.float32,
            )
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1] = hidden
        return result


class _SlowController:
    def __init__(self, payload: Mapping[str, Any]):
        self.hidden_size = int(payload["hidden_size"])
        self.decimation = int(payload["decimation"])
        self.weight_ih = np.asarray(payload["weight_ih"], dtype=np.float32)
        self.weight_hh = np.asarray(payload["weight_hh"], dtype=np.float32)
        self.bias_ih = np.asarray(payload["bias_ih"], dtype=np.float32)
        self.bias_hh = np.asarray(payload["bias_hh"], dtype=np.float32)
        self.projection_weight = np.asarray(
            payload["projection_weight"], dtype=np.float32
        )
        self.projection_bias = np.asarray(payload["projection_bias"], dtype=np.float32)
        self.hidden = np.zeros(self.hidden_size, dtype=np.float32)
        self.accumulator = np.zeros(3, dtype=np.float32)
        self.count = 0

    def reset(self) -> None:
        self.hidden.fill(0.0)
        self.accumulator.fill(0.0)
        self.count = 0

    @staticmethod
    def _sigmoid(value: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.asarray(1.0 / (1.0 + np.exp(-value)), dtype=np.float32)

    def _update_hidden(self) -> None:
        features = np.asarray(self.accumulator / self.decimation, dtype=np.float32)
        input_gates = np.asarray(
            self.weight_ih @ features + self.bias_ih, dtype=np.float32
        )
        hidden_gates = np.asarray(
            self.weight_hh @ self.hidden + self.bias_hh, dtype=np.float32
        )
        hidden_size = self.hidden_size
        reset = self._sigmoid(input_gates[:hidden_size] + hidden_gates[:hidden_size])
        update = self._sigmoid(
            input_gates[hidden_size : 2 * hidden_size]
            + hidden_gates[hidden_size : 2 * hidden_size]
        )
        candidate = np.tanh(
            input_gates[2 * hidden_size :] + reset * hidden_gates[2 * hidden_size :]
        ).astype(np.float32)
        self.hidden = np.asarray(
            candidate + update * (self.hidden - candidate), dtype=np.float32
        )

    def modulation(self) -> NDArray[np.float32]:
        raw = np.asarray(
            self.projection_weight @ self.hidden + self.projection_bias,
            dtype=np.float32,
        )
        bounded = np.tanh(raw).astype(np.float32)
        return np.asarray(
            (
                np.exp(np.float32(0.25) * bounded[0]),
                np.float32(0.1) * bounded[1],
                np.exp(np.float32(0.25) * bounded[2]),
            ),
            dtype=np.float32,
        )

    def observe(self, sample: np.float32) -> None:
        self.accumulator += np.asarray(
            (np.abs(sample), sample * sample, sample), dtype=np.float32
        )
        self.count += 1
        if self.count == self.decimation:
            self._update_hidden()
            self.accumulator.fill(0.0)
            self.count = 0


class R1NativeReference:
    """Stateful sample-accurate reference for the documented native v1 graph."""

    def __init__(self, payload: Mapping[str, Any]):
        validate_r1_native_payload(payload)
        core = payload["core"]
        self.filters = [_CausalFir(item["coefficients"]) for item in core["filters"]]
        self.shapers = [_Spline(item) for item in core["shapers"]]
        self.output_gain = np.float32(core["output_gain"])
        slow = payload["slow_controller"]
        self.slow = None if slow == "none" else _SlowController(slow)
        residual = payload.get("residual")
        self.residual = residual
        if residual is not None:
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

    def reset(self) -> None:
        for fir in self.filters:
            fir.reset()
        for layer in getattr(self, "layers", []):
            layer.reset()
        if self.slow is not None:
            self.slow.reset()

    def _sample(self, input_sample: np.float32) -> np.float32:
        modulation = (
            np.asarray((1.0, 0.0, 1.0), dtype=np.float32)
            if self.slow is None
            else self.slow.modulation()
        )
        core = self.filters[0].sample(input_sample)
        for index, shaper in enumerate(self.shapers):
            shaped = (
                shaper.sample(core, modulation[0], modulation[1])
                if index == 0
                else shaper.sample(core)
            )
            core = self.filters[index + 1].sample(shaped)
        core = np.float32(self.output_gain * modulation[2] * core)
        if self.slow is not None:
            self.slow.observe(input_sample)
        if self.residual is None:
            return core
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

    def process(self, samples: Any) -> NDArray[np.float32]:
        signal = _array(samples, "samples").reshape(-1)
        output = np.empty_like(signal)
        for index, value in enumerate(signal):
            output[index] = self._sample(value)
        return output
