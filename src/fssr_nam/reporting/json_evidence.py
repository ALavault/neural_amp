"""Canonical strict-JSON transport for scientific extended-real diagnostics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

TYPE_KEY = "__fssr_nam_type__"
VALUE_KEY = "value"
EXTENDED_REAL_TYPE = "extended-real-v1"
EXTENDED_REAL_TOKENS = frozenset({"-inf", "+inf", "nan"})


def _extended_real(token: str) -> dict[str, str]:
    return {TYPE_KEY: EXTENDED_REAL_TYPE, VALUE_KEY: token}


def encode_extended_reals(value: Any) -> Any:
    """Convert a nested evidence value to a strict-JSON-compatible structure."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return _extended_real("nan")
        if value == math.inf:
            return _extended_real("+inf")
        if value == -math.inf:
            return _extended_real("-inf")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        if TYPE_KEY in value:
            if (
                set(value) == {TYPE_KEY, VALUE_KEY}
                and value.get(TYPE_KEY) == EXTENDED_REAL_TYPE
                and value.get(VALUE_KEY) in EXTENDED_REAL_TOKENS
            ):
                return dict(value)
            raise ValueError(f"evidence mappings reserve key {TYPE_KEY!r}")
        if not all(isinstance(key, str) for key in value):
            raise TypeError("scientific evidence mapping keys must be strings")
        return {key: encode_extended_reals(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [encode_extended_reals(item) for item in value]
    raise TypeError(f"unsupported scientific evidence type: {type(value).__name__}")


def _decode_extended_real(value: Mapping[str, Any]) -> float | None:
    if value.get(TYPE_KEY) != EXTENDED_REAL_TYPE:
        return None
    if set(value) != {TYPE_KEY, VALUE_KEY}:
        raise ValueError("extended-real objects require exactly type and value")
    token = value.get(VALUE_KEY)
    if token not in EXTENDED_REAL_TOKENS:
        raise ValueError(f"unknown extended-real token: {token!r}")
    return {"-inf": -math.inf, "+inf": math.inf, "nan": math.nan}[token]


def decode_extended_reals(value: Any) -> Any:
    """Decode the canonical structure back to Python extended-real values."""
    if isinstance(value, Mapping):
        decoded = _decode_extended_real(value)
        if decoded is not None:
            return decoded
        return {key: decode_extended_reals(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_extended_reals(item) for item in value]
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def dumps_strict_evidence(value: Any, *, indent: int | None = 2) -> str:
    """Serialize evidence only after canonical extended-real encoding."""
    encoded = encode_extended_reals(value)
    return json.dumps(
        encoded,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
        separators=None if indent is not None else (",", ":"),
    )


def loads_strict_evidence(serialized: str) -> Any:
    """Parse standard JSON and decode canonical extended-real objects."""
    return decode_extended_reals(loads_strict_json(serialized))


def loads_strict_json(serialized: str) -> Any:
    """Parse strict standard JSON while retaining canonical tagged objects."""
    return json.loads(serialized, parse_constant=_reject_json_constant)


def is_extended_real_object(value: Any, *, token: str | None = None) -> bool:
    """Return whether a JSON value is one canonical extended-real object."""
    if not isinstance(value, Mapping):
        return False
    if set(value) != {TYPE_KEY, VALUE_KEY} or value.get(TYPE_KEY) != EXTENDED_REAL_TYPE:
        return False
    observed = value.get(VALUE_KEY)
    return observed in EXTENDED_REAL_TOKENS and (token is None or observed == token)
