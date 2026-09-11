from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from fssr_nam.reporting.json_evidence import (
    TYPE_KEY,
    dumps_strict_evidence,
    encode_extended_reals,
    is_extended_real_object,
    loads_strict_evidence,
)

EXTENDED_AND_FINITE_VALUES = (
    -math.inf,
    -1.0,
    -0.0,
    0.0,
    1.0,
    math.inf,
    math.nan,
    np.float32(-math.inf),
    np.float64(math.inf),
    np.float32(math.nan),
)

EVIDENCE_LEAVES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
    st.text(max_size=20),
    st.floats(width=64, allow_nan=True, allow_infinity=True),
)
EVIDENCE_VALUES = st.recursive(
    EVIDENCE_LEAVES,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(
            st.sampled_from(("a", "b", "diagnostic", "nested")),
            children,
            max_size=4,
        ),
    ),
    max_leaves=12,
)


@given(payload=EVIDENCE_VALUES)
@example(payload={"diagnostic": -math.inf, "nested": [math.inf, math.nan]})
@settings(max_examples=100, deadline=None)
def test_strict_evidence_codec_roundtrips_generated_supported_domain(
    payload: object,
) -> None:
    encoded = encode_extended_reals(payload)
    serialized = dumps_strict_evidence(payload, indent=None)
    assert json.loads(serialized) == encoded
    assert encode_extended_reals(loads_strict_evidence(serialized)) == encoded


@pytest.mark.parametrize("value", EXTENDED_AND_FINITE_VALUES)
def test_strict_evidence_codec_has_canonical_roundtrip_property(value: float) -> None:
    payload = {
        "scalar": value,
        "nested": [value, {"again": value}],
        "ordinary": {"none": None, "bool": True, "integer": 3},
    }
    encoded = encode_extended_reals(payload)
    serialized = dumps_strict_evidence(payload, indent=None)
    parsed_standard_json = json.loads(serialized)
    assert parsed_standard_json == encoded
    assert encode_extended_reals(loads_strict_evidence(serialized)) == encoded


def test_exact_zero_db_is_preserved_as_tagged_negative_infinity() -> None:
    encoded = encode_extended_reals({"periodicity_error_db": -math.inf})
    assert is_extended_real_object(encoded["periodicity_error_db"], token="-inf")
    assert math.isinf(
        loads_strict_evidence(dumps_strict_evidence(encoded))["periodicity_error_db"]
    )


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_strict_loader_rejects_nonstandard_bare_json_constants(constant: str) -> None:
    with pytest.raises(ValueError, match="non-standard JSON numeric constant"):
        loads_strict_evidence(f'{{"value":{constant}}}')


def test_codec_rejects_reserved_type_key_collision() -> None:
    with pytest.raises(ValueError, match="reserve key"):
        encode_extended_reals({TYPE_KEY: "user-value"})
