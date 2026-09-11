"""Export and reference inference helpers for native FSSR models."""

from .r1_native import (
    R1NativeReference,
    build_r1_native_payload,
    export_r1_native_model,
    load_r1_native_payload,
    validate_r1_native_payload,
    write_r1_native_payload,
)
from .r2_native import (
    R2NativeReference,
    build_r2_native_payload,
    export_r2_native_model,
    load_r2_native_payload,
    validate_r2_native_payload,
    write_r2_native_payload,
)

__all__ = [
    "R1NativeReference",
    "R2NativeReference",
    "build_r1_native_payload",
    "build_r2_native_payload",
    "export_r1_native_model",
    "export_r2_native_model",
    "load_r1_native_payload",
    "load_r2_native_payload",
    "validate_r1_native_payload",
    "validate_r2_native_payload",
    "verify_r1_cpp_parity",
    "verify_r2_cpp_parity",
    "write_r1_native_payload",
    "write_r2_native_payload",
]


def __getattr__(name: str) -> object:
    if name == "verify_r1_cpp_parity":
        from .r1_parity import verify_r1_cpp_parity

        return verify_r1_cpp_parity
    if name == "verify_r2_cpp_parity":
        from .r2_parity import verify_r2_cpp_parity

        return verify_r2_cpp_parity
    raise AttributeError(name)
