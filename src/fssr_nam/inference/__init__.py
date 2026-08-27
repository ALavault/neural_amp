"""Export and reference inference helpers for native FSSR models."""

from .r1_native import (
    R1NativeReference,
    build_r1_native_payload,
    export_r1_native_model,
    load_r1_native_payload,
    validate_r1_native_payload,
    write_r1_native_payload,
)

__all__ = [
    "R1NativeReference",
    "build_r1_native_payload",
    "export_r1_native_model",
    "load_r1_native_payload",
    "validate_r1_native_payload",
    "verify_r1_cpp_parity",
    "write_r1_native_payload",
]


def __getattr__(name: str) -> object:
    if name == "verify_r1_cpp_parity":
        from .r1_parity import verify_r1_cpp_parity

        return verify_r1_cpp_parity
    raise AttributeError(name)
