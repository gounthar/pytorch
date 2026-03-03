import functools
import logging

from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
)
from .registry import _RegisterFn, register_op_registerer


log = logging.getLogger(__name__)


@functools.cache
def _check_runtime_available() -> tuple[bool, tuple[int, int, int] | None]:
    """
    Check if cutedsl (and deps) are available.

    NOTE: Doesn't import at this point
    """
    deps = [
        ("nvidia_cutlass_dsl", "cutlass"),
        ("apache_tvm_ffi", "tvm_ffi"),
        ("cuda_bindings", "cuda.bindings.driver"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_version("nvidia_cutlass_dsl")
    else:
        log.info(
            "CuTeDSL operators require optional Python packages "
            "`nvidia-cutlass-dsl`, `apache-tvm-ffi`, and `cuda-bindings` "
            "(from NVIDIA cuda-python); %s",
            reason,
        )
        available = False
        version = None
    return available, version


def runtime_available() -> None | bool:
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> None | tuple[int, int, int]:
    _, version = _check_runtime_available()
    return version


def register_op(fn: _RegisterFn) -> None:
    available, _ = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    register_op_registerer(fn)
