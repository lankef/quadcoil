"""Optional Clarabel XLA FFI extension for QUADCOIL."""

try:
    from ._clarabel_ffi import clarabel_solve
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'quadcoil_clarabel_ffi extension is not built. '
        'Install from ffi/clarabel (requires CMake, Rust, nanobind), '
        'or use the clarabel PyPI wheel via qp_cpu.pure_callback.'
    ) from e

__all__ = ['clarabel_solve']
