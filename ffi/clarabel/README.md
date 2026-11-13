# quadcoil_clarabel_ffi

Optional XLA FFI extension that wraps [Clarabel.cpp](https://github.com/oxfordcontrol/Clarabel.cpp)
for the QUADCOIL special QP/SOCP path (`special_backend='clarabel'`).

## Status

The Python module [`quadcoil.solvers.special.qp_cpu`](../../src/quadcoil/solvers/special/qp_cpu.py)
already provides a jittable Clarabel backend via `jax.pure_callback` around the
`clarabel` PyPI wheel.  That path is the default fallback and is what the tests
and topology benchmark use.

This package is the compiled FFI alternative.  Build it when you want to avoid
the Python callback overhead or to ship a self-contained Clarabel binary.

## Build

Requires: CMake ≥ 3.20, a C++17 compiler, Rust/cargo (for Clarabel's core),
nanobind, and a matching JAX install (`jax.ffi.include_dir()`).

```bash
cd ffi/clarabel
pip install .
```

CMake fetches Clarabel.cpp at a pinned tag.  Build separately for each JAX
environment (the XLA FFI headers must match):

```bash
# default env (e.g. JAX 0.9.x)
pip install .

# desc env (JAX 0.6.x)
/scratch/miniconda3/envs/desc/bin/pip install .
```

## Handler contract

`clarabel_solve` takes dense float64 buffers `(P, q, A, b, G, h, settings)` plus
an int32 cone-spec attribute, and returns `(x, y, z, s, info)` with the shared
status codes of `CONIC_STATUS`.

## GPU

CPU only.  Run with `JAX_PLATFORMS=cpu` when using this backend.  A CUDA
copy-to-host shim is future work.
