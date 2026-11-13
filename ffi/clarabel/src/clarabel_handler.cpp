// XLA FFI handler wrapping Clarabel.cpp for dense QP / SOCP.
//
// Inputs  (f64 buffers): P(n,n), q(n), A(p,n), b(p), G(m,n), h(m), settings(5,)
// Attr    (i32 span):    cone_attr = [n_blocks, tag0,a0,b0, ...]
// Outputs (f64):         x(n), y(p), z(m), s(m), info(5,)
//
// info = [pobj, dobj, status_code, iterations, solve_time]
// status codes match CONIC_STATUS in quadcoil.solvers.special.qp_form.

#include "xla/ffi/api/ffi.h"

#include <clarabel.h>   // Clarabel C API (from Clarabel.cpp)

#include <cmath>
#include <cstring>
#include <vector>

namespace ffi = xla::ffi;

namespace {

int MapStatus(ClarabelSolverStatus st) {
  switch (st) {
    case ClarabelSolved: return 0;
    case ClarabelAlmostSolved: return 1;
    case ClarabelPrimalInfeasible:
    case ClarabelAlmostPrimalInfeasible: return 2;
    case ClarabelDualInfeasible:
    case ClarabelAlmostDualInfeasible: return 3;
    case ClarabelMaxIterations:
    case ClarabelMaxTime: return 4;
    default: return 5;
  }
}

void FillNan(ffi::ResultBuffer<ffi::F64>& buf) {
  auto* p = buf->typed_data();
  for (size_t i = 0; i < buf->element_count(); ++i) p[i] = NAN;
}

}  // namespace

static ffi::Error ClarabelSolveImpl(
    ffi::Buffer<ffi::F64> P,
    ffi::Buffer<ffi::F64> q,
    ffi::Buffer<ffi::F64> A,
    ffi::Buffer<ffi::F64> b,
    ffi::Buffer<ffi::F64> G,
    ffi::Buffer<ffi::F64> h,
    ffi::Buffer<ffi::F64> settings,
    ffi::Buffer<ffi::S32> cone_attr,
    ffi::ResultBuffer<ffi::F64> x_out,
    ffi::ResultBuffer<ffi::F64> y_out,
    ffi::ResultBuffer<ffi::F64> z_out,
    ffi::ResultBuffer<ffi::F64> s_out,
    ffi::ResultBuffer<ffi::F64> info_out) {
  // NOTE: Full Clarabel C API wiring (CSC conversion, cone construction,
  // solve, status mapping) belongs here.  The extension builds against
  // Clarabel.cpp; until the C API surface is fully bound, return a clear
  // numerical-error status so callers fall back to the pure_callback path.
  (void)P; (void)q; (void)A; (void)b; (void)G; (void)h;
  (void)settings; (void)cone_attr;
  FillNan(x_out);
  FillNan(y_out);
  FillNan(z_out);
  FillNan(s_out);
  auto* info = info_out->typed_data();
  info[0] = NAN; info[1] = NAN; info[2] = 5.0; info[3] = -1.0; info[4] = NAN;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ClarabelSolve, ClarabelSolveImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());
