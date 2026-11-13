#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Defined by XLA_FFI_DEFINE_HANDLER_SYMBOL in clarabel_handler.cpp.
extern "C" void ClarabelSolve(void);

NB_MODULE(_clarabel_ffi, m) {
  m.def("clarabel_solve", []() {
    // Return the handler as a PyCapsule for jax.ffi.register_ffi_target.
    return nb::capsule(reinterpret_cast<void*>(&ClarabelSolve),
                       "xla._CUSTOM_CALL_TARGET");
  }, "XLA FFI capsule for ClarabelSolve");
}
