cmake_minimum_required(VERSION 3.20)
project(quadcoil_clarabel_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# JAX FFI headers (xla/ffi/api/ffi.h)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jax.ffi; print(jax.ffi.include_dir())"
  OUTPUT_VARIABLE JAX_FFI_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "JAX FFI include: ${JAX_FFI_INCLUDE}")

# Clarabel.cpp (pinned).  Requires Rust/cargo on the build machine.
include(FetchContent)
FetchContent_Declare(
  clarabel_cpp
  GIT_REPOSITORY https://github.com/oxfordcontrol/Clarabel.cpp.git
  GIT_TAG        0.11.1
  GIT_SHALLOW    TRUE
)
set(CLARABEL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(clarabel_cpp)

nanobind_add_module(_clarabel_ffi src/bindings.cpp src/clarabel_handler.cpp)
target_include_directories(_clarabel_ffi PRIVATE
  "${JAX_FFI_INCLUDE}"
  "${clarabel_cpp_SOURCE_DIR}/include"
)
target_link_libraries(_clarabel_ffi PRIVATE libclarabel_c_shared)

install(TARGETS _clarabel_ffi LIBRARY DESTINATION quadcoil_clarabel_ffi)
