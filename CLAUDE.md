# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QUADCOIL is a global coil optimization code for stellarators using a winding surface approach. It optimizes coil designs by minimizing the complexity of sheet current distributions on a winding surface surrounding the plasma. Supports constrained optimization, automatic differentiation through all physics quantities, and interoperability with simsopt and DESC.

## Commands

**Install (development mode):**
```bash
pip install -e .
```

**Run all tests:**
```bash
cd tests && python -m unittest discover
```

**Run a single test:**
```bash
cd tests && python -m unittest test_regcoil.QuadcoilKTest.test_regcoil
cd tests && python -m unittest test_desc.QuadcoilDESCTest.test_simple_wrapper
```

Tests use the stdlib `unittest` framework. Some tests skip automatically if optional deps (simsopt, DESC) are unavailable.

## Architecture

### Entry Point

`quadcoil()` in `src/quadcoil/quadcoil.py` is the main optimizer. It is JIT-compiled with ~23 static arguments (listed in `QUADCOIL_STATIC_ARGNAMES`). It takes plasma/winding surface specs and optimization parameters, returns objective values, `QuadcoilParams`, current potential coefficients, and solver status.

### Data Flow

```
quadcoil()
  → creates QuadcoilParams (plasma + winding surface + quadrature)
  → parses objectives/constraints via wrapper.py
  → calls solve_constrained() or run_opt_lbfgs() in solver.py
  → returns solution + metrics
```

### Key Modules

| Module | Role |
|--------|------|
| `quadcoil.py` | Main entry; JIT-compiled optimizer (~1300 lines) |
| `quadcoil_params.py` | `QuadcoilParams` pytree class (surfaces, currents, quadrature points) |
| `solver.py` | `run_opt_lbfgs`, `run_opt_optax`, `solve_constrained` |
| `wrapper.py` | String-based objective/constraint parsing; `merge_callables()` |
| `surface.py` | `SurfaceRZFourierJAX` — JAX-native Fourier surface; converts to/from simsopt and DESC |
| `winding_surface.py` | Winding surface generators (`gen_winding_surface_offset`, `_arc`, `_atan`) |
| `math_utils.py` | Vector ops, linear solvers, activation functions (notably `gplus`) |
| `quantity/` | Physical quantities (see below) |
| `io/` | Interfaces: `desc.py`, `simsopt.py`, `jax.py`, `coil_cutting.py`, `focus.py` |

### `quantity/` Module

Each physical quantity (B-normal, curvature K·∇K, self-force, regularization, etc.) inherits from `_Quantity` in `quantity/quantity.py`. The base class manages switching between:
- **C⁰ "raw" formulations** — intuitive implementations, no smoothing
- **C¹ "scaled" formulations** — use slack variables for smooth constrained optimization

Class methods `generate_c2()`, `generate_linf_norm()`, `generate_huber()` build objective/constraint callables from the base quantity.

Objectives and constraints are referenced by string names (e.g., `'f_B'`) and resolved from `quadcoil.quantity` namespace in `wrapper.py`.

### JAX Design Conventions

- All core computations use JAX (JIT, `vmap`, `grad`, `custom_vjp`)
- `QuadcoilParams` and `SurfaceRZFourierJAX` are registered as JAX pytrees
- Static vs. traced argument separation is critical for JIT — physical shapes/symmetries/solver options are static; current potential coefficients are traced
- `@partial(jit, static_argnames=...)` pattern is used throughout

### DESC Interface

`io/desc.py` contains DESC-specific scaling and unit conversions. The DESC interface has its own unit definitions inside each `_Quantity` subclass for automatic scaling.
