# Agent Guide: QUADCOIL

This repository is **QUADCOIL**, a global coil optimization code for stellarators using a **winding surface** approach. The core is JAX-first (JIT, autodiff) and interoperates with **simsopt** and **DESC**.

Use this document as the “how to work here safely” guide when making code changes as an AI agent.

## Quick commands

- **Install (editable/dev)**:

```bash
pip install -e .
```

- **Run all tests**:

```bash
cd tests && python -m unittest discover
```

- **Run a single test**:

```bash
cd tests && python -m unittest test_regcoil.QuadcoilKTest.test_regcoil
cd tests && python -m unittest test_desc.QuadcoilDESCTest.test_simple_wrapper
```

Notes:
- Tests use `unittest`. Some tests skip if optional deps (simsopt, DESC) aren’t installed.
- The current full solver stack depends on `jax`, `lineax`, `optimistix`, and `slsqp-jax`, and the package metadata now requires **Python 3.11+**.

## Mental model (architecture + data flow)

### Primary entry point

- **`quadcoil()`** in `src/quadcoil/quadcoil.py` is the main optimizer entry point.
- It is **JIT-compiled** with many **static arguments** listed in `QUADCOIL_STATIC_ARGNAMES`.
- It takes plasma + winding surface specs + objective/constraint configuration and returns objective values, `QuadcoilParams`, current potential coefficients, and solver status/metrics.

### Data flow (high level)

```
quadcoil()
  → build QuadcoilParams (plasma + winding surface + quadrature)
  → parse objective/constraint strings via wrapper.py
  → dispatch into src/quadcoil/solvers/ (constrained or unconstrained)
  → return solution + metrics
```

### Repository map (where things live)

- **Core orchestration**
  - `src/quadcoil/quadcoil.py`: main optimizer, static-vs-traced separation, objective/constraint plumbing
  - `src/quadcoil/quadcoil_params.py`: `QuadcoilParams` pytree (surfaces, currents, quadrature, dof packing)
  - `src/quadcoil/wrapper.py`: resolves string names into quantity objects/callables; merges callables; quadpoint defaults/validation
  - `src/quadcoil/solvers/__init__.py`: public solver exports
  - `src/quadcoil/solvers/auglag.py`: augmented-Lagrangian solver; unconstrained inner solve now uses `optimistix.LBFGS`
  - `src/quadcoil/solvers/slsqp.py`: SLSQP wrapper built on `slsqp-jax` and `optimistix.minimise`
  - `src/quadcoil/solvers/ipm.py`: interior-point solver
  - `src/quadcoil/solvers/kkt_adjoint.py`: shared KKT stationarity/adjoint machinery used by multiple solvers

- **Surfaces**
  - `src/quadcoil/surface.py`: all JAX-native surface types live here (`SurfaceJAX`, `SurfaceRZFourierJAX`, `SurfaceXYZTensorFourierJAX`, `SurfaceXYZFourierJAX`)
  - `src/quadcoil/winding_surface.py`: winding surface generators (offset/arc/atan variants), fit-to-surface workflows

- **Physics quantities (objectives/constraints)**
  - `src/quadcoil/quantities/`: all physical quantities as `_Quantity` instances (see below)

- **Interfaces**
  - `src/quadcoil/io/`: adapters for DESC / simsopt / JAX IO and other integrations

- **Tests**
  - `tests/`: `unittest` suite validating operators, wrappers, and (optionally) DESC integration

## JAX invariants (do not break these)

These are the most common ways to accidentally “break the codebase” even if unit tests pass locally.

- **Static vs traced arguments**
  - Any option that changes array shapes, symmetry, mode counts, solver selection, or control-flow should be treated as **static** under JIT.
  - If you add a new option that must be static, ensure it is included in `QUADCOIL_STATIC_ARGNAMES` (and propagated where needed).
  - `lbfgs_memory` is now a top-level static argument to `quadcoil()` and the solver wrappers; do **not** move it back into `solver_options`, because it affects JAX-visible array shapes.

- **Pure functions + no side effects**
  - Objective and constraint implementations must be **functional**: output depends only on inputs.
  - Avoid hidden global state; avoid mutating Python containers inside jitted code paths.

- **Use `jax.numpy` (`jnp`) in traced computations**
  - Avoid `numpy` operations on traced values.
  - Prefer `jax.lax` / `vmap` / `jit` patterns over Python loops inside performance-critical kernels.

- **Pytrees stay pytrees**
  - `QuadcoilParams` and surface classes are designed to be JAX pytrees.
  - Don’t introduce non-pytree fields into objects that are passed through JIT/autodiff unless you understand the pytree registration impact.
  - For surfaces specifically, keep `tree_flatten` / `tree_unflatten` on each concrete subclass. Even though the implementations are similar, lifting them to `SurfaceJAX` breaks subclass reconstruction under JAX pytree registration.

- **Numerical dtype expectations**
  - The codebase enables **64-bit** (`jax_enable_x64=True`) in core modules. Don’t silently change precision unless explicitly required and tested.

## How objectives/constraints work (string → quantity → callable)

QUADCOIL commonly takes objective/constraint terms as **string names** (e.g. `'f_B'`), which are resolved via `src/quadcoil/wrapper.py`:

- `get_quantity(name)` looks up an attribute in `quadcoil.quantities` and requires it to be an instance of `_Quantity`.
- `merge_callables(...)` combines multiple callables into a single callable by concatenating flattened outputs (and can optionally “merge” inequalities under smoothing).

If you add a new objective/constraint term, you usually need to:
- implement it as a `_Quantity` instance in `src/quadcoil/quantities/`
- export it from `src/quadcoil/quantities/__init__.py` so `get_quantity()` can find it
- add/adjust tests in `tests/` (or ensure existing wrapper tests cover it)

## Adding a new physical quantity (recommended workflow)

1. **Create the quantity** in `src/quadcoil/quantities/` as an instance of a class inheriting `_Quantity`.
2. **Define compatibility** correctly (where it can be used): objective (`'f'`) and/or constraints (`'<='`, `'>='`, `'=='`).
3. **Provide both “raw” and “scaled” implementations** if the base class expects them:
   - “raw” formulations are typically C⁰ and intuitive
   - “scaled” formulations are typically C¹-friendly and may introduce slack variables for smooth constrained optimization
4. **Export it** in `src/quadcoil/quantities/__init__.py`.
5. **Add tests** (or update existing ones) to cover:
   - string resolution via `get_quantity`
   - shape + dtype stability under JIT
   - behavior with/without smoothing modes (if applicable)

## Common pitfalls (what to watch for in PRs)

- **Changing shapes inside JIT** (e.g., building arrays whose size depends on traced values) will often compile but fail at runtime or explode compilation time.
- **Python conditionals** on traced values: use `jax.lax.cond` / `jax.lax.switch`.
- **Debug printing**: use `jax.debug.print` guarded by a `verbose` flag; avoid noisy prints in jitted hot paths.
- **Optional dependencies**: keep DESC/simsopt integrations import-safe; tests may run without them.
- **Namespace exports**: `src/quadcoil/__init__.py` exports many symbols via `import *`. If you add a new public API, consider whether it should be exported and ensure it doesn’t cause circular imports.
- **Legacy solver paths**: `auglag.py` now keeps legacy implementations alongside new ones. Preserve the `_legacy` functions unless you are intentionally deleting a compatibility path.
- **KKT differentiation**: the AugLag and SLSQP paths now recover multipliers from stationarity/KKT systems and route adjoint differentiation through `solvers/kkt_adjoint.py`. Reuse that shared machinery instead of duplicating solver-specific adjoint logic.
- **Surface abstractions**: `SurfaceJAX` now owns the common `__init__`, `dof_to_gamma`, `gammadash`, `fit_dofs_from_gamma`, and `copy_and_set_quadpoints` behavior. Subclasses should mainly provide `dof_to_gamma_op`, `_build_surface_fit_matrices`, and their own pytree registration methods.

## Where to start when investigating a bug

- **API-level behavior**: `src/quadcoil/quadcoil.py` (`quadcoil()`) and `src/quadcoil/wrapper.py` (parsing + resolution)
- **Solver behavior**:
  - `src/quadcoil/solvers/auglag.py` for AugLag penalty updates, multiplier recovery, and inner L-BFGS behavior
  - `src/quadcoil/solvers/slsqp.py` for `slsqp-jax` configuration/plumbing
  - `src/quadcoil/solvers/ipm.py` for interior-point logic
  - `src/quadcoil/solvers/kkt_adjoint.py` for shared stationarity/adjoint differentiation
- **Geometry / surface issues**: `src/quadcoil/surface.py` and `src/quadcoil/winding_surface.py`
- **A specific objective/constraint**: locate its `_Quantity` implementation in `src/quadcoil/quantities/`

## Recent changes (May 2026)

### Dependency refactoring

**Core dependencies updated** (see `pyproject.toml`):
- **Removed**: `optax` (legacy solver code was removed)
- **Added**: `optimistix >= 0.0.1` (required for auglag-lbfgs and slsqp solvers)
- **Added**: `slsqp-jax >= 0.0.1` (required for slsqp solver)
- **Made optional**: `scipy` and `matplotlib`

**Optional dependency groups**:
```toml
[project.optional-dependencies]
visualization = ["matplotlib"]          # For plotting.py functions
coil-cutting = ["scipy", "matplotlib"]  # For io/coil_cutting.py
all = ["matplotlib", "scipy"]           # Everything
```

**What changed**:
- `scipy.constants.mu_0` is now hardcoded in `math_utils.py` as `mu_0 = 1.25663706127e-06`
- `io/desc.py` and `quantities/current.py` now import `mu_0` from `math_utils`
- `io/coil_cutting.py` checks for scipy/matplotlib availability and raises helpful errors
- `plotting.py` checks for matplotlib availability and raises helpful errors

**Installation**:
```bash
pip install quadcoil                # Core only
pip install quadcoil[visualization]  # Add plotting support
pip install quadcoil[coil-cutting]   # Add coil cutting + plotting
pip install quadcoil[all]            # Everything
```

### Plane fitting utilities in `math_utils.py`

**New functions** for projecting 3D points onto least-squares fit planes:
- `project_points_to_plane(gamma_pol)` - Fits plane via SVD, projects points, returns 2D coordinates
- `reconstruct_3d_from_plane(x_pol, y_pol, plane_data)` - Inverse operation
- `plane_fitting_error(gamma_pol, plane_data)` - Computes RMS and max fitting errors

All are JIT-compiled and vmap-compatible. Used by the generalized winding surface generator.

### Generalized winding surface generator

**New function**: `gen_winding_surface_general()` in `winding_surface.py`

This extends `gen_winding_surface_arc` to support multiple surface types with automatic plane fitting:

**Behavior**:
- For `cls=SurfaceRZFourierJAX`: Uses direct cylindrical R,Z extraction (backward compatible)
- For other surface types (`SurfaceXYZTensorFourierJAX`, `SurfaceXYZFourierJAX`): 
  - Applies plane fitting to each toroidal slice using `project_points_to_plane`
  - Extracts better R,Z coordinates from plane-fitted coordinates
  - R,Z values are only used for self-intersection filtering (`weight_remove_invalid`)

**Usage**:
```python
from quadcoil.winding_surface import gen_winding_surface_general
from quadcoil import SurfaceXYZTensorFourierJAX

dofs = gen_winding_surface_general(
    plasma_gamma, d_expand=0.1,
    nfp=3, stellsym=True,
    mpol=5, ntor=5,
    cls=SurfaceXYZTensorFourierJAX  # Automatically uses plane fitting
)
```

The original `gen_winding_surface_arc` remains unchanged for comparison and backward compatibility.

