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
  → solve via solver.py (constrained or unconstrained)
  → return solution + metrics
```

### Repository map (where things live)

- **Core orchestration**
  - `src/quadcoil/quadcoil.py`: main optimizer, static-vs-traced separation, objective/constraint plumbing
  - `src/quadcoil/quadcoil_params.py`: `QuadcoilParams` pytree (surfaces, currents, quadrature, dof packing)
  - `src/quadcoil/wrapper.py`: resolves string names into quantity objects/callables; merges callables; quadpoint defaults/validation
  - `src/quadcoil/solver.py`: optimizers (`run_opt_lbfgs`, other Optax-based solvers; constrained solve helpers)

- **Surfaces**
  - `src/quadcoil/surface.py`: JAX-native surface types; includes `SurfaceRZFourierJAX` and surface geometry interface
  - `src/quadcoil/winding_surface.py`: winding surface generators (offset/arc/atan variants), fit-to-surface workflows

- **Physics quantities (objectives/constraints)**
  - `src/quadcoil/quantity/`: all physical quantities as `_Quantity` instances (see below)

- **Interfaces**
  - `src/quadcoil/io/`: adapters for DESC / simsopt / JAX IO and other integrations

- **Tests**
  - `tests/`: `unittest` suite validating operators, wrappers, and (optionally) DESC integration

## JAX invariants (do not break these)

These are the most common ways to accidentally “break the codebase” even if unit tests pass locally.

- **Static vs traced arguments**
  - Any option that changes array shapes, symmetry, mode counts, solver selection, or control-flow should be treated as **static** under JIT.
  - If you add a new option that must be static, ensure it is included in `QUADCOIL_STATIC_ARGNAMES` (and propagated where needed).

- **Pure functions + no side effects**
  - Objective and constraint implementations must be **functional**: output depends only on inputs.
  - Avoid hidden global state; avoid mutating Python containers inside jitted code paths.

- **Use `jax.numpy` (`jnp`) in traced computations**
  - Avoid `numpy` operations on traced values.
  - Prefer `jax.lax` / `vmap` / `jit` patterns over Python loops inside performance-critical kernels.

- **Pytrees stay pytrees**
  - `QuadcoilParams` and surface classes are designed to be JAX pytrees.
  - Don’t introduce non-pytree fields into objects that are passed through JIT/autodiff unless you understand the pytree registration impact.

- **Numerical dtype expectations**
  - The codebase enables **64-bit** (`jax_enable_x64=True`) in core modules. Don’t silently change precision unless explicitly required and tested.

## How objectives/constraints work (string → quantity → callable)

QUADCOIL commonly takes objective/constraint terms as **string names** (e.g. `'f_B'`), which are resolved via `src/quadcoil/wrapper.py`:

- `get_quantity(name)` looks up an attribute in `quadcoil.quantity` and requires it to be an instance of `_Quantity`.
- `merge_callables(...)` combines multiple callables into a single callable by concatenating flattened outputs (and can optionally “merge” inequalities under smoothing).

If you add a new objective/constraint term, you usually need to:
- implement it as a `_Quantity` instance in `src/quadcoil/quantity/`
- export it from `src/quadcoil/quantity/__init__.py` so `get_quantity()` can find it
- add/adjust tests in `tests/` (or ensure existing wrapper tests cover it)

## Adding a new physical quantity (recommended workflow)

1. **Create the quantity** in `src/quadcoil/quantity/` as an instance of a class inheriting `_Quantity`.
2. **Define compatibility** correctly (where it can be used): objective (`'f'`) and/or constraints (`'<='`, `'>='`, `'=='`).
3. **Provide both “raw” and “scaled” implementations** if the base class expects them:
   - “raw” formulations are typically C⁰ and intuitive
   - “scaled” formulations are typically C¹-friendly and may introduce slack variables for smooth constrained optimization
4. **Export it** in `src/quadcoil/quantity/__init__.py`.
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

## Where to start when investigating a bug

- **API-level behavior**: `src/quadcoil/quadcoil.py` (`quadcoil()`) and `src/quadcoil/wrapper.py` (parsing + resolution)
- **Solver behavior**: `src/quadcoil/solver.py` (stopping criteria, optax wrappers, constrained solve)
- **Geometry / surface issues**: `src/quadcoil/surface.py` and `src/quadcoil/winding_surface.py`
- **A specific objective/constraint**: locate its `_Quantity` implementation in `src/quadcoil/quantity/`

