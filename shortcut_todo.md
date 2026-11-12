# shortcut branch roadmap

Special convex solvers for narrow problem classes. Generic AugLag/IPM/SLSQP remain the default when classification returns `None`. Special detection overrides `smoothing` and `merge_constraints`.

## 1. Architecture and routing

- `solvers/special/` package: `classify`, `canonicalize` (placeholder), `qcqp_one` (placeholder)
- `classify_problem` detects `qcqp_one` only
- `quadcoil()` routes special problems outside the whole-program JIT into `_quadcoil_special`
- Placeholders raise `NotImplementedError`
- Focused classify + routing tests

## 2. Support for `qcqp_one`

- Canonicalize: single quadratic objective + one scalar quadratic `<=` constraint
- Implement `solve_qcqp_one` (jittable linear-solve based method; no CVXPY)
- Return `fin_x`, duals, etc. and wire KKT adjoint on the special path
- Matching problems stop raising and return real solutions

## 3. Support for QP (no `linear_slack`)

- Detect QP: quadratic/linear objectives, affine constraints only
- Canonicalize into a `canonicalized` dict (no QuadraticProgram class)
- Host or JAX QP solve for problems without L∞/L1 slack reformulations
- Fall back to generic when unsupported

## 4. Support for QP with `linear_slack`

- Extend classification and canonicalization for `linear_slack` quantities
- Introduce slack variables and affine inequalities in the `canonicalized` dict
- Cover L∞ / L1 objectives and `<=` constraints built from affine fields

## 5. Performance testing, new tests, and green suite

- Benchmarks vs AugLag on convex special cases
- New unit tests for classify / canonicalize / special solves / KKT
- Restore full `unittest` suite (including cases that currently hit `qcqp_one` and raise)
