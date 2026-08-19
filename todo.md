# Deferred work (from vectorization / KKT adjoint redesign)

## Active-set masking / symmetrizing (not yet)

The current KKT Jacobian uses the complementarity form

```
J = [[H, A^T], [diag(z) A, diag(g)]]
```

which is poorly scaled and often needs `nan_to_num` / `well_posed=False`.
A masked active-set form is preferred eventually:

```
a_i = 1 if constraint i active else 0  (stop_gradient)
R = [grad_x f + A^T (a ⊙ z);  a ⊙ g - (1-a) ⊙ z]
```

**Important:** the active set depends on traced inputs, so a traced boolean
mask must **not** be used to slice rows (data-dependent shapes break JAX).
Apply the active set as a **fixed-shape mask**, or as a fixed-capacity
`lax.top_k` gather with an in-block mask so over-selection is safe.

Do not pursue matrix-free iterative solves unless dense factorization
becomes prohibitive.

## Unify constrained / unconstrained paths

Define a single residual over `w = [x, z_g, z_h]` so unconstrained is the
`m=0` special case. Drop the four-way branching in `stationarity_kkt` /
`adjoint_kkt`.

## Solver contract for differentiation

Every solver should return `(x_opt, z_g, z_h)` for the same `(f, g, h)` it
was given. Move `recover_multipliers` into `auglag.py` as postprocessing so
`_quadcoil_pure` no longer branches on `solver` when building KKT data.

## Equality constraints in ipm / slsqp sensitivity

`ipm` does not support `h_eq` yet. `slsqp`'s `fin_z` covers inequalities
only; equality multipliers are absent from the sensitivity, so gradients
with `'=='` constraints can be wrong for those solvers. Fix when unifying
the residual.

## Batch all metrics into one AD pass

Today each metric rebuilds `y_to_qp` / surface geometry inside `f_metric`
and gets its own `jacrev`. Concatenate metrics into one vector-valued
function so winding-surface generation and Fourier evaluations are traced
once. `full_jacobian` then becomes one more stacked entry.

## Factor-once / multi-RHS and forward vs adjoint

`lx.linear_solve` is called per metric row and re-factorizes. Prefer one
factorization + multi-RHS. Revive or delete dead `_choose_fwd_rev` in
`quadcoil.py` and pick forward vs adjoint from `(K_tot, n_y)` at trace time.

## Rank >= 2 metrics

`adjoint_kkt` flattens the metric for the Jacobian. `'value'` keeps the
original shape, but gradient leaves currently carry a flattened leading
axis of size `k`. Unflatten those leaves when higher-rank metrics are needed.

## Split `_quadcoil_pure`

Break the ~700-line body into helpers (`_resolve_solver_options`,
`_build_y_dict`, `_build_preconditioner`, `_run_solver`, `_evaluate_metrics`)
for readability once the above are done.
