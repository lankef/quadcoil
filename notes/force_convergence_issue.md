# Force-minimization convergence issue (DESC QUADCOIL tutorial)

Date: 2026-10-26

Notebook: `DESC/docs/notebooks/tutorials/coil_optimization_QUADCOIL.ipynb`

## Summary

The `lineax` error in the quasi-single-stage cell is a downstream symptom. The
root problem is that QUADCOIL's force-minimization solve (`f_l1_force_cyl`
objective) does not converge, so its derivatives come out as NaN. When DESC
builds its first Jacobian for the `k=4` continuation step, those NaNs reach a
`lineax` linear solve inside QUADCOIL, which raises. DESC itself does not use
`lineax`.

Observed error:

```
JaxRuntimeError: INTERNAL: CpuCallback error calling callback: ...
_EquinoxRuntimeError: A linear solver received non-finite (NaN or inf) input
and cannot determine a solution.
```

raised from `lsqtr` -> `LinearConstraintProjection.jac_scaled_error` ->
`ProximalProjection._jvp` -> `_proximal_jvp_blocked_pure`.

## How the error is reached

- `QuadcoilProxy` has one output (`dim_f = 1`), so DESC differentiates it in
  reverse mode. That goes through QUADCOIL's custom JVP
  (`quadcoil.io.jax.gen_quadcoil_for_diff`), which re-runs the full QUADCOIL
  solve with `value_only=False` and then the KKT adjoint.
- The adjoint is evaluated at a point that is not a solution, so the
  derivatives are NaN. Reproduced outside the optimizer: a single
  `jax.jvp` of `QuadcoilProxy.compute` gives a finite value but a NaN
  derivative, on both the original ARIES-CS equilibrium and the saved step-3
  equilibrium (`data_aries_force/f_force_l1_eq_3.h5`).
- In the forward-mode test the NaN comes back silently. In DESC's reverse pass
  it reaches a `lineax` solve with `throw=True` (lineax raises
  `nonfinite_input` when both the right-hand side and the solution are
  non-finite). Not confirmed which call. Candidates are the
  `lx.linear_solve(..., lx.SVD())` calls in `winding_surface.py` and
  `nescoil.py`, and `safe_linear_solve` in `math_utils.py`.

## Evidence the force-minimization solve is broken

- Notebook cell 53 (problem 2, low-force filament) already shows it:
  - chi^2_B = 48633, against a constraint of <= 3.33.
  - L1 force = 2.9e16, about 1e6 times worse than the reference 3.5e10.
- QUADCOIL solver status at ARIES-CS (`solver='auglag-lbfgs'`):
  - `fin_f` is `inf` (`-inf` on the step-3 equilibrium), which is impossible
    for a smoothed L1 norm.
  - `last_inner_dx`, `last_inner_du`, `last_inner_dl` are NaN.
  - `last_inner_niter = 500` (inner L-BFGS hits its cap every outer loop).
  - `fin_g` is roughly `[1.5e4, 1.3e5, 3.0e7]`: all three constraints violated
    by factors of 1e4 to 1e7.
  - `fin_mu` is roughly `[2e5, 2e6, 5e8]`: multipliers blowing up.
- Step 3 of the continuation only looked fine because it was loaded from cache
  ("Step 3 exists"). That cache was written at 14:01 on 2026-10-26 with the
  same broken proxy; the quasi-symmetry error went from within bounds to 9834.
  The files in `DESC/docs/notebooks/tutorials/data_aries_force/` should be
  treated as bad.

## Ruled out

Each was rerun on the same force problem.

| Suspect | Result |
|---|---|
| QUADCOIL uncommitted changes (`max_lse` rewrite, cached field operators) | QUADCOIL `HEAD` (`35c03d2`) fails the same way |
| Any recent QUADCOIL commit | Commit `4f3df32` (2026-07-14), before the last good run, also fails |
| Reducing ARIES-CS to `L=M=N=6` | Full-resolution ARIES-CS also fails |
| `jax`, `equinox`, `lineax`, `optimistix` versions | Unchanged since April/May 2026 |
| DESC `QuadcoilProxy` wrapper | Only cosmetic changes since 2026-08-12 (move to `desc.external`) |
| Old `max_lse` (without `log(n)` scaling) | Still fails |

## Main lead: the last good run used a setting that no longer exists

- The committed notebook outputs, run on 2026-08-12, show this problem
  converging: chi^2_B = 3.61, exactly at its target, with L1 force 3.30e10,
  lower than the reference 3.68e10.
- In that version, `quadcoil_kwargs_force_l1` included
  `"max_linesearch_steps": 80`, with a comment that force optimization needs
  more L-BFGS line-search steps. DESC commit `17c83ed6c` removed it 18 minutes
  after that run.
- Current QUADCOIL source has no `max_linesearch_steps` anywhere. It was
  removed in May 2026 when the augmented Lagrangian switched to
  `optimistix.LBFGS` (commits `f1ead19`, `0be8106`). The current
  `optx.LBFGS` call exposes only `rtol`, `atol`, and `history_length`.
- This suggests the 2026-08-12 run used a different, older QUADCOIL install.
  Since even the 2026-07-14 commit fails, the problem may have been solvable
  only with the older solver. Confirmed by the inner-solver study below: the
  pre-May optax L-BFGS solves this problem, and the current optimistix L-BFGS
  does not.

## Unrelated bug in the uncommitted `max_lse` change

`math_utils.max_lse` now scales epsilon by `log(x.size)`, the size of the whole
array. When called from `abs_lse`, the reduction is only over a stacked axis of
length 2 (`axis=-1`). For a 33x33x3 field that makes the smoothing about 9
times sharper than intended. It is not the cause of this issue: restoring the
old `max_lse` still fails.

## Other notebook differences since the last good run

These do not cause the error, but will change results:

- DESC optimizer changed from `proximal-lsq-auglag` to `proximal-lsq-exact`.
- `maxiter` per continuation step changed from 30 to 2.

## Suggested next steps

1. Delete `DESC/docs/notebooks/tutorials/data_aries_force/` so the bad step-3
   cache is not reused.
2. Replace the inner unconstrained solve with `optax.lbfgs` (zoom line
   search, 20 steps) and keep the current outer augmented Lagrangian.
   Confirmed at full ARIES-CS resolution below. Do not return to the
   pre-May outer loop, and do not keep `optimistix.LBFGS` as the inner
   solver. A zero start does not replace NESCOIL.
3. Optionally make the derivative path fail loudly when the forward solve did
   not converge, instead of producing NaN derivatives.

## Inner solver study

Date: 2026-10-27. No library code was changed. The harness is in
`quadcoil/scratch_conv/` (`capture.py`, `auglag_harness.py`,
`inner_solvers.py`, `run_study.py`). It rebuilds the notebook force-L1
problem at ARIES-CS with `L=M=N=6` and runs the same outer augmented
Lagrangian (`c` starts at 1 and grows by 1.1 only after a converged inner
solve that still violates a constraint). Each run is capped at 8 outer
iterations. The first inner problem (`c=1`, multipliers zero) is traced
step by step. Plots: `scratch_conv/results/inner0_accepted.png` (accepted
steps of that first problem) and `scratch_conv/results/inner0_traces.png`
(every recorded evaluation; the optimistix curves there include rejected
trials).

Captured starts, `x` length 144:

- NESCOIL: `||x||=33.67`, raw force 1.362 times the reference unit, `f_B`
  already under the target, current constraints violated (`g = [-0.63, 54.2, 11.7]`).
- Zeros: `||x||=0`, raw force 0.742 times the unit, `f_B` about 100 times
  the target. Under the SVD preconditioner `phi_unit` is zero here, so this
  is not the same scaling as the NESCOIL start.

The reference unit is the best-case L1 force, about `3.55e10`. The `f_B`
target is 3.335. The 2026-08-12 notebook (full resolution) had a target near
3.61 and a force of `3.30e10`; the numbers below are the `L=M=N=6` problem,
so they are not identical, but the success criterion is the same: `f_B` at
or under its target and L1 force below the reference.

### What was compared

- **A.** Current solver: `optimistix.LBFGS`, Armijo backtracking, `max_steps=500`.
  Every line-search trial counts as one step. `num_accepted_steps` counts
  only accepted updates.
- **B20 / B80.** Pre-May solver: `optax.lbfgs` with a zoom (strong Wolfe)
  line search, `max_linesearch_steps` 20 or 80. One optax step is one
  accepted update. 500 is the iteration cap.
- **C.** `optax.lbfgs` with Armijo backtracking (decrease 0.5, slope 0.1,
  same as optimistix), so the line search is separated from the library.
- **Stopping.** `cauchy` is optimistix's rule (step and change in `f` small
  in the max norm). `legacy` is the pre-May `fstop`/`xstop`/`gtol` of
  `1e-7`. Optimistix always uses `cauchy`.
- **A2000.** Same as A with `max_steps=2000`, to give optimistix about as
  many accepted updates as optax gets in 500 steps. Only the first inner
  problem is a fair comparison: the outer loop still treats an inner solve
  as failed once it passes 500 steps, so the penalty does not grow on the
  later A2000 outers.

### Results

Final state after at most 8 outer iterations. "Solved" means `f_B` is under
its target and the L1 force is below the reference unit. No run met the
formal outer tolerance (`dx` and constraint violation below `1e-7`); the
optax runs from NESCOIL finish with a constraint violation around `1e-3`
or smaller and a force the notebook would accept.

| Solver | Start | Stop | Solved | First inner (steps / accepted / evals) | First non-finite | `f_B` / target | L1 / unit |
|---|---|---|---|---|---|---|---|
| A | NESCOIL | cauchy | no, NaN at outer 3 | 500 / 127 / 500 | rejected trial 1; accepted state NaN at outer 3 | 14300 | 8.9e5 |
| A | zeros | cauchy | no, stalls finite | 500 / 120 / 500 | rejected trial 253 | 5.85 | 1.08 |
| B20 | NESCOIL | cauchy | yes | 500 / 500 / 2068 | none | 0.979 | 0.896 |
| B20 | zeros | cauchy | no | 500 / 500 / 2224 | none | 0.569 | 0.999 |
| B80 | NESCOIL | cauchy | yes | 500 / 500 / 2058 | none | 0.904 | 0.942 |
| B80 | zeros | cauchy | no | 500 / 500 / 2224 | none | 0.614 | 0.989 |
| C | NESCOIL | cauchy | yes | 500 / 500 / 2434 | none | 0.904 | 0.942 |
| C | zeros | cauchy | no | 500 / 500 / 2308 | none | 0.587 | 1.00 |
| A2000 | NESCOIL | cauchy | no (see note) | 2000 / 582 / 2000 | rejected trial 1 | — | — |
| B20 | NESCOIL | legacy | yes | same first inner as cauchy | none | 0.984 | 0.897 |
| B80 | NESCOIL | legacy | yes | same first inner as cauchy | none | 0.967 | 0.902 |
| C | NESCOIL | legacy | yes | same first inner as cauchy | none | 0.943 | 0.942 |

B20 from NESCOIL is the closest to the 2026-08-12 notebook: `f_B = 3.264`
against a target of 3.335, L1 force `3.18e10` (0.90 times the reference
unit). Later inner solves there converge (482, 250, 2, 323, 2 steps) and
the penalty grows from 1 to 1.21. Solver A from NESCOIL never grows the
penalty, because every inner solve hits 500 steps. At outer 3 the accepted
state is NaN (`dx = 429`, constraint violation `1.7e7`), which is the same
failure as notebook cell 53.

### Answers

**The inner solve does not converge with optimistix, from the first outer
iteration.** Solver A always stops at the 500-step cap, with 120 to 160 of
those steps accepted. The gradient at the end of the first inner solve is
about 2400, against about 30 for optax zoom. From the NESCOIL start the
accepted state is still finite for two more outers, with the gradient
climbing to `1e4`–`1e5`, and is NaN at outer 3. From zeros it stays finite
for all 8 outers but the constraint violation stays near 5 and the gradient
near `1e5`. It is stalling, not converging.

**The line search is not slowly failing to find a step.** Optax uses a
median of 3 function evaluations per accepted step. The zoom cap of 20 is
touched (B20 records at most 21 evaluations in one step) but raising it to
80 does not change the outcome: B20 and B80 both solve the NESCOIL problem,
and their first-inner objectives are almost the same (augmented Lagrangian
1.554 vs 1.569 after 500 steps). What does consume the budget is
optimistix's accounting. Rejected backtracking trials count toward
`max_steps`, so 500 steps are about 130 real updates, roughly four trials
per accept. Optax's 500 steps are 500 accepted updates.

**Optimistix is worse than optax zoom on this problem, for two separable
reasons.** On the first inner problem, the augmented Lagrangian after equal
numbers of accepted steps:

| Accepted steps | optimistix | optax Armijo (C) | optax zoom (B20) |
|---|---|---|---|
| 127 | 38.3 | 36.7 | 23.4 |
| 500 | 3.25 (from the 2000-trial run, 582 accepts end at 3.00) | 3.06 | 1.55 |

Zoom is better per step than Armijo. Optimistix's Armijo and optax's Armijo
are close per accepted step, so the quasi-Newton update itself is not the
gap. The gap that breaks the solve is that optimistix spends three quarters
of `max_steps` on rejected trials, so the penalty never grows and a later
outer iteration diverges. Optax Armijo, given 500 accepted steps per outer
iteration, also solves the NESCOIL problem (L1 force 0.94 times the unit).
Wolfe vs Armijo matters for speed and final force; counting trials as
iterations is what makes the current solver fail.

**The first non-finite value is a rejected line-search trial at a huge
`x`, not a NaN gradient at a finite point.** From NESCOIL, optimistix
evaluations 1 through 7 of the first inner solve are NaN. The candidate at
evaluation 1 has `||x|| ≈ 1.7e4`, up from 34, so the objective overflows on
a wild step. None of those trials are accepted. Optax zoom and optax Armijo
record no non-finite evaluation. The accepted optimistix state stays finite
until outer 3, where the step itself jumps (`dx = 429`) and `f`, the
gradient, and `x` are NaN together.

**Starting from zeros does not fix it.** It keeps optimistix finite, but
after 8 outers `f_B` is still 5.8 times the target and the force is worse
than the reference. For the solvers that work, zeros is the worse start:
`f_B` ends up under the target, but the force is not reduced and the
current constraints stay violated by a few tenths. NESCOIL is the start
that optax turns into a feasible low-force coil.

The stopping rule is not the difference. On the first inner solve neither
`cauchy` nor `legacy` fires before 500 steps, so the traces match. Over 8
outers, legacy also solves the NESCOIL problem for B20, B80, and C, with
final forces within a few percent of the `cauchy` runs.

### Conclusion

The regression is the inner solver, not the outer multiplier update, the
initial condition, or the line-search cap of 80. `optimistix.LBFGS` with
the default Armijo search and `max_steps=500` does not converge this
augmented Lagrangian: rejected trials, including an initial run of
overflowing steps, use up the iteration budget, the penalty stays at 1, and
the NESCOIL run is NaN by outer iteration 3. The pre-May `optax.lbfgs` with
a zoom line search converges it from the NESCOIL start and reproduces the
2026-08-12 result (`f_B` at the target, L1 force about 0.9 times the
reference). Optax with Armijo backtracking also converges it, a bit less
tightly, which isolates the failure to optimistix's step accounting and its
unbounded backtracking rather than to Armijo conditions as such. A zero
initial condition avoids the NaN but does not produce a feasible low-force
coil.

## Full-resolution rerun

Same harness, native ARIES-CS resolution `L=24, M=12, N=8` (the equilibrium
the 2026-08-12 notebook used, before the tutorial reduced it to `L=M=N=6`).
Quadrature is still 33 by 33 and the current potential is still `mpol=ntor=8`,
so `x` is still length 144. Logs are in `scratch_conv/results_full/`.

The best-case solve that sets the targets also ended its last inner iteration
at the 500-step cap (`fin_f=1.322`). The resulting `f_B` target is 3.230 and
the reference L1 force is `3.20e10`. That target is tighter than the
notebook's 3.61, so the ratios below are not directly the August numbers.
NESCOIL starts at raw force 1.56 times the reference, with `f_B` already
under the target and the current constraints violated
(`g = [-0.62, 59.7, 19.6]`).

For the 2000-trial optimistix runs, the outer loop's cap test used 2000, so
the penalty was allowed to grow if an inner solve converged inside that
budget. It never did.

### Eight outer iterations

| Solver | Start | Stop | What happened | `f_B` / target | L1 / unit |
|---|---|---|---|---|---|
| optimistix, 500 | NESCOIL | cauchy | `inf` at outer 7; 133–170 accepts out of 500; penalty stays 1 | 59200 | 1.8e6 |
| optimistix, 500 | zeros | cauchy | `inf` at outer 1, then stuck | 52 | 114 |
| optax zoom, 20 | NESCOIL | cauchy | finite; penalty grows; constraint residual stalls at 0.106 | 1.106 | 0.913 |
| optax zoom, 20 | zeros | cauchy | finite, still far from feasible | 1.22 | 0.948 |
| optax zoom, 80 | NESCOIL and zeros | cauchy | identical to the 20-step runs | same | same |
| optax Armijo | NESCOIL | cauchy | finite, every inner hits 500, residual 0.55 | 1.55 | 0.950 |
| optax Armijo | zeros | cauchy | finite, residual 0.58 | 1.38 | 0.958 |
| optimistix, 2000 | NESCOIL | cauchy | finite over 4 outers, ~600 accepts, still hits the cap, penalty stays 1, gradient 8400 | 0.645 | 0.943 |
| optimistix, 2000 | zeros | cauchy | stalls, residual 12.5 | 13.5 | 1.12 |
| optax zoom, 20 | NESCOIL | legacy | still reducing the residual (0.055) at outer 7 | 1.055 | 0.915 |
| optax Armijo | NESCOIL | legacy | stops after 1 step with gradient ~2e4 | 0.515 | 0.999 |

B20 and B80 match to the printed digits, including the legacy runs. The
20-step zoom cap is not active on this problem.

### Longer optax runs, NESCOIL start

The production outer loop allows 10000 inner steps in total. Eight outers
is inside that, but zoom had not finished, so these three were continued.

| Solver | Outers | Final `c` | `f_B` / target | L1 / unit | Max constraint violation |
|---|---|---|---|---|---|
| optax zoom, Cauchy stop | 40 | 28 | 1.029 | 0.923 | 0.029 |
| optax zoom, legacy stop | 20 | 1.95 | 1.017 | 0.918 | 0.017 |
| optax Armijo, Cauchy stop | 20 | 2.36 | 1.000 | 0.927 | 0.019 |

None of these met the formal outer tolerance of `1e-7`. All three are in
the region the notebook treated as success: L1 force about 0.92 of the
reference (`2.94e10` to `2.97e10`), and `f_B` at or a few percent over a
target that is already tighter than the August notebook's. Zoom with the
legacy stop gets there in fewer outers. Zoom with the Cauchy stop stalls
whenever the step drops below the tolerance, then moves again only after
the penalty has grown. Armijo with the Cauchy stop also arrives, with a
bumpier residual along the way. Armijo with the legacy stop does not: a
tiny step is treated as convergence while the gradient is still ~1e4.

### What this says for the implementation

Keep the current outer augmented Lagrangian. Replace only the inner solver
with `optax.lbfgs` and a zoom line search of 20 steps, from the NESCOIL
start. That is the run that stays finite at this resolution and reaches a
low-force coil inside the existing iteration budget.

A larger optimistix budget does not substitute for that. With 2000 trials
the NESCOIL run stays finite and does reduce the force, and it still never
converges an inner solve, so the penalty never grows and the gradient stays
in the thousands. The zero start diverges for optimistix at this resolution.

The Cauchy stop can stay. It is the rule the current outer loop already
assumes, and with zoom it reaches the same coil as the old hand-tuned
stop, just with more outer iterations spent on penalty growth. The old
`fstop`/`xstop`/`gtol` rule is slightly faster here on zoom and wrongly
stops Armijo. It does not need to come back with the outer loop.
