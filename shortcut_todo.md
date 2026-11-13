# shortcut branch roadmap

Special convex solvers for narrow problem classes. Generic AugLag/IPM/SLSQP remain the default when classification returns `None`. Special detection overrides `smoothing` and `merge_constraints`.

## 1. Architecture and routing — done

- `solvers/special/` package: `classify`, `qcqp_one`, `qp_form`, `ipm_conic`, `qp_cpu`
- `classify_problem` detects `qcqp_one` then `qp` (static args; runs at trace time)
- Special path lives inside jitted `_quadcoil_pure` via static `enable_special`
- `kkt_mode='special'` for `qcqp_one`; `kkt_mode='special_conic'` for multi-constraint QP
- Focused classify + routing tests in `tests/test_special_classify.py`

## 2. Support for `qcqp_one` — done

- No separate canonicalize step: `solve_qcqp_one(f, g, x_init, ...)` materializes \(H,b,c\) once, rescales `g`, simultaneous-diagonalizes via Cholesky of \(H_f + s H_g\), closed-form \(y(\lambda)\), safeguarded Newton + bisection
- Fully jittable: integer `status_code` (0 unconstrained / 1 active / 2 infeasible), fixed-shape `fin_z=(1,)`, no Python `bool()` branches
- `quadcoil()` maps concrete `status_code` → `status` string via per-class tables (`SPECIAL_STATUS`)
- End-to-end tests in `tests/test_qcqp_one.py` (cases, gradients vs generic unconstrained, no-recompile on `constraint_value`)

## 3. Support for QP (no `linear_slack`) — done

- `classify_problem` returns `'qp'` for quadratic/linear objectives + linear `<=`/`>=` constraints
- `qp_form.canonicalize_qp` materialises `(P, q, G, h[, A, b])`
- `ipm_conic.solve_conic_ipm`: speed-oriented Mehrotra for orthant (QP) and SOCP branches
- **Default QP backend is `special_backend='ipm_conic'`** (alias `'ipm'`). Clarabel is opt-in via `special_backend='clarabel'`
- `qp_cpu.solve_conic_clarabel`: Clarabel via `jax.pure_callback` (optional FFI scaffold in `ffi/clarabel/`)
- `kkt_mode='special_conic'` for the QP adjoint (no scalar-`g` reshape)
- Generic `ipm.py` fixed (F1–F7): full Mehrotra, no m×m solves, infeasible starts, Cholesky inertia correction
- Tests: `test_ipm`, `test_ipm_conic`, `test_qp_cpu`

## 4. Support for SOCP (linear + quadratic `<=`, no slack) — done

- `classify_problem` returns `'socp'` for quadratic/linear objectives with
  linear `<=`/`>=` plus at least one convex quadratic `<=` constraint
- `canonicalize_socp` rewrites each scalar quadratic as a rotated SOC of
  size `n+2`; dual map `λ = z₀ + z_last` for the adjoint
- Routed through `ipm_conic` SOCP branch (`kkt_mode='special_conic'`)
- Hardened SOCP IPM: Lorentz step-to-boundary quadratic, corrector cent
  sign, Ruiz SOC max-equalization
- Tests: `test_socp`, extended `test_ipm_conic` / `test_special_classify`
- Benchmark: `scratch_qp/bench_regcoil_socp.py` (filament REGCOIL)

Still open (slack quantities):
- Extend classification for `linear_slack` / `quadratic_slack`
- Introduce slack variables and affine inequalities in the special problem representation
- Cover L∞ / L1 objectives and `<=` constraints built from affine fields

## 5. Performance testing, new tests, and green suite — mostly done

Done:
- Special vs AugLag (`enable_special=False`) on CPU and GPU for `qcqp_one`
- Topology QP CPU/GPU summary (below); scripts/JSON under `scratch_qp/`
- REGCOIL SOCP CPU/GPU summary (below); `scratch_qp/bench_regcoil_socp.py`
- Unit tests: `test_qcqp_one`, `test_special_classify`, `test_ipm`, `test_ipm_conic`, `test_qp_cpu`, `test_socp` green
- Fixed `abs_lse` scale (was sharpened by `log(2N)`); `approx` L1 / L∞ objective values shift by up to `O(eps)`

### Topology QP benchmark

From `docs/examples/filament.ipynb` §2.2 (li383 modular-coil topology constraint):

| Item | Value |
|------|--------|
| Plasma | li383 (`docs/examples/data/surfaces.json`) |
| Net poloidal current \(G\) | `1.1884578094260072e7` A |
| Objective | `f_B` |
| Constraint | `K_theta >= 0` pointwise (`constraint_type` from `sign(G)`) |
| `constraint_value` / `constraint_unit` | `[0.]` / `(None,)` |
| Fourier resolution | `mpol = ntor = 6` (stellarator symmetry ⇒ \(n = 84\) dofs) |
| Quadrature | default → \(m = 32 \times 34 = 1088\) inequality rows |
| Slack variables | none (pure QP when `smoothing=None`) |
| Initialization | zeros (`phi_init_with_nescoil=False`) |
| Metrics | `f_B`, pointwise `min K_theta`, warm solve time |

**Smoothing.** Special path always forces `smoothing=None` (exact pointwise QP). Generic AugLag / IPM were also run with `smoothing='approx'`, `lse_epsilon=1e-1` (single LSE constraint — different math, same engineering goal).

**Hardware.** Intel i7-14700 (CPU), NVIDIA GeForce RTX 4060 (GPU). Warm time = median of 5 runs after one cold compile (`value_only=True`).

| Solver | f_B (CPU) | min K_θ (CPU) | warm CPU [s] | f_B (GPU) | min K_θ (GPU) | warm GPU [s] | GPU/CPU |
|--------|----------:|--------------:|-------------:|----------:|--------------:|-------------:|--------:|
| Clarabel (`special_backend='clarabel'`) | 4.891611e-02 | +3.34e-02 | 1.1242 | — | — | — | CPU-only |
| `ipm_conic` (default QP) | 4.891611e-02 | +8.74e-02 | 0.7810 | 4.891611e-02 | +8.74e-02 | 0.2766 | 2.82× |
| Generic IPM, `smoothing=None` | 4.891612e-02 | +8.29e-01 | 1.7999 | 4.891612e-02 | +8.29e-01 | 0.2936 | 6.13× |
| AugLag, `smoothing=None` | 6.512024e-02 | −2.10e-03 | 7.5431 | 5.773484e-02 | −2.71e+01 | 2.0386 | 3.70× |
| Generic IPM, `smoothing=approx` | 5.263522e-02 | +5.89e+05 | 1.1851 | 5.004735e-02 | −5.35e+05 | 0.6475 | 1.83× |
| AugLag, `smoothing=approx` | 5.264011e-02 | +5.92e+05 | 12.7389 | 5.264059e-02 | +5.89e+05 | 2.0776 | 6.13× |

Takeaways:
- Exact-QP trio (Clarabel, `ipm_conic`, generic IPM/`None`) agree on `f_B ≈ 4.891611e-02` and are pointwise feasible
- `approx` designs land near `f_B ≈ 5.26e-02` (~8% higher); IPM/`approx` on GPU was not pointwise-feasible here
- AugLag/`None` did not finish cleanly within the iteration budget (incomplete solves)
- No-recompile: `_quadcoil_pure._cache_size()` unchanged when `constraint_value` changes (`ipm_conic` and Clarabel)

Reproduce:
```bash
BENCH_PLATFORM=cpu JAX_PLATFORMS=cpu python scratch_qp/bench_cpu_gpu_summary.py
BENCH_PLATFORM=cuda JAX_PLATFORMS=cuda python scratch_qp/bench_cpu_gpu_summary.py
JAX_PLATFORMS=cpu python scratch_qp/bench_topology.py
```

### REGCOIL SOCP benchmark

From `docs/examples/filament.ipynb` constrained REGCOIL (minimize `f_K` s.t. `K_theta` topology + `f_B <= f_B_target`). Script: `scratch_qp/bench_regcoil_socp.py`; merged table also in `scratch_qp/summary_regcoil.md`.

| Item | Value |
|------|--------|
| Plasma | li383 (`docs/examples/data/surfaces.json`) |
| Net poloidal current \(G\) | `1.1884578094260072e7` A |
| Objective | `f_K` (unit = topology `f_K_ref`) |
| Constraints | `K_theta >= 0` pointwise; `f_B <= f_B_target` |
| Targets (from topology approx) | `f_B_topo=5.262037e-02`, `f_B_target=1.052407e-01`, `f_K_ref=9.263001e+13` |
| Fourier resolution | `mpol = ntor = 6` |
| Special class | `'socp'` (`ipm_conic`); generic rows use `enable_special=False` |
| Init | cold = zeros; warm = NESCOIL; `ipm_conic` init n/a (CVXOPT-style start) |
| Metrics | `f_K`, `f_B`, `f_B/target`, pointwise `min K_theta`, cold + warm time |

**Smoothing.** Special path forces `smoothing=None`. Generic AugLag / IPM also run with `smoothing='approx'`, `lse_epsilon=1e-1` (different math, same engineering comparison).

**Hardware.** Intel i7-14700 (CPU), NVIDIA GeForce RTX 4060 (GPU). Warm time = median of 3 runs after one cold compile (`value_only=True`).

| Device | Algorithm | Smoothing | Init | Status | niter | f_K | f_B | f_B/target | min K_θ | cold [s] | warm [s] |
|--------|-----------|-----------|------|--------|------:|----:|----:|----------:|--------:|---------:|---------:|
| cpu | ipm_conic | special | n/a | solved | 20 | 6.108772e+13 | 1.052407e-01 | 1.0000 | 3.0859e-02 | 3.782 | 1.216 |
| cpu | ipm | None | cold | solved | 9 | 6.108772e+13 | 1.052407e-01 | 1.0000 | 1.3653e-02 | 3.706 | 1.503 |
| cpu | ipm | None | warm | solved | 16 | 6.108772e+13 | 1.052407e-01 | 1.0000 | 1.5579e-02 | 5.688 | 1.347 |
| cpu | auglag | None | cold | not_converged | 697 | 6.109420e+13 | 1.052405e-01 | 1.0000 | 1.9813e-01 | 12.613 | 8.268 |
| cpu | auglag | None | warm | not_converged | 1127 | 6.108801e+13 | 1.052419e-01 | 1.0000 | −6.0639e+00 | 12.973 | 8.109 |
| cpu | ipm | approx | cold | solved | 7 | 6.109011e+13 | 1.052407e-01 | 1.0000 | 1.6787e+05 | 4.027 | 1.188 |
| cpu | ipm | approx | warm | max_iterations | 100 | 3.032940e+19 | 9.313067e+01 | 884.9299 | −3.0666e+10 | 4.868 | 1.659 |
| cpu | auglag | approx | cold | not_converged | 1948 | 6.108772e+13 | 1.052530e-01 | 1.0001 | 1.6919e+05 | 16.065 | 10.734 |
| cpu | auglag | approx | warm | not_converged | 1410 | 6.107900e+13 | 1.052946e-01 | 1.0005 | 1.6801e+05 | 13.243 | 8.639 |
| cuda | ipm_conic | special | n/a | solved | 20 | 6.108772e+13 | 1.052407e-01 | 1.0000 | 3.0859e-02 | 8.635 | 0.275 |
| cuda | ipm | None | cold | solved | 9 | 6.108772e+13 | 1.052407e-01 | 1.0000 | 1.3653e-02 | 7.576 | 0.301 |
| cuda | ipm | None | warm | solved | 16 | 6.108772e+13 | 1.052407e-01 | 1.0000 | 1.5579e-02 | 7.834 | 0.341 |
| cuda | auglag | None | cold | not_converged | 845 | 6.108786e+13 | 1.052443e-01 | 1.0000 | −1.1433e+01 | 10.670 | 1.489 |
| cuda | auglag | None | warm | not_converged | 1149 | 6.108854e+13 | 1.052406e-01 | 1.0000 | −3.9348e-02 | 11.564 | 2.201 |
| cuda | ipm | approx | cold | solved | 7 | 6.109011e+13 | 1.052407e-01 | 1.0000 | 1.6787e+05 | 7.398 | 0.276 |
| cuda | ipm | approx | warm | max_iterations | 100 | 3.136819e+19 | 9.377723e+01 | 891.0735 | −3.0941e+10 | 8.323 | 0.891 |
| cuda | auglag | approx | cold | not_converged | 2026 | 6.109018e+13 | 1.052408e-01 | 1.0000 | 1.6947e+05 | 12.048 | 2.741 |
| cuda | auglag | approx | warm | not_converged | 1329 | 6.108909e+13 | 1.052468e-01 | 1.0001 | 1.6940e+05 | 11.839 | 1.999 |

Takeaways:
- Exact trio (`ipm_conic`, IPM/`None` cold & warm) agree on `f_K ≈ 6.108772e+13`, `f_B/target = 1`, and are pointwise feasible
- GPU warm speedup for `ipm_conic`: 1.216 s → 0.275 s (~4.4×)
- IPM/`approx`/cold matches exact `f_B`/`f_K` closely and is fast; IPM/`approx`/warm diverges on both devices
- AugLag often reaches similar `f_B` but not fully converged / not always pointwise feasible within the budget

Reproduce:
```bash
BENCH_PLATFORM=cpu python scratch_qp/bench_regcoil_socp.py
BENCH_PLATFORM=cuda python scratch_qp/bench_regcoil_socp.py   # use a CUDA jax env
python scratch_qp/merge_regcoil_summary.py
```

Still open:
- Broader suite green (full `unittest discover`)
- Complete Clarabel.cpp FFI handler body (scaffold in `ffi/clarabel/` is a stub)
- Slack quantities (`linear_slack` / `quadratic_slack`) for the special path
