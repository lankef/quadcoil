"""
Mehrotra predictor-corrector primal-dual interior-point method (PDIP)
for small-scale NLP in JAX/Lineax.

Problem:
    min  f_obj(x)
    s.t. g_ineq(x) <= 0   (m constraints)

Convention: slacks s > 0 satisfy g(x) + s = 0 at feasibility.
Dual variables z > 0 are the Lagrange multipliers.

Provides the QUADCOIL solver interface:
    solve_unconstrained_ipm
    solve_constrained_ipm
    stationarity_ipm
    adjoint_ipm

Adapted from pdip_solver.py (reference implementation).

Potential further optimisations
-------------------------------
#4  AAt_reg in SOC backtracking is recomputed every IPM iteration.
    For larger m, cache it or use the W_c factorization via the
    Schur-complement identity to compute the SOC projection.
#6  stationarity_ipm materialises the full (n+m)x(n+m) KKT Jacobian.
    For large m this is wasteful; a Schur-complement adjoint that
    only forms the n×n reduced system would save memory and FLOPs.
"""

import jax
import jax.numpy as jnp
from jax import grad, jacrev, hessian, jvp, debug
from jax.lax import while_loop
import lineax as lx
from jax import config as config_jax
config_jax.update('jax_enable_x64', True)

from .kkt_adjoint import stationarity_kkt, adjoint_kkt




# ── Utility helpers ──────────────────────────────────────────────────────

def fraction_to_boundary(x, dx, tau=0.995):
    """Largest alpha in (0,1] keeping x + alpha*dx > 0."""
    ratios = jnp.where(dx < 0.0, -x / dx, jnp.inf)
    return jnp.minimum(1.0, tau * jnp.min(ratios))


def kkt_residual(grad_f, A, g, s, z):
    r"""
    Scaled KKT residual (stationarity + primal feasibility).

    Returns (max(sd, pf), r_stat, r_prim) where
    sd = \|grad_f + A^T z\| / max(1, \|grad_f\|),
    pf = \|g + s\|          / max(1, \|g\|).
    """
    r_stat = grad_f + A.T @ z
    r_prim = g + s
    sd = jnp.linalg.norm(r_stat) / jnp.maximum(1.0, jnp.linalg.norm(grad_f))
    pf = jnp.linalg.norm(r_prim) / jnp.maximum(1.0, jnp.linalg.norm(g))
    return jnp.maximum(sd, pf), r_stat, r_prim



# ── Solvers ──────────────────────────────────────────────────────────────

DEFAULT_IPM_OPTIONS = {
    'tol_kkt': 1e-6,
    'tau': 0.995,
    'delta_init': 1e-6,
    'delta_min': 1e-10,
    'delta_max': 1e-2,
}


def solve_unconstrained_ipm(
    init_params, 
    fun,
    convex=False, 
    maxiter: int = 100,
    solver_options=None, 
    verbose=0
):
    r'''
    Unconstrained optimization via a single Newton step (direct solve).

    Thin wrapper around :func:`solve_constrained_ipm` with no constraints.

    Parameters
    ----------
    init_params : ndarray, shape (n,)
    fun : Callable, x -> scalar
    maxiter : int, optional, default=100
        (Static) Maximum IPM iterations.
    solver_options : dict, optional
    verbose : int

    Returns
    -------
    dict
        ``'fin_f'``, ``'fin_x'``, ``'niter'``, ``'fin_kkt_res'``,
        ``'converged'``.
    '''
    result = solve_constrained_ipm(
        x_init=init_params,
        f_obj=fun,
        convex=convex,
        maxiter=maxiter,
        solver_options=solver_options,
        verbose=verbose,
    )
    return {
        'fin_f': result['fin_f'],
        'fin_x': result['fin_x'],
        'niter': result['niter'],
        'fin_kkt_res': result['fin_kkt_res'],
        'converged': result['converged'],
    }


def solve_constrained_ipm(
    x_init,
    f_obj,
    h_eq=lambda x: jnp.zeros(0),
    g_ineq=lambda x: jnp.zeros(0),
    convex=False,
    maxiter: int = 100,
    solver_options=None,
    verbose=0,
):
    r'''
    Primal-dual interior-point method (Mehrotra predictor-corrector) for:

    .. math::

        \min_x f(x) \quad \text{s.t.}\; g(x) \le 0

    Parameters
    ----------
    x_init : ndarray, shape (n,)
    f_obj : Callable, x -> scalar
    h_eq : Callable, optional
        Equality constraints — **not yet supported**; must return ``jnp.zeros(0)``.
    g_ineq : Callable, optional
        Inequality constraints, x -> (m,), g(x) <= 0.
    solver_options : dict, optional
        IPM parameters.  Recognised keys:

        - ``'tol_kkt'`` (``1e-6``) — KKT residual convergence tolerance.
        - ``'tau'`` (``0.995``) — fraction-to-boundary parameter.
        - ``'delta_init'`` (``1e-6``) — initial primal regularisation.
        - ``'delta_min'`` (``1e-10``) — minimum regularisation.
        - ``'delta_max'`` (``1e-2``) — maximum regularisation.
    verbose : int, optional, default=0

    Returns
    -------
    status : dict
        - ``'fin_x'`` — primal solution, shape ``(n,)``.
        - ``'fin_f'`` — objective at solution.
        - ``'fin_g'`` — constraint values at solution, shape ``(m,)``.
        - ``'fin_z'`` — dual variables (Lagrange multipliers), shape ``(m,)``.
        - ``'fin_s'`` — slack variables, shape ``(m,)``.
        - ``'niter'`` — iterations taken.
        - ``'fin_kkt_res'`` — final KKT residual.
        - ``'converged'`` — bool.
    '''
    opts = {**DEFAULT_IPM_OPTIONS}
    if solver_options is not None:
        opts.update(solver_options)

    tol_kkt = opts['tol_kkt']
    tau = opts['tau']
    delta_init = opts['delta_init']
    delta_min = opts['delta_min']
    delta_max = opts['delta_max']

    # ── Core Mehrotra predictor-corrector PDIP loop ─────────────────────
    x_init = jnp.asarray(x_init, dtype=jnp.float64)
    n = x_init.shape[0]

    g0 = g_ineq(x_init)
    m = g0.shape[0]

    # ── m = 0: unconstrained — single Newton step (direct solve) ────────
    if m == 0:
        grad_f0 = grad(f_obj)(x_init)
        H = hessian(f_obj)(x_init) + jnp.float64(delta_init) * jnp.eye(n)
        tags = (lx.symmetric_tag, lx.positive_semidefinite_tag) if convex \
            else (lx.symmetric_tag,)
        H_op = lx.MatrixLinearOperator(H, tags=tags)
        sol = lx.linear_solve(
            H_op, -grad_f0,
            solver=lx.AutoLinearSolver(well_posed=convex),
            throw=False,
        )
        x_fin = x_init + sol.value
        kkt0 = jnp.linalg.norm(grad(f_obj)(x_fin))
        empty = jnp.zeros(0, dtype=jnp.float64)
        return {
            'fin_x': x_fin,
            'fin_f': f_obj(x_fin),
            'fin_g': empty,
            'fin_z': empty,
            'fin_s': empty,
            'niter': jnp.int32(1),
            'fin_kkt_res': kkt0,
            'converged': kkt0 < tol_kkt,
        }

    # ── m > 0: full PDIP with barrier ───────────────────────────────────
    s_init = jnp.maximum(-g0, jnp.float64(1e-4))

    A0 = jacrev(g_ineq)(x_init)
    grad_f0 = grad(f_obj)(x_init)
    eye = jnp.eye(m, dtype=jnp.float64)
    AAt = A0 @ A0.T + 1e-8 * eye
    rhs = A0 @ (-grad_f0)
    z_ls = jnp.linalg.solve(AAt, rhs)
    z_init = jnp.maximum(z_ls, 1.0)

    mu_b0 = jnp.dot(s_init, z_init) / m

    if verbose > 0:
        jax.debug.print(
            "IPM SOLVER: n={n}, m={m}, tol_kkt={tol}, maxiter={mi}",
            n=n, m=m, tol=tol_kkt, mi=maxiter,
        )

    # Cache grad_f, A, g_val in the state to avoid recomputing at
    # the start of the next iteration (optimisation #1).
    init_state = {
        "x":          x_init,
        "s":          s_init,
        "z":          z_init,
        "mu_barrier": mu_b0,
        "delta":      jnp.float64(delta_init),
        "n_iter":     jnp.int32(0),
        "kkt_res":    jnp.float64(jnp.inf),
        "converged":  jnp.bool_(False),
        "grad_f":     grad_f0,
        "A":          A0,
        "g_val":      g0,
    }

    def not_converged(state):
        return (
            (state["n_iter"] == 0)
            | ((state["n_iter"] < maxiter) & ~state["converged"])
        )

    def ipm_step(state):
        x      = state["x"]
        s      = state["s"]
        z      = state["z"]
        mu_b   = state["mu_barrier"]
        delta  = state["delta"]
        # Reuse cached derivatives from previous iteration (#1)
        g_val  = state["g_val"]
        grad_f = state["grad_f"]
        A      = state["A"]

        def lagrangian(xx):
            return f_obj(xx) + jnp.dot(z, g_ineq(xx))

        r_stat = grad_f + A.T @ z
        r_prim = g_val + s
        zs     = z / s

        # ── Condensed n×n system: W_c Δx = rhs (explicit dense) ─────────
        H_lag = hessian(lagrangian)(x)
        W_c_mat = H_lag + delta * jnp.eye(n) + A.T @ (zs[:, None] * A)

        # Factorize W_c once, solve predictor + corrector with
        # back-substitution only (optimisation #3).
        if convex:
            W_factor = jax.scipy.linalg.cho_factor(W_c_mat)
            _solve = lambda rhs_v: jax.scipy.linalg.cho_solve(W_factor, rhs_v)
        else:
            W_lu, W_piv = jax.scipy.linalg.lu_factor(W_c_mat)
            _solve = lambda rhs_v: jax.scipy.linalg.lu_solve(
                (W_lu, W_piv), rhs_v
            )

        # ── Predictor (affine scaling) ───────────────────────────────────
        rhs_x_aff = -(grad_f + A.T @ (zs * r_prim))
        dx_aff = _solve(rhs_x_aff)

        ds_aff = -(r_prim + A @ dx_aff)
        dz_aff = -z - zs * ds_aff

        # ── Mehrotra centering parameter ─────────────────────────────────
        alpha_p_aff = fraction_to_boundary(s, ds_aff, tau)
        alpha_d_aff = fraction_to_boundary(z, dz_aff, tau)
        alpha_aff   = jnp.minimum(alpha_p_aff, alpha_d_aff)

        s_trial = s + alpha_aff * ds_aff
        z_trial = z + alpha_aff * dz_aff
        mu_aff  = jnp.dot(s_trial, z_trial) / m
        sigma   = jnp.clip((mu_aff / (mu_b + 1e-30)) ** 3, 0.0, 1.0)
        mu_rhs  = sigma * mu_b * jnp.ones(m)

        # ── Corrector (reuses same factorization) ────────────────────────
        rhs_x_cor = -(grad_f + A.T @ (zs * r_prim) + A.T @ (mu_rhs / s))
        dx = _solve(rhs_x_cor)

        ds = -(r_prim + A @ dx)
        dz = mu_rhs / s - z - zs * ds

        # Check for NaN in search direction (factorization failure proxy)
        factor_failed = ~jnp.isfinite(dx).all()

        # ── Step lengths ────────────────────────────────────────────────
        alpha_p_max = fraction_to_boundary(s, ds, tau)
        alpha_d = fraction_to_boundary(z, dz, tau)

        # ── Primal step with SOC-augmented backtracking ───────────────
        # Uses while_loop with early exit for efficiency (#2).
        AAt_reg = A @ A.T + jnp.float64(1e-12) * jnp.eye(m)

        def _soc_continue(bt_state):
            _, _, accepted, bt_iter = bt_state
            return (~accepted) & (bt_iter < 20)

        def _soc_body(bt_state):
            alpha, best_x, accepted, bt_iter = bt_state
            x_trial = x + alpha * dx
            g_trial = g_ineq(x_trial)
            g_corr = jnp.maximum(g_trial + jnp.float64(1e-6), jnp.float64(0.0))
            dx_soc = -A.T @ jnp.linalg.solve(AAt_reg, g_corr)
            x_soc = x_trial + dx_soc
            g_soc = g_ineq(x_soc)
            feasible = jnp.all(g_soc < 0) & jnp.isfinite(x_soc).all()
            best_x = jnp.where(feasible, x_soc, best_x)
            accepted = accepted | feasible
            return (alpha * jnp.float64(0.5), best_x, accepted, bt_iter + 1)

        _, x_new, found, _ = while_loop(
            _soc_continue, _soc_body,
            (alpha_p_max, x, jnp.bool_(False), jnp.int32(0)),
        )
        x_new = jnp.where(found, x_new, x)

        z_new = jnp.maximum(z + alpha_d * dz, jnp.float64(1e-300))

        # ── Slack reset ────────────────────────────────────────────────
        g_at_new = g_ineq(x_new)
        s_new = jnp.maximum(-g_at_new, jnp.float64(1e-8))

        mu_b_new = jnp.dot(s_new, z_new) / m

        # ── Regularisation schedule ──────────────────────────────────────
        delta_new = jnp.where(
            factor_failed,
            jnp.minimum(delta * 10.0, jnp.float64(delta_max)),
            jnp.maximum(delta / 3.0,  jnp.float64(delta_min)),
        )

        # ── NaN guard ────────────────────────────────────────────────────
        step_ok = (
            jnp.isfinite(x_new).all()
            & jnp.isfinite(s_new).all()
            & jnp.isfinite(z_new).all()
        )
        x_new    = jnp.where(step_ok, x_new,    x)
        s_new    = jnp.where(step_ok, s_new,    s)
        z_new    = jnp.where(step_ok, z_new,    z)
        mu_b_new = jnp.where(step_ok, mu_b_new, mu_b)

        # ── KKT check + cache for next iteration (#1) ──────────────────
        grad_f_new = grad(f_obj)(x_new)
        A_new      = jacrev(g_ineq)(x_new)
        g_new      = g_ineq(x_new)
        res_sf, r_stat_new, r_prim_new = kkt_residual(grad_f_new, A_new, g_new, s_new, z_new)
        kkt_res_new  = jnp.maximum(res_sf, mu_b_new)
        converged_new = kkt_res_new < tol_kkt

        if verbose > 1:
            jax.debug.print(
                "IPM iter {it}: kkt={kkt:.3e}  mu_b={mu:.3e}  "
                "ap={ap:.3f}  ad={ad:.3f}  delta={d:.2e}  sigma={sg:.3f}",
                it=state["n_iter"] + 1,
                kkt=kkt_res_new, mu=mu_b_new,
                ap=alpha_p_max, ad=alpha_d, d=delta_new, sg=sigma,
            )
        elif verbose > 0:
            jax.debug.print(
                "IPM iter {it}: kkt={kkt:.3e}  mu_b={mu:.3e}",
                it=state["n_iter"] + 1, kkt=kkt_res_new, mu=mu_b_new,
            )

        return {
            "x":          x_new,
            "s":          s_new,
            "z":          z_new,
            "mu_barrier": mu_b_new,
            "delta":      delta_new,
            "n_iter":     state["n_iter"] + 1,
            "kkt_res":    kkt_res_new,
            "converged":  converged_new,
            "grad_f":     grad_f_new,
            "A":          A_new,
            "g_val":      g_new,
        }

    final = while_loop(not_converged, ipm_step, init_state)

    if verbose > 0:
        jax.debug.print(
            "IPM DONE: iters={it}  kkt_res={kkt:.3e}  converged={cv}",
            it=final["n_iter"], kkt=final["kkt_res"], cv=final["converged"],
        )

    x_opt = final["x"]
    g_final = final["g_val"]
    s_final = final["s"]
    z_final = final["z"]

    return {
        'fin_x':       x_opt,
        'fin_f':       f_obj(x_opt),
        'fin_g':       g_final,
        'fin_z':       z_final,
        'fin_s':       s_final,
        'niter':       final["n_iter"],
        'fin_kkt_res': final["kkt_res"],
        'converged':   final["converged"],
    }


# ── Stationarity (KKT Jacobian setup) ───────────────────────────────────

def stationarity_ipm(
    constrained,
    convex,
    solve_results,
    y_flat,
    f_g_ineq_h_eq_from_y,
    unravel_y,
    unravel_unscale_x,
    solver_options,
    verbose,
):
    r'''
    Build the KKT stationarity condition for implicit differentiation
    through the IPM solution.  Delegates to the shared
    :func:`kkt_adjoint.stationarity_kkt`.

    Parameters
    ----------
    constrained : bool
    convex : bool
    solve_results : dict
        Output of :func:`solve_constrained_ipm` or
        :func:`solve_unconstrained_ipm`.
    y_flat : ndarray
    f_g_ineq_h_eq_from_y : Callable
    unravel_y : Callable
    unravel_unscale_x : Callable
    solver_options : dict
    verbose : int

    Returns
    -------
    stationarity_data : dict
        Opaque state consumed by :func:`adjoint_ipm`.
    '''
    x_opt = solve_results['fin_x']
    z_opt = solve_results.get('fin_z', jnp.zeros(0, dtype=x_opt.dtype))

    return stationarity_kkt(
        constrained=constrained,
        convex=convex,
        x_opt=x_opt,
        z_opt=z_opt,
        y_flat=y_flat,
        f_g_ineq_h_eq_from_y=f_g_ineq_h_eq_from_y,
        unravel_y=unravel_y,
        unravel_unscale_x=unravel_unscale_x,
        verbose=verbose,
    )


# ── Adjoint (per-metric derivative) ─────────────────────────────────────

def adjoint_ipm(
    f_metric,
    stationarity_data,
    y_flat,
    implicit_linear_solver,
    verbose,
):
    r'''
    Compute the total derivative of a single metric w.r.t. all problem
    parameters via the KKT adjoint.  Delegates to the shared
    :func:`kkt_adjoint.adjoint_kkt`.

    Parameters
    ----------
    f_metric : Callable
        ``(x_flat, y_flat) -> scalar``.
    stationarity_data : dict
        Output of :func:`stationarity_ipm`.
    y_flat : ndarray
    implicit_linear_solver : lineax.AbstractLinearSolver
    verbose : int

    Returns
    -------
    metric_value : scalar
    dfdy_arr : ndarray, shape ``(ny,)``
    debug_info : dict
    '''
    return adjoint_kkt(
        f_metric=f_metric,
        stationarity_data=stationarity_data,
        y_flat=y_flat,
        implicit_linear_solver=implicit_linear_solver,
        verbose=verbose,
    )
