"""
SLSQP solver backend for QUADCOIL using slsqp-jax (Optimistix interface).

Provides the standard four-function QUADCOIL solver interface:
    solve_unconstrained_slsqp
    solve_constrained_slsqp
    stationarity_slsqp
    adjoint_slsqp

Notation/convention differences (quadcoil vs slsqp-jax):
──────────────────────────────────────────────────────────
  Aspect              quadcoil                slsqp-jax
  ─────────────────── ─────────────────────── ───────────────────────────
  Inequality dir.     g(x) <= 0               c_ineq(x) >= 0
  Equality            h(x) = 0                c_eq(x) = 0
  Objective sig.      f(x) -> scalar          f(x, args) -> (scalar, aux)
  Multipliers         z >= 0 for g <= 0       lambda for c >= 0
  Constraint Jac.     A = jac(g), (m,n)       J = jac(c), (m,n);  A = -J
  ─────────────────── ─────────────────────── ───────────────────────────

Conversion:  pass  ineq_constraint_fn = lambda x, args: -g_ineq(x)
             so that  c_ineq(x) = -g(x) >= 0  iff  g(x) <= 0.

Multiplier mapping:  slsqp-jax's multiplier for c_ineq >= 0 corresponds
             directly to quadcoil's z for g <= 0 (the sign flip in the
             constraint and the sign flip in the dual cancel).
"""

import jax
import jax.numpy as jnp
from jax import grad, jacrev, debug
import lineax as lx
import optimistix as optx
from slsqp_jax import SLSQP, SLSQPConfig, ToleranceConfig
from jax import config as config_jax
config_jax.update('jax_enable_x64', True)

from .kkt_adjoint import stationarity_kkt, adjoint_kkt


# ── Default options ──────────────────────────────────────────────────────

DEFAULT_SLSQP_OPTIONS = {
    'atol': 1e-7,
    'rtol': 1e-7,
}


# ── Multiplier recovery from KKT conditions ─────────────────────────────

def _recover_multipliers(f_obj, g_ineq, x_opt):
    r"""
    Recover Lagrange multipliers at x_opt from KKT stationarity:

        grad_f + A^T z = 0   =>   z = -(A A^T)^{-1} A grad_f

    where A = jac(g)(x_opt).  Uses pseudoinverse for robustness.
    Only active constraints (g ~= 0) get nonzero multipliers;
    inactive ones are clipped to zero.
    """
    grad_f = grad(f_obj)(x_opt)
    A = jacrev(g_ineq)(x_opt)
    m = A.shape[0]
    if m == 0:
        return jnp.zeros(0, dtype=x_opt.dtype)
    # z = -pinv(A^T) @ grad_f  =  -(A A^T)^{-1} A @ grad_f
    AAt = A @ A.T
    AAt_reg = AAt + 1e-10 * jnp.eye(m)
    rhs = -A @ grad_f
    z = jnp.linalg.solve(AAt_reg, rhs)
    # Clip to non-negative (inactive constraints have z=0)
    z = jnp.maximum(z, 0.0)
    return z


# ── Unconstrained solve ──────────────────────────────────────────────────

def solve_unconstrained_slsqp(
    init_params,
    fun,
    convex=False,
    maxiter: int = 200,
    solver_options=None,
    verbose=0,
    lbfgs_memory=10,
):
    r'''
    Unconstrained minimization via SLSQP (no constraints).

    Parameters
    ----------
    init_params : ndarray, shape (n,)
    fun : Callable, x -> scalar
    convex : bool
        Unused (for interface compatibility).
    maxiter : int, optional, default=200
        (Static) Maximum SQP iterations.
    solver_options : dict, optional
    lbfgs_memory : int, optional, default=10
        (Static) L-BFGS history length.
    verbose : int

    Returns
    -------
    dict
        ``'fin_f'``, ``'fin_x'``, ``'niter'``, ``'converged'``.
    '''
    result = solve_constrained_slsqp(
        x_init=init_params,
        f_obj=fun,
        convex=convex,
        maxiter=maxiter,
        solver_options=solver_options,
        verbose=verbose,
        lbfgs_memory=lbfgs_memory,
    )
    return {
        'fin_f': result['fin_f'],
        'fin_x': result['fin_x'],
        'niter': result['niter'],
        'converged': result['converged'],
    }


# ── Constrained solve ────────────────────────────────────────────────────

def solve_constrained_slsqp(
    x_init,
    f_obj,
    h_eq=lambda x: jnp.zeros(0),
    g_ineq=lambda x: jnp.zeros(0),
    convex=False,
    maxiter: int = 200,
    solver_options=None,
    verbose=0,
    lbfgs_memory=10,
):
    r'''
    Constrained optimization via slsqp-jax:

    .. math::

        \min_x f(x) \quad \text{s.t.}\; g(x) \le 0,\; h(x) = 0

    Parameters
    ----------
    x_init : ndarray, shape (n,)
    f_obj : Callable, x -> scalar
    h_eq : Callable, optional
        Equality constraints, x -> (p,), h(x) = 0.
    g_ineq : Callable, optional
        Inequality constraints, x -> (m,), g(x) <= 0.
    convex : bool, optional
        Unused (for interface compatibility).
    solver_options : dict, optional
        SLSQP parameters.  Recognised keys:

        - ``'atol'`` (1e-7) — absolute KKT tolerance.
        - ``'rtol'`` (1e-7) — relative KKT tolerance.
    lbfgs_memory : int, optional, default=10
        (Static) L-BFGS history length.
    verbose : int, optional

    Returns
    -------
    status : dict
        - ``'fin_x'`` — primal solution, shape (n,).
        - ``'fin_f'`` — objective at solution.
        - ``'fin_g'`` — inequality constraint values at solution, shape (m,).
        - ``'fin_z'`` — dual variables (Lagrange multipliers), shape (m,).
        - ``'niter'`` — iterations taken.
        - ``'converged'`` — bool.

    Notes
    -----
    **Convention conversion:**
    quadcoil uses g(x) <= 0; slsqp-jax uses c_ineq(x) >= 0.
    We pass ``-g(x)`` as the inequality constraint function to slsqp-jax.

    quadcoil uses h(x) = 0; slsqp-jax uses c_eq(x) = 0.  Same convention.
    '''
    opts = {**DEFAULT_SLSQP_OPTIONS}
    if solver_options is not None:
        opts.update(solver_options)

    atol = opts['atol']
    rtol = opts['rtol']

    x_init = jnp.asarray(x_init, dtype=jnp.float64)

    # Probe constraint dimensions
    g0 = g_ineq(x_init)
    h0 = h_eq(x_init)
    m_g = g0.shape[0]
    m_h = h0.shape[0]

    # ── Build slsqp-jax-compatible functions ─────────────────────────────
    # Objective: (x, args) -> (scalar, aux)
    def objective(x, args):
        return f_obj(x), None

    # Inequality: c_ineq(x) >= 0  <=>  -g(x) >= 0
    eq_fn = None
    ineq_fn = None

    if m_h > 0:
        def eq_fn(x, args):
            return h_eq(x)

    if m_g > 0:
        def ineq_fn(x, args):
            return -g_ineq(x)

    # ── Configure SLSQP solver ───────────────────────────────────────────
    from slsqp_jax import LBFGSConfig
    config = SLSQPConfig(
        tolerance=ToleranceConfig(atol=atol, rtol=rtol),
        lbfgs=LBFGSConfig(memory=lbfgs_memory),
    )

    solver = SLSQP(
        eq_constraint_fn=eq_fn,
        n_eq_constraints=m_h,
        ineq_constraint_fn=ineq_fn,
        n_ineq_constraints=m_g,
        config=config,
    )

    # ── Run optimization ─────────────────────────────────────────────────
    sol = optx.minimise(
        objective, solver, x_init,
        has_aux=True,
        max_steps=maxiter,
        throw=False,
    )

    x_opt = sol.value
    f_opt = f_obj(x_opt)
    g_opt = g_ineq(x_opt)

    # ── Extract multipliers ──────────────────────────────────────────────
    # Recover from KKT conditions (robust, solver-agnostic).
    if m_g > 0:
        z_ineq = _recover_multipliers(f_obj, g_ineq, x_opt)
    else:
        z_ineq = jnp.zeros(0, dtype=x_opt.dtype)

    # Determine convergence: compare result code as JAX-traced int
    # (avoids TracerBoolConversionError inside JIT)
    converged = sol.result == optx.RESULTS.successful

    niter = sol.stats.get('num_steps', jnp.int32(maxiter))

    if verbose > 0:
        debug.print(
            "SLSQP DONE: niter={it}  f={f:.6e}  converged={cv}",
            it=niter, f=f_opt, cv=converged,
        )
        if m_g > 0:
            debug.print(
                "  max(g) = {mg:.3e}  max(z) = {mz:.3e}",
                mg=jnp.max(g_opt), mz=jnp.max(z_ineq),
            )

    return {
        'fin_x':       x_opt,
        'fin_f':       f_opt,
        'fin_g':       g_opt,
        'fin_z':       z_ineq,
        'niter':       niter,
        'converged':   converged,
    }


# ── Stationarity (delegates to shared KKT module) ───────────────────────

def stationarity_slsqp(
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
    Build KKT stationarity data for implicit differentiation through the
    SLSQP solution.  Delegates to the shared :func:`kkt_adjoint.stationarity_kkt`.

    Parameters
    ----------
    (Same interface as stationarity_ipm.)

    Returns
    -------
    stationarity_data : dict
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


# ── Adjoint (delegates to shared KKT module) ────────────────────────────

def adjoint_slsqp(
    f_metric,
    stationarity_data,
    y_flat,
    implicit_linear_solver,
    verbose,
):
    r'''
    Compute dm/dy via the KKT adjoint system.
    Delegates to the shared :func:`kkt_adjoint.adjoint_kkt`.

    Parameters
    ----------
    (Same interface as adjoint_ipm.)

    Returns
    -------
    metric_value : scalar
    dfdy_arr : ndarray, shape (ny,)
    debug_info : dict
    '''
    return adjoint_kkt(
        f_metric=f_metric,
        stationarity_data=stationarity_data,
        y_flat=y_flat,
        implicit_linear_solver=implicit_linear_solver,
        verbose=verbose,
    )
