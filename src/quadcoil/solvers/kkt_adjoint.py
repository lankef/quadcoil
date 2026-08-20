"""
Shared KKT-based stationarity and adjoint for implicit differentiation.

Both the IPM and SLSQP solvers produce (x_opt, z_opt) at a KKT point.
The implicit differentiation through the KKT conditions is identical
regardless of which solver found that point.  This module factors out
that shared logic so both solvers can reuse it.

KKT system (inequality-only, complementarity form):

    grad_x L = grad_x f + A^T z = 0     (stationarity)
    z_i * g_i(x) = 0                      (complementarity)

where z >= 0, g(x) <= 0.

The Jacobian of the KKT residual w.r.t. (x, z) is:

    J_KKT = [[H_xx,       A^T     ],
             [diag(z)*A,  diag(g) ]]

``adjoint_kkt`` takes a single batched metric callable that returns a
flattened concatenation of all metrics. Metric Jacobians are obtained
with one ``jacrev`` pass and one dense multi-RHS ``lstsq`` solve.
Cross-derivatives of the KKT residual / stationarity condition w.r.t.
problem parameters (``dRdy``, ``H_xy``) are never materialized; they
are applied via VJP against the adjoint rows. Adjoint mode is the only
mode, with the precondition ``n_metrics_flat <= n_y``.
"""

import jax.numpy as jnp
from jax import grad, jacrev, hessian, debug, vjp, vmap
from jax import config as config_jax
config_jax.update('jax_enable_x64', True)


def stationarity_kkt(
    constrained,
    x_opt,
    z_opt,
    y_flat,
    f_g_ineq_h_eq_from_y,
    unravel_y,
    flat_x_to_dofs,
    verbose,
):
    r'''
    Build the KKT stationarity data for implicit differentiation.

    This is solver-agnostic: any solver that produces (x_opt, z_opt) at
    a KKT point can call this function.

    Parameters
    ----------
    constrained : bool
    x_opt : ndarray, shape (n,)
        Primal solution.
    z_opt : ndarray, shape (m,) or None
        Dual variables (Lagrange multipliers for g <= 0).
        Ignored when ``constrained=False``.
    y_flat : ndarray
        Flattened problem parameters.
    f_g_ineq_h_eq_from_y : Callable
        ``(y_dict) -> (f_obj, g_ineq, h_eq, n_g, n_h, aux_dofs)``
    unravel_y : Callable
        ``(y_flat) -> y_dict``
    flat_x_to_dofs : Callable
        ``(x_flat) -> dofs_dict``
    verbose : int

    Returns
    -------
    stationarity_data : dict
        Unconstrained keys: ``H_mat``, ``grad_y_stationarity``.
        Constrained keys: ``J_KKT_mat``, ``R_y``, ``z_opt``, ``n_x``, ``m_g``.
    '''
    stationarity_data = {
        'constrained': constrained,
        'x_flat_precond': x_opt,
    }

    if not constrained:
        def f_xy(x, y):
            f_obj, _, _, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y))
            return f_obj(flat_x_to_dofs(x))

        H_mat = hessian(lambda x: f_xy(x, y_flat))(x_opt)  # (n_x, n_x)
        # Closure for VJP: maps y -> grad_x f(x_opt, y). Never materialize H_xy.
        grad_y_stationarity = lambda y: grad(lambda x: f_xy(x, y))(x_opt)
        stationarity_data['H_mat'] = H_mat
        stationarity_data['grad_y_stationarity'] = grad_y_stationarity

    else:
        n_x = x_opt.shape[0]
        m_g = z_opt.shape[0]

        def _f_scaled(x, y=y_flat):
            f_obj, _, _, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y))
            return f_obj(flat_x_to_dofs(x))

        def _g_scaled(x, y=y_flat):
            _, g_ineq, _, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y))
            return g_ineq(flat_x_to_dofs(x))

        def _lagrangian(x, y=y_flat, z=z_opt):
            return _f_scaled(x, y) + jnp.dot(z, _g_scaled(x, y))

        H_xx = hessian(_lagrangian)(x_opt)
        A = jacrev(_g_scaled)(x_opt)
        g_val = _g_scaled(x_opt)

        # J_KKT = [[H_xx,       A^T     ],
        #          [diag(z)*A,  diag(g) ]]
        top = jnp.concatenate([H_xx, A.T], axis=1)
        bottom = jnp.concatenate([jnp.diag(z_opt) @ A, jnp.diag(g_val)], axis=1)
        J_KKT_mat = jnp.concatenate([top, bottom], axis=0)
        J_KKT_mat = jnp.nan_to_num(J_KKT_mat, nan=0.0, posinf=0.0, neginf=0.0)

        stationarity_data['J_KKT_mat'] = J_KKT_mat
        stationarity_data['z_opt'] = z_opt
        stationarity_data['n_x'] = n_x
        stationarity_data['m_g'] = m_g

        # R_y: KKT residual as a function of y (x, z frozen at optimum).
        # Stored as a closure; adjoint_kkt applies it via VJP.
        def R_y(y, x=x_opt, z=z_opt):
            f_obj_y, g_ineq_y, _, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y))
            f_sc = lambda xx: f_obj_y(flat_x_to_dofs(xx))
            g_sc = lambda xx: g_ineq_y(flat_x_to_dofs(xx))
            L_x = lambda xx: f_sc(xx) + jnp.dot(z, g_sc(xx))
            grad_x_L = grad(L_x)(x)
            g_v = g_sc(x)
            return jnp.concatenate([grad_x_L, z * g_v])

        stationarity_data['R_y'] = R_y

    return stationarity_data


def adjoint_kkt(
    f_metrics_flat,
    stationarity_data,
    y_flat,
    verbose,
):
    r'''
    Compute dm/dy for a batch of flattened metrics via the KKT adjoint.

    Unconstrained:
        H V^T = J_x^T,  then  dm/dy = J_y - VJP(grad_y_stationarity, V)

    Constrained:
        J_KKT^T V^T = [J_x, 0]^T,  then  dm/dy = J_y - VJP(R_y, V)

    Requires ``n_metrics_flat <= n_y`` (adjoint-only).

    Parameters
    ----------
    f_metrics_flat : Callable
        ``(x_flat, y_flat) -> (n_metrics_flat,)`` — all metrics flattened and
        concatenated into one vector.
    stationarity_data : dict
        Output of :func:`stationarity_kkt`.
    y_flat : ndarray
    verbose : int

    Returns
    -------
    all_values : ndarray, shape (n_metrics_flat,)
    dfdy : ndarray, shape (n_metrics_flat, n_y)
    debug_info : dict
    '''
    constrained = stationarity_data['constrained']
    x_opt = stationarity_data['x_flat_precond']
    n_y = y_flat.shape[0]

    all_values = f_metrics_flat(x_opt, y_flat)  # (n_metrics_flat,)
    J_x = jacrev(f_metrics_flat, argnums=0)(x_opt, y_flat)  # (n_metrics_flat, n_x)
    J_y = jacrev(f_metrics_flat, argnums=1)(x_opt, y_flat)  # (n_metrics_flat, n_y)

    n_metrics_flat = J_x.shape[0]
    if n_metrics_flat > n_y:
        raise ValueError(
            f'adjoint_kkt requires n_metrics_flat <= n_y, got '
            f'{n_metrics_flat} > {n_y}. Reduce the number of array-valued '
            'metrics or restore the forward-mode branch.'
        )

    debug_info = {}

    if not constrained:
        H_mat = stationarity_data['H_mat']  # (n_x, n_x)
        grad_y_stationarity = stationarity_data['grad_y_stationarity']
        # Adjoint: factor once, n_metrics_flat RHS
        V = jnp.linalg.lstsq(H_mat, J_x.T)[0].T  # (n_metrics_flat, n_x)
        _, vjp_fn = vjp(grad_y_stationarity, y_flat)
        dfdy = J_y - vmap(lambda v: vjp_fn(v)[0])(V)  # (n_metrics_flat, n_y)
        if verbose > 0:
            debug_info['vihp'] = V

    else:
        J_KKT_mat = stationarity_data['J_KKT_mat']  # (n_w, n_w)
        R_y = stationarity_data['R_y']
        m_g = stationarity_data['m_g']
        # Adjoint: J_KKT^T V^T = [J_x, 0]^T
        rhs = jnp.concatenate(
            [J_x, jnp.zeros((n_metrics_flat, m_g))], axis=1,
        ).T  # (n_w, n_metrics_flat)
        V = jnp.linalg.lstsq(J_KKT_mat.T, rhs)[0].T  # (n_metrics_flat, n_w)
        _, vjp_fn = vjp(R_y, y_flat)
        dfdy = J_y - vmap(lambda v: vjp_fn(v)[0])(V)  # (n_metrics_flat, n_y)
        if verbose > 0:
            solve_err = jnp.linalg.norm(J_KKT_mat.T @ V.T - rhs)
            debug_info['v'] = V
            debug_info['J_KKT_solve_err'] = solve_err
            debug.print(
                'KKT adjoint solve error: {e}', e=solve_err,
            )

    return all_values, dfdy, debug_info
