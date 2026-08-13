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
flattened concatenation of all metrics. Gradients are obtained with one
jacrev pass and one dense multi-RHS ``lstsq`` solve. Forward vs adjoint
mode is selected statically from ``metric_K`` vs ``n_y``.
"""

import jax.numpy as jnp
from jax import grad, jacrev, hessian, debug
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
        Unconstrained keys: ``H_mat``, ``H_xy``.
        Constrained keys: ``J_KKT_mat``, ``dRdy``, ``z_opt``, ``n_x``, ``m_g``.
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
        H_xy = jacrev(
            lambda y: grad(lambda x: f_xy(x, y))(x_opt)
        )(y_flat)  # (n_x, n_y)
        stationarity_data['H_mat'] = H_mat
        stationarity_data['H_xy'] = H_xy

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

        # R_y: KKT residual as a function of y (x, z frozen at optimum)
        def R_y(y, x=x_opt, z=z_opt):
            f_obj_y, g_ineq_y, _, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y))
            f_sc = lambda xx: f_obj_y(flat_x_to_dofs(xx))
            g_sc = lambda xx: g_ineq_y(flat_x_to_dofs(xx))
            L_x = lambda xx: f_sc(xx) + jnp.dot(z, g_sc(xx))
            grad_x_L = grad(L_x)(x)
            g_v = g_sc(x)
            return jnp.concatenate([grad_x_L, z * g_v])

        dRdy = jacrev(R_y)(y_flat)  # (n_w, n_y)
        stationarity_data['dRdy'] = dRdy

    return stationarity_data


def adjoint_kkt(
    f_metrics_flat,
    metric_K,
    stationarity_data,
    y_flat,
    verbose,
):
    r'''
    Compute dm/dy for a batch of flattened metrics via the KKT system.

    Unconstrained:
        H V^T = J_x^T,  then  dm/dy = J_y - V @ H_xy
        (or the forward form when ``metric_K > n_y``).

    Constrained:
        J_KKT^T V^T = [J_x, 0]^T,  then  dm/dy = J_y - V @ dRdy
        (or the forward form when ``metric_K > n_y``).

    Parameters
    ----------
    f_metrics_flat : Callable
        ``(x_flat, y_flat) -> (K_tot,)`` — all metrics flattened and
        concatenated into one vector.
    metric_K : int
        ``K_tot``. Must be a concrete Python int (static under JIT).
    stationarity_data : dict
        Output of :func:`stationarity_kkt`.
    y_flat : ndarray
    verbose : int

    Returns
    -------
    all_values : ndarray, shape (K_tot,)
    dfdy : ndarray, shape (K_tot, n_y)
    debug_info : dict
    '''
    constrained = stationarity_data['constrained']
    x_opt = stationarity_data['x_flat_precond']
    n_y = y_flat.shape[0]

    all_values = f_metrics_flat(x_opt, y_flat)  # (K_tot,)
    J_x = jacrev(f_metrics_flat, argnums=0)(x_opt, y_flat)  # (K_tot, n_x)
    J_y = jacrev(f_metrics_flat, argnums=1)(x_opt, y_flat)  # (K_tot, n_y)

    debug_info = {}

    if not constrained:
        H_mat = stationarity_data['H_mat']  # (n_x, n_x)
        H_xy = stationarity_data['H_xy']    # (n_x, n_y)
        if metric_K <= n_y:
            # Adjoint: factor once, K_tot RHS
            V = jnp.linalg.lstsq(H_mat, J_x.T)[0].T  # (K_tot, n_x)
            dfdy = J_y - V @ H_xy                     # (K_tot, n_y)
            if verbose > 0:
                debug_info['vihp'] = V
        else:
            # Forward: factor once, n_y RHS
            S = jnp.linalg.lstsq(H_mat, -H_xy)[0]  # (n_x, n_y)
            dfdy = J_y + J_x @ S                    # (K_tot, n_y)
            if verbose > 0:
                debug_info['S'] = S

    else:
        J_KKT_mat = stationarity_data['J_KKT_mat']  # (n_w, n_w)
        dRdy = stationarity_data['dRdy']            # (n_w, n_y)
        n_x = stationarity_data['n_x']
        m_g = stationarity_data['m_g']
        if metric_K <= n_y:
            # Adjoint: J_KKT^T V^T = [J_x, 0]^T
            rhs = jnp.concatenate(
                [J_x, jnp.zeros((metric_K, m_g))], axis=1,
            ).T  # (n_w, K_tot)
            V = jnp.linalg.lstsq(J_KKT_mat.T, rhs)[0].T  # (K_tot, n_w)
            dfdy = J_y - V @ dRdy                         # (K_tot, n_y)
            if verbose > 0:
                solve_err = jnp.linalg.norm(J_KKT_mat.T @ V.T - rhs)
                debug_info['v'] = V
                debug_info['J_KKT_solve_err'] = solve_err
                debug.print(
                    'KKT adjoint solve error: {e}', e=solve_err,
                )
        else:
            # Forward: J_KKT S = -dRdy
            S = jnp.linalg.lstsq(J_KKT_mat, -dRdy)[0]  # (n_w, n_y)
            dfdy = J_y + J_x @ S[:n_x]                  # (K_tot, n_y)
            if verbose > 0:
                debug_info['S'] = S

    return all_values, dfdy, debug_info
