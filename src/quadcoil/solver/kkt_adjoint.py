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

For the adjoint, we solve  J_KKT^T v = [grad_x m, 0]  and then

    dm/dy = partial_y m  -  v^T @ (partial_y R)
"""

import jax
import jax.numpy as jnp
from jax import grad, jacrev, hessian, jvp, debug
import lineax as lx
from jax import config as config_jax
config_jax.update('jax_enable_x64', True)


def stationarity_kkt(
    constrained,
    convex,
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
    convex : bool
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
    '''
    stationarity_data = {
        'constrained': constrained,
        'x_flat_precond': x_opt,
    }

    if not constrained:
        def f_xy(x, y):
            f_obj, _, _, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y))
            return f_obj(flat_x_to_dofs(x))

        grad_x_f = jacrev(f_xy, argnums=0)
        tags = (lx.symmetric_tag, lx.positive_semidefinite_tag) if convex \
            else (lx.symmetric_tag,)
        vihp_hess = lx.JacobianLinearOperator(
            grad_x_f, x_opt, args=y_flat, tags=tags,
        )
        stationarity_data['vihp_hess'] = vihp_hess
        stationarity_data['grad_y_stationarity'] = jacrev(f_xy, argnums=1)

        if verbose > 0:
            hess_mat = jacrev(grad_x_f)(x_opt, y_flat)
            hess_cond = jnp.linalg.cond(hess_mat)
            stationarity_data['hess_cond'] = hess_cond
            debug.print('KKT unconstrained Hessian cond: {x}', x=hess_cond)

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

        stationarity_data['R_y'] = R_y

        if verbose > 0:
            J_cond = jnp.linalg.cond(J_KKT_mat)
            J_rank = jnp.linalg.matrix_rank(J_KKT_mat)
            stationarity_data['J_KKT_cond'] = J_cond
            stationarity_data['J_KKT_rank'] = J_rank
            debug.print(
                'KKT Jacobian — rank: {r}, cond: {c}',
                r=J_rank, c=J_cond,
            )

    return stationarity_data


def adjoint_kkt(
    f_metric,
    stationarity_data,
    y_flat,
    implicit_linear_solver,
    verbose,
):
    r'''
    Compute dm/dy via the KKT adjoint system.

    Unconstrained:
        H v = grad_x m,  then  dm/dy = grad_y m - v^T (d/dy grad_x f)

    Constrained:
        J_KKT^T v = [grad_x m, 0],  then  dm/dy = grad_y m - v^T (dR/dy)

    Parameters
    ----------
    f_metric : Callable
        ``(x_flat, y_flat) -> scalar``
    stationarity_data : dict
        Output of :func:`stationarity_kkt`.
    y_flat : ndarray
    implicit_linear_solver : lineax.AbstractLinearSolver
    verbose : int

    Returns
    -------
    metric_value : scalar
    dfdy_arr : ndarray, shape (ny,)
    debug_info : dict
    '''
    constrained = stationarity_data['constrained']
    x_opt = stationarity_data['x_flat_precond']

    grad_x_m = jacrev(f_metric, argnums=0)(x_opt, y_flat)
    grad_y_m = jacrev(f_metric, argnums=1)(x_opt, y_flat)
    metric_value = f_metric(x_opt, y_flat)

    debug_info = {}

    if not constrained:
        vihp_hess = stationarity_data['vihp_hess']
        vihp = lx.linear_solve(
            vihp_hess, grad_x_m, solver=implicit_linear_solver,
        ).value

        grad_y_stat = stationarity_data['grad_y_stationarity']
        grad_y_stat_at_y = lambda x, y_flat=y_flat: grad_y_stat(x, y_flat)
        _, dfdy1 = jvp(grad_y_stat_at_y, primals=[x_opt], tangents=[vihp])
        dfdy_arr = -dfdy1 + grad_y_m

        if verbose > 0:
            debug_info['vihp'] = vihp

    else:
        J_KKT_mat = stationarity_data['J_KKT_mat']
        n_x = stationarity_data['n_x']
        m_g = stationarity_data['m_g']
        R_y = stationarity_data['R_y']

        rhs = jnp.concatenate([grad_x_m, jnp.zeros(m_g)])
        J_KKT_T_op = lx.MatrixLinearOperator(J_KKT_mat.T)
        v = lx.linear_solve(
            J_KKT_T_op, rhs, solver=implicit_linear_solver,
        ).value

        _, vjp_fn = jax.vjp(R_y, y_flat)
        dRdy_T_v = vjp_fn(v)[0]
        dfdy_arr = grad_y_m - dRdy_T_v

        if verbose > 0:
            solve_err = jnp.linalg.norm(J_KKT_mat.T @ v - rhs)
            debug_info['v'] = v
            debug_info['J_KKT_solve_err'] = solve_err
            debug_info['J_KKT_cond'] = stationarity_data.get('J_KKT_cond', jnp.nan)
            debug_info['J_KKT_rank'] = stationarity_data.get('J_KKT_rank', jnp.nan)
            debug.print(
                'KKT adjoint solve error: {e}', e=solve_err,
            )

    return metric_value, dfdy_arr, debug_info
