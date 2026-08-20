"""
Augmented-Lagrangian solver with L-BFGS inner loop.

Potential further optimisations
-------------------------------
#5  In solve_constrained_auglag_lbfgs, the verbose>1 block inside
    body_fun_augmented_lagrangian evaluates grad_f, grad_g, grad_h
    for printing. These are gated behind a static `if` so they are
    compiled away when verbose<=1, but if verbose were ever traced
    they would become unnecessary overhead.
#6  stationarity_kkt computes two SVDs (one on the 3 Hessian
    terms, one on the concatenated projection).  For large n, consider
    making the SVD preconditioning optional or using a cheaper
    approximation (e.g. incomplete Cholesky).
"""
import jax.numpy as jnp
from functools import partial
from jax import vmap, grad, jacrev
import jax
from jax.lax import while_loop
from jax import config as config_jax
config_jax.update('jax_enable_x64', True)

import optimistix as optx

lstsq_vmap = vmap(jnp.linalg.lstsq)

def delta_normalized(x1, x2):
    diff = jnp.abs(x1-x2)
    max = jnp.maximum(jnp.abs(x1), jnp.abs(x2))
    return jnp.where(max>0, diff/max, max)

def solve_unconstrained_auglag_lbfgs(
    init_params,
    fun,
    convex,
    maxiter: int = 10000,
    solver_options=None,
    verbose=0,
    lbfgs_memory=10,
):
    r'''
    Performs unconstrained optimization using ``optimistix.LBFGS``.

    Parameters
    ----------
    init_params : ndarray, shape (N,)
        The initial condition.
    fun : Callable
        The objective function.
    convex : bool
        Whether to assume the problem is convex (unused, for interface
        compatibility).
    solver_options : dict
        L-BFGS options.  Recognised keys:

        - ``'atol'`` — absolute gradient tolerance.
        - ``'rtol'`` — relative gradient tolerance.
    maxiter : int, optional, default=10000
        (Static) Maximum L-BFGS iteration count.
    lbfgs_memory : int, optional, default=10
        (Static) L-BFGS history length.
    verbose : int
        Output levels of detail.

    Returns
    -------
    status : dict
        Contains the following entries:

        - ``'fin_f'`` — objective value at the optimum.
        - ``'fin_x'`` — the optimum, shape ``(N,)``.
        - ``'niter'`` — iteration count.
        - ``'fin_dx'`` — set to ``jnp.nan`` (not tracked by optimistix).
        - ``'fin_du'`` — set to ``jnp.nan`` (not tracked by optimistix).
        - ``'fin_df'`` — set to ``jnp.nan`` (not tracked by optimistix).
    '''
    atol = solver_options.get('atol', 1e-6) if solver_options else 1e-6
    rtol = solver_options.get('rtol', 1e-6) if solver_options else 1e-6

    solver = optx.LBFGS(
        rtol=rtol,
        atol=atol,
        history_length=lbfgs_memory,
    )

    def objective(y, args):
        return fun(y), None

    sol = optx.minimise(
        objective,
        solver,
        init_params,
        has_aux=True,
        max_steps=maxiter,
        throw=False,
    )

    x_opt = sol.value
    f_opt = fun(x_opt)
    niter = sol.stats.get('num_steps', jnp.int32(maxiter))

    if verbose > 0:
        jax.debug.print(
            'LBFGS done: niter={n}  f={f}',
            n=niter, f=f_opt,
        )

    return {
        'fin_f': f_opt,
        'fin_x': x_opt,
        'niter': niter,
        'fin_dx': jnp.nan,
        'fin_du': jnp.nan,
        'fin_df': jnp.nan,
    }

# Thresholding function for g+.
# The original one is gplus_hard.
# Introducing soft thresholding may improve differentiation behavior.
gplus_hard = lambda x, mu, c, g_ineq: jnp.maximum(g_ineq(x), -mu/c)

def gplus_elu(x, mu, c, g_ineq, scale=1):
    gval_shifted = g_ineq(x) + mu/c
    return jnp.where(
        gval_shifted<0,
        (jnp.exp(scale * gval_shifted) - 1)/scale - mu/c,
        gval_shifted - mu/c
    )
    
def gplus_softplus(x, mu, c, g_ineq, scale=1):
    gval_shifted = g_ineq(x) + mu/c
    return jnp.log(1 + jnp.exp(scale * gval_shifted))/scale - mu/c

def solve_constrained_auglag_lbfgs(
        x_init,
        f_obj,
        h_eq=lambda x:jnp.zeros(0),
        g_ineq=lambda x:jnp.zeros(0),
        convex=False,
        maxiter: int = 10000,
        maxiter_inner: int = 500,
        solver_options=None,
        verbose=0,
        c_k_safe=1e15,
        gplus_mask=gplus_hard,
        lbfgs_memory=10,
    ):
    r'''
    Solves the constrained optimization problem:

    .. math::

        \min_x f(x) \\
        \text{subject to } \\
        h(x) = 0, \\
        g(x) \leq 0 \\
        
    using the augmented Lagrangian method in 
    *Constrained Optimization and Lagrange Multiplier Methods* Chapter 3.
    Please refer to the chapter for notation.
    
    Parameters
    ----------  
    x_init : ndarray, shape (Nx,)
        The initial condition.
    f_obj : Callable
        The objective function.
    h_eq : Callable, optional
        The equality constraint function. 
        Must map ``x`` to an ``ndarray`` with shape ``(Nh,)``.
        No constraints by default.
    g_ineq : Callable, optional
        The inequality constraint function. 
        Must map ``x`` to an ``ndarray`` with shape ``(Ng,)``.
        No constraints by default.
    solver_options : dict
        Augmented-Lagrangian and inner-solver options. Recognised keys:

        - ``'c_init'`` (``1.``) — initial penalty :math:`c` factor.
        - ``'c_growth_rate'`` (``1.1``) — multiplicative growth of :math:`c` each outer step.
        - ``'lam_init'`` (``jnp.zeros(0)``) — initial :math:`\lambda` multiplier for equality constraints.
        - ``'mu_init'`` (``jnp.zeros(0)``) — initial :math:`\mu` multiplier for inequality constraints.
        - ``'xstop_outer'`` (``1e-7``) — outer-loop ``x`` convergence rate tolerance.
        - ``'ctol_outer'`` (``1e-7``) — outer-loop constraint-violation tolerance.
        - ``'atol_inner'`` (``1e-6``) — absolute gradient tolerance for inner L-BFGS solves.
        - ``'rtol_inner'`` (``1e-6``) — relative gradient tolerance for inner L-BFGS solves.
        - ``'atol_inner_last'`` (``1e-10``) — absolute gradient tolerance for the final inner solve.
        - ``'rtol_inner_last'`` (``1e-10``) — relative gradient tolerance for the final inner solve.
    maxiter : int, optional, default=10000
        (Static) Maximum total outer-loop iterations.
    maxiter_inner : int, optional, default=500
        (Static) Maximum inner L-BFGS iterations per outer step.
    verbose : int, optional, default=0
        (Static) Verbosity. ``>1`` prints outer iteration convergence info.
    c_k_safe : float, optional, default=1e15
        Upper bound on the penalty parameter :math:`c` to prevent divergence.
    gplus_mask : Callable, optional
        Thresholding function for :math:`g^+`. Defaults to ``gplus_hard``.
    lbfgs_memory : int, optional, default=10
        (Static) L-BFGS history length for the inner solver.

    Returns
    -------
    status : dict
        The end state of the iteration. Contains the following entries:

        - ``'niter'`` — total outer iteration count.
        - ``'outer_dx'`` — L2 norm of the change in ``x`` between the last two outer iterations.
        - ``'fin_f'`` — value of ``f`` at the optimum.
        - ``'fin_g'`` — value of ``g`` at the optimum.
        - ``'fin_h'`` — value of ``h`` at the optimum.
        - ``'fin_x'`` — the optimum, shape ``(Nx,)``.
        - ``'fin_l_aug'`` — augmented Lagrangian value at the optimum.
        - ``'fin_c'`` — final value of :math:`c`.
        - ``'fin_lam'`` — final :math:`\lambda` multiplier.
        - ``'fin_mu'`` — final :math:`\mu` multiplier.
        - ``'last_inner_niter'`` — inner L-BFGS iteration count in the last outer step.
        - ``'last_inner_dx'`` — L2 norm of the change in ``x`` in the last inner solve.
        - ``'last_inner_du'`` — L2 norm of the change in updates in the last inner solve.
        - ``'last_inner_dl'`` — change in the augmented Lagrangian in the last inner solve.
    '''
    # Reading solver options
    opts = solver_options or {}
    c_init=opts.get('c_init', 1.)
    c_growth_rate=opts.get('c_growth_rate', 1.1)
    mu_init=opts.get('mu_init', jnp.zeros(0))
    lam_init=opts.get('lam_init', jnp.zeros(0))
    xstop_outer=opts.get('xstop_outer', 1e-7)
    ctol_outer=opts.get('ctol_outer', 1e-7)
    atol_inner=opts.get('atol_inner', 1e-6)
    rtol_inner=opts.get('rtol_inner', 1e-6)
    atol_inner_last=opts.get('atol_inner_last', 1e-10)
    rtol_inner_last=opts.get('rtol_inner_last', 1e-10)

    # Has shape n_cons_ineq
    # gplus = lambda x, mu, c: jnp.max(jnp.array([g_ineq(x), -mu/c]), axis=0)
    gplus = partial(gplus_mask, g_ineq=g_ineq)
    grad_f = grad(f_obj)
    grad_g = jacrev(g_ineq)
    grad_h = jacrev(h_eq)
    if verbose>0:
        jax.debug.print(
            'SOLVER INITIALIZED. \nginit = {g} \nviolating elements: {c}', 
            g = g_ineq(x_init), 
            c = jnp.sum(jnp.where(g_ineq(x_init) > 0, 1., 0))
        )
    # True when non-convergent.
    # @jit
    def outer_convergence_criterion(dict_in):
        x_k = dict_in['fin_x']
        outer_dx = dict_in['outer_dx']
        tot_niter = dict_in['niter']
        g_k = dict_in['fin_g']
        h_k = dict_in['fin_h']
        c_k = dict_in['fin_c']
        # outer_dgrad_l = dict_in['outer_dgrad_l']
        # outer_dg = dict_in['outer_dg']
        # outer_dh = dict_in['outer_dh']
        # f_k = dict_in['fin_f']
        # This is the convergence condition (True when not converged yet)
        if verbose>1:
            jax.debug.print(
                'OUTER CONVERGENCE CRITERIA\n'\
                '    (tot_niter == 0): {x1}\n'\
                '    (tot_niter < maxiter): {x2}\n'\
                '    (outer_dx >= xstop_outer): {x3}\n'\
                '    (jnp.any(g_k >= ctol_outer) | jnp.any(jnp.abs(h_k) >= ctol_outer)): {x4}\n'\
                '    (c_k <= c_k_safe): {x5}\n',
                x1 = (tot_niter == 0),
                x2 = (tot_niter < maxiter),
                x3 = (outer_dx >= xstop_outer),
                x4 = (jnp.any(g_k >= ctol_outer) | jnp.any(jnp.abs(h_k) >= ctol_outer)),
                x5 = (c_k <= c_k_safe),
            )
        return(
            (tot_niter == 0) | (
                (tot_niter < maxiter) 
                # & (outer_dx >= xstop_outer * x_norm)
                & (
                    # Continue iteration when dx is significant
                    (outer_dx >= xstop_outer)
                    # Or when constraint violation is sufficiently strong,
                    # because sometimes the iteration terminates before 
                    # c becomes large enough. However, when c_k exceeds 
                    # our safe limit, to prevent endless outer iteration, 
                    # disble the constraint checking.
                    | (
                        (jnp.any(g_k >= ctol_outer) | jnp.any(jnp.abs(h_k) >= ctol_outer)) 
                        & (c_k <= c_k_safe)
                    )
                ) 
            )
        )

    # Recursion
    # @jit
    def body_fun_augmented_lagrangian(
        dict_in, 
        atol_inner=atol_inner,
        rtol_inner=rtol_inner,
    ):
        x_km1 = dict_in['fin_x']
        c_k = dict_in['fin_c']
        lam_k = dict_in['fin_lam']
        mu_k = dict_in['fin_mu']
        f_km1 = dict_in['fin_f']
        g_km1 = dict_in['fin_g']
        h_km1 = dict_in['fin_h']
        x_unit = dict_in['x_unit']
        # normalizing x with the sln from the previous step is not great either
        # abs_x_km1 = jnp.abs(x_km1)
        # mode_scaling = jnp.where(abs_x_km1>1e-5, abs_x_km1, 1e-5)
        # x_unit = x_unit_in * mode_scaling
        # grad_l_val_km1 = dict_in['outer_grad_l']
        # Eq (10) on p160 of Constrained Optimization and Multiplier Method
        l_k = lambda x, x_unit=x_unit, mu_k=mu_k, c_k=c_k: (
            f_obj(x*x_unit) 
            + lam_k@h_eq(x*x_unit) 
            + mu_k@gplus(x*x_unit, mu_k, c_k)
            + c_k/2 * (
                jnp.sum(h_eq(x*x_unit)**2) 
                + jnp.sum(gplus(x*x_unit, mu_k, c_k)**2)
            )
        ) 
        # Solving a stage of the problem
        inner_result = solve_unconstrained_auglag_lbfgs(
            x_km1/x_unit, l_k, 
            convex=convex,
            maxiter=maxiter_inner,
            solver_options={'atol': atol_inner, 'rtol': rtol_inner},
            verbose=verbose,
            lbfgs_memory=lbfgs_memory,
        )
        x_k_raw = inner_result['fin_x']
        val_l_k = inner_result['fin_f']
        niter_inner_k = inner_result['niter']
        dx_k = inner_result['fin_dx']
        du_k = inner_result['fin_du']
        dL_k = inner_result['fin_df']
        x_k = x_k_raw*x_unit
        x_norm = jnp.linalg.norm(x_k)
        x_unit_new = jnp.where(x_norm!=0, x_norm, 1.)
        f_k = f_obj(x_k)
        g_k = g_ineq(x_k)
        h_k = h_eq(x_k)
        gp_k = gplus(x_k, mu_k, c_k)
        # ----- Upsdating c and the multipliers
        # If constraints are sufficiently 
        # satisfied, or c is too large, 
        # or if the inner hasn't converged, 
        # update the multiplier only. 
        # otherwise, update c only.
        update_multiplier = (
            (
                # if all constraints are satisfied,
                jnp.all(g_k < ctol_outer) 
                & jnp.all(jnp.abs(h_k) < ctol_outer)
            )   # 
            | (c_k >= c_k_safe) 
            | (niter_inner_k >= maxiter_inner)
        )
        c_k_new = jnp.where(update_multiplier, c_k, c_k * c_growth_rate) 
        lam_k = lam_k + c_k * h_k
        mu_k = mu_k + c_k * gp_k
        df = jnp.linalg.norm(f_km1 - f_k)
        dg = jnp.linalg.norm(g_km1 - g_k)
        dh = jnp.linalg.norm(h_km1 - h_k)
        if verbose>1:
            jax.debug.print(
                'OUTER: \n'\
                '    Iteration: {tot_niter}/{maxiter}\n'\
                '        f       : {f}\n'\
                '        g       : {gmin}, {gmax}\n'\
                '        g+      : {gpmin}, {gpmax}\n'\
                '        h       : {hmin}, {hmax}\n'\
                '        |grad f|: {xx}\n'\
                '        |grad g|: {xg}\n'\
                '        |grad h|: {xh}\n'\
                '        mu      : {mu1}, {mu2}\n'\
                '        dmu     : {dmu1}, {dmu2}\n'\
                '        lam     : {lam1}, {lam2}\n'\
                '        dlam    : {dlam1}, {dlam2}\n'\
                '    Outer stopping criteria (False = satisfied)\n'\
                '        |x_k - x_km1| >= xstop_outer: {b}\n'\
                '        outer_dx    = {outer_dx}\n'\
                '        outer_df    = {outer_df}\n'\
                '        outer_dg    = {outer_dg}\n'\
                '        outer_dh    = {outer_dh}\n'\
                '        xstop_outer = {xstop_outer}\n'\
                # '    grad_l_val: {x}, d_grad_l_val: {dx}\n'\
                '    inner iter #: {z}\n'\
                '    c_k: {c_k}',
                f=f_k,
                gmin=_print_min_blank(g_k),
                gmax=_print_max_blank(g_k),
                gpmin=_print_min_blank(gp_k),
                gpmax=_print_max_blank(gp_k),
                hmin=_print_min_blank(h_k),
                hmax=_print_max_blank(h_k),
                c_k=c_k,
                mu1=_print_min_blank(mu_k),
                mu2=_print_max_blank(mu_k),
                lam1=_print_min_blank(lam_k),
                lam2=_print_max_blank(lam_k),
                dmu1=_print_min_blank(c_k * gp_k),
                dmu2=_print_max_blank(c_k * gp_k),
                dlam1=_print_min_blank(c_k * h_k),
                dlam2=_print_max_blank(c_k * h_k),
                xx=jnp.linalg.norm(grad_f(x_k)),
                xg=jnp.linalg.norm(grad_g(x_k)),
                xh=jnp.linalg.norm(grad_h(x_k)),
                z=niter_inner_k,
                tot_niter=dict_in['niter']+niter_inner_k,
                maxiter=maxiter,
                outer_dx=jnp.linalg.norm(x_k - x_km1),
                outer_df=df,
                outer_dg=dg,
                outer_dh=dh,
                xstop_outer=xstop_outer * jnp.linalg.norm(x_k),
                b=(jnp.linalg.norm(x_k - x_km1) >= xstop_outer),

            )
        # There is the possibility that the 
        dict_out = {
            'niter': dict_in['niter']+niter_inner_k,
            'outer_dx': jnp.linalg.norm(x_k - x_km1),
            'outer_df': df,
            'outer_dg': dg,
            'outer_dh': dh,
            'fin_f': f_k,
            'fin_g': g_k,
            'fin_h': h_k,
            'fin_x': x_k,
            'fin_l_aug': val_l_k,
            'fin_c': c_k_new,
            'fin_lam': lam_k,
            'fin_mu': mu_k,
            'last_inner_niter': niter_inner_k,
            'last_inner_dx': dx_k,
            'last_inner_du': du_k,
            'last_inner_dl': dL_k,
            'x_unit': x_unit_new,
        }
        return(dict_out)
    init_dict = {
        'niter': 0,       
        'outer_dx': 0.,
        'outer_df': 0.,
        'outer_dg': 0.,
        'outer_dh': 0.,
        'fin_f': f_obj(x_init),
        'fin_g': g_ineq(x_init),
        'fin_h': h_eq(x_init),
        'x_unit': 1.,
        'fin_x': x_init,
        'fin_l_aug': 0.,
        'fin_c': c_init,
        'fin_lam': lam_init,
        'fin_mu': mu_init,
        'last_inner_niter': 0,
        'last_inner_dx': 0.,
        'last_inner_du': 0.,
        'last_inner_dl': 0.,
    }
    # Apply a looser tolerance for most of the iteration
    result_dict = while_loop(
        cond_fun=outer_convergence_criterion,
        body_fun=body_fun_augmented_lagrangian,
        init_val=init_dict,
    )
    # Apply tight tolerance in the last iteration
    result_dict = body_fun_augmented_lagrangian(
        result_dict, 
        atol_inner=atol_inner_last,
        rtol_inner=rtol_inner_last,
    )
    return(result_dict)# Changes in f, g, h between the kth and k-1th iteration

def _print_min_blank(a):
    return jnp.min(a) if a.size > 0 else jnp.nan

def _print_max_blank(a):
    return jnp.max(a) if a.size > 0 else jnp.nan

# ── KKT-based stationarity and adjoint ───────────────────────────────────

def recover_multipliers(x_opt, y_flat, f_g_ineq_h_eq_from_y, unravel_y,
                        flat_x_to_dofs):
    r'''
    Recover Lagrange multipliers at the solution by solving the KKT
    stationarity condition as a least-squares problem:

    .. math::

        [J_g^T \mid J_h^T] \begin{bmatrix} z_{\rm ineq} \\ z_{\rm eq}
        \end{bmatrix} \approx -\nabla_x f(x^*)

    Parameters
    ----------
    x_opt : ndarray, shape (n,)
        Primal solution (scaled/flattened).
    y_flat : ndarray
        Flattened problem parameters.
    f_g_ineq_h_eq_from_y : Callable
        ``(y_dict) -> (f_obj, g_ineq, h_eq, n_g, n_h, aux_dofs)``
    unravel_y : Callable
        ``(y_flat) -> y_dict``
    flat_x_to_dofs : Callable
        ``(x_flat) -> dofs_dict``

    Returns
    -------
    z_ineq : ndarray, shape (n_g,)
        Recovered inequality multipliers.
    z_eq : ndarray, shape (n_h,)
        Recovered equality multipliers.
    '''
    f_obj, g_ineq, h_eq, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y_flat))
    f_scaled = lambda x: f_obj(flat_x_to_dofs(x))
    g_scaled = lambda x: g_ineq(flat_x_to_dofs(x))
    h_scaled = lambda x: h_eq(flat_x_to_dofs(x))

    grad_f = grad(f_scaled)(x_opt)
    J_g = jacrev(g_scaled)(x_opt)          # (n_g, n_x)
    J_h = jacrev(h_scaled)(x_opt)          # (n_h, n_x)
    A_T = jnp.concatenate([J_g.T, J_h.T], axis=1)  # (n_x, n_g + n_h)

    z_all = jnp.linalg.lstsq(A_T, -grad_f)[0]

    n_g = J_g.shape[0]
    return z_all[:n_g], z_all[n_g:]