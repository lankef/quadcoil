"""
Augmented-Lagrangian solver with L-BFGS inner loop.

Potential further optimisations
-------------------------------
#5  In solve_constrained_auglag_lbfgs, the verbose>1 block inside
    body_fun_augmented_lagrangian evaluates grad_f, grad_g, grad_h
    for printing. These are gated behind a static `if` so they are
    compiled away when verbose<=1, but if verbose were ever traced
    they would become unnecessary overhead.
#6  stationarity_auglag_lbfgs computes two SVDs (one on the 3 Hessian
    terms, one on the concatenated projection).  For large n, consider
    making the SVD preconditioning optional or using a cheaper
    approximation (e.g. incomplete Cholesky).
"""
import warnings
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from functools import partial
from jax import jit, vmap, grad, jacrev, jvp, hessian, debug
import lineax as lx
import jax
from jax.lax import while_loop
from jax import config as config_jax
config_jax.update('jax_enable_x64', True)

from .kkt_adjoint import stationarity_kkt, adjoint_kkt

lstsq_vmap = vmap(jnp.linalg.lstsq)

# def wl_debug(cond_fun, body_fun, init_val):
#     val = init_val
#     iter_num_wl = 1
#     while cond_fun(val):
#         val = body_fun(val)
#     return val
def delta_normalized(x1, x2):
    diff = jnp.abs(x1-x2)
    max = jnp.maximum(jnp.abs(x1), jnp.abs(x2))
    return jnp.where(max>0, diff/max, max)

def solve_unconstrained_auglag_lbfgs(
    init_params, 
    fun, 
    convex,
    solver_options, 
    max_linesearch_steps, 
    verbose
):
    r'''
    Performs unconstrained optimization using ``optax.lbfgs``.
    
    Parameters
    ----------  
    init_params : ndarray, shape (N,)
        The initial condition.
    fun : Callable
        The objective function.
    solver_options : dict
        (Traced) LBFGS options. Recognised keys:

        - ``'maxiter'`` — maximum iteration count.
        - ``'fstop'`` — objective convergence rate tolerance.
        - ``'xstop'`` — unknown convergence rate tolerance.
        - ``'gtol'`` — gradient tolerance.
    max_linesearch_steps : int
        The maximum steps in the LBFGS zoom line search.
    verbose : int
        Output levels of detail.

    Returns
    -------
    status : dict
        Contains the following entries:

        - ``'fin_f'`` — objective value at the optimum.
        - ``'fin_x'`` — the optimum, shape ``(N,)``.
        - ``'niter'`` — iteration count.
        - ``'fin_dx'`` — L2 norm of the change in x at termination.
        - ``'fin_du'`` — L2 norm of the change in updates at termination.
        - ``'fin_df'`` — change in f at termination.
    '''
    maxiter = solver_options['maxiter']
    fstop = solver_options['fstop']
    xstop = solver_options['xstop']
    gtol = solver_options['gtol']
    x, f, g, niter, dx, du, df = run_opt_optax(
        init_params, 
        fun, 
        maxiter, 
        fstop, xstop, gtol, 
        opt=optax.lbfgs(
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=max_linesearch_steps, 
                initial_guess_strategy='one'
            )
        ), 
        verbose=verbose
    )
    return {
        'fin_f': f,
        'fin_x': x,
        'niter': niter,
        'fin_dx': dx,
        'fin_du': du,
        'fin_df': df,
    }

def run_opt_optax(init_params, fun, maxiter, fstop, xstop, gtol, opt, verbose, n_history=10):
    r'''
    A wrapper for performing unconstrained optimization using ``optax.base.GradientTransformationExtraArgs``.
    
    Parameters
    ----------  
    init_params : ndarray, shape (N,)
        The initial condition.
    fun : Callable
        The objective function.
    maxiter : int
        The maximum iteration number.
    fstop : float
        The objective function convergence rate tolerance. 
        Terminates when any one of the tolerances is satisfied.
    xstop : float
        The unknown convergence rate tolerance. 
        Terminates when any one of the tolerances is satisfied.
    gtol : float
        The gradient tolerance. 
        Terminates when any one of the tolerances is satisfied.
    opt : optax.base.GradientTransformationExtraArgs
        The optimizer of choice.
    
    Returns
    -------
    x : ndarray, shape (N,)
        The optimum.
    f : float
        The objective at the optimum.
    grad : ndarray, shape (N,)
        The gradient at the optimum.
    count : int
        The iteration number.
    final_dx : float
        The rate of change of x at the optimum.
    final_du : float
        The rate of change of updates at the optimum.
    final_df : float
        The rate of change of f at the optimum.
    '''
    init_val = fun(init_params)
    init_carry = (
        init_params,  # params
        jnp.zeros_like(init_params), # update
        init_val, # value
        jnp.linspace(1000*fstop, 0, n_history) + init_val * 2, # val_rec
        jnp.zeros_like(init_params), # dx
        jnp.zeros_like(init_params), # du
        0, # df
        # 0., 0., 0., 0.,
        opt.init(init_params) # state1
    )
    g0 = grad(fun)(init_params)
    g0_norm = jnp.linalg.norm(g0)
    g0_max = jnp.max(jnp.abs(g0))
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
    if verbose>1:
        jax.debug.print('INNER: starting gradient L2 norm: {a}', a=g0_norm)
    # Carry is params, update, value, val_rec, dx, du, df, state1
    def step(carry):
        params1, updates1, value1, val_rec, _, _, _, state1 = carry
        value2, grad2 = value_and_grad_fun(params1, state=state1)
        updates2, state2 = opt.update(
            grad2, state1, params1, value=value2, grad=grad2, value_fn=fun
        )
        params2 = optax.apply_updates(params1, updates2)
        return(
            params2, updates2, value2, 
            jnp.append(val_rec[1:], value2),
            jnp.abs(params2 - params1), # jnp.linalg.norm(params2 - params1), 
            jnp.abs(updates2 - updates1), # jnp.linalg.norm(updates2 - updates1), 
            jnp.abs(value2 - value1), 
            # jnp.linalg.norm(delta_normalized(params2, params1)), 
            # jnp.linalg.norm(delta_normalized(updates2, updates1)), 
            # delta_normalized(value2, value1), 
            state2
        )
  
    def continuing_criterion(carry):
        params, _, value, val_rec, dx, du, df, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_norm(grad)
        # DEBUG 
        param2 = dx + params
        dx1 = param2 - params
        dx_norm = jnp.linalg.norm(dx)
        du_norm = jnp.linalg.norm(du)
        params_norm = jnp.linalg.norm(params)
        avg_improvement = jnp.average(val_rec[:-1] - val_rec[1:])
        if verbose>2:
            jax.debug.print(
                'INNER: L: {l}, dx: {dx}, du: {du}, df: {df}, \n'\
                '    grad:{g}, grad/g0:{gnorm}, Average improvement: {adf}\n'\
                '    Value record: {val_rec}\n'\
                '    Stopping criteria: \n'
                '(iter_num < maxiter): {a}\n'
                '& (err > gtol) : {b}\n'
                '& (avg_improvement > fstop): {ff}'
                '& ((dx_norm > xstop) | (du_norm > xstop) | (df > fstop)): {c}, {d}, {e}\n'
                '(dx_norm > xstop): {dx_norm} > {xstop})\n'
                '(du_norm > xstop): {du_norm} > {xstop}\n'
                '(df > fstop):      {df} > {fstop}\n'
                '',
                adf=avg_improvement,
                val_rec=val_rec,
                a=(iter_num < maxiter),
                b=(err > gtol),
                c=(dx_norm > xstop),
                d=(du_norm > xstop),
                e=(df > fstop),
                ff=(avg_improvement > fstop),
                l=value,
                dx=jnp.max(dx),
                du=jnp.max(du),
                g=err,
                gnorm=err/g0_norm,
                dx_norm=dx_norm,
                du_norm=du_norm,
                xstop=xstop,
                df=df,
                fstop=fstop,
            )
        return (iter_num == 0) | (
            (iter_num < maxiter) 
            & (err > gtol) 
            & (avg_improvement > fstop) # Added May 27
            & ((dx_norm > xstop) | (du_norm > xstop) | (df > fstop)) # The last one is added on May 19
            # & ((dx_norm > xstop * params_norm) | (du_norm > xstop * params_norm))
            # & (df > fstop * value) 
        )
    final_params, final_updates, final_value, val_rec, final_dx, final_du, final_df, final_state = while_loop(
        continuing_criterion, step, init_carry
    )
    return(
        final_params, 
        final_value,
        otu.tree_get(final_state, 'grad'), 
        otu.tree_get(final_state, 'count'),
        jnp.linalg.norm(final_dx),# final_dx, # Changes in x
        jnp.linalg.norm(final_du),# final_du, # Changes in u
        final_df, # Changes in f
    )

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
        solver_options={
            'c_init': 1.,
            'c_growth_rate': 1.1,
            'mu_init': jnp.zeros(0),
            'lam_init': jnp.zeros(0),
            'xstop_outer': 1e-7, # convergence rate tolerance
            'ctol_outer': 1e-7, # constraint tolerance, used in multiplier update
            'fstop_inner': 1e-7,
            'xstop_inner': 1e-7,
            'gtol_inner': 1e-7,
            'fstop_inner_last': 1e-7,
            'xstop_inner_last': 1e-7,
            'gtol_inner_last': 1e-7,
            'maxiter_tot': 10000,
            'maxiter_inner': 500,
        },
        max_linesearch_steps=20,
        verbose=0,
        c_k_safe=1e15,
        gplus_mask=gplus_hard,
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
        - ``'fstop_inner'`` (``1e-7``) — inner ``f`` convergence rate tolerance.
        - ``'xstop_inner'`` (``1e-7``) — inner ``x`` convergence rate tolerance.
        - ``'gtol_inner'`` (``1e-7``) — inner gradient tolerance.
        - ``'fstop_inner_last'`` (``1e-7``) — ``f`` tolerance for the final inner solve.
        - ``'xstop_inner_last'`` (``1e-7``) — ``x`` tolerance for the final inner solve.
        - ``'gtol_inner_last'`` (``1e-7``) — gradient tolerance for the final inner solve.
        - ``'maxiter_tot'`` (``10000``) — maximum total outer-loop iterations.
        - ``'maxiter_inner'`` (``500``) — maximum inner L-BFGS iterations per outer step.
    max_linesearch_steps : int, optional, default=20
        (Static) Maximum steps in the L-BFGS zoom line search.
    verbose : int, optional, default=0
        (Static) Verbosity. ``>1`` prints outer iteration convergence info.
    c_k_safe : float, optional, default=1e15
        Upper bound on the penalty parameter :math:`c` to prevent divergence.
    gplus_mask : Callable, optional
        Thresholding function for :math:`g^+`. Defaults to ``gplus_hard``.

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
    c_init=solver_options['c_init']
    c_growth_rate=solver_options['c_growth_rate']
    mu_init=solver_options['mu_init']
    lam_init=solver_options['lam_init']
    xstop_outer=solver_options['xstop_outer']
    ctol_outer=solver_options['ctol_outer']
    fstop_inner=solver_options['fstop_inner']
    xstop_inner=solver_options['xstop_inner']
    gtol_inner=solver_options['gtol_inner']
    fstop_inner_last=solver_options['fstop_inner_last']
    xstop_inner_last=solver_options['xstop_inner_last']
    gtol_inner_last=solver_options['gtol_inner_last']
    maxiter_tot=solver_options['maxiter_tot']
    maxiter_inner=solver_options['maxiter_inner']

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
                '    (tot_niter < maxiter_tot): {x2}\n'\
                '    (outer_dx >= xstop_outer): {x3}\n'\
                '    (jnp.any(g_k >= ctol_outer) | jnp.any(jnp.abs(h_k) >= ctol_outer)): {x4}\n'\
                '    (c_k <= c_k_safe): {x5}\n',
                x1 = (tot_niter == 0),
                x2 = (tot_niter < maxiter_tot),
                x3 = (outer_dx >= xstop_outer),
                x4 = (jnp.any(g_k >= ctol_outer) | jnp.any(jnp.abs(h_k) >= ctol_outer)),
                x5 = (c_k <= c_k_safe),
            )
        return(
            (tot_niter == 0) | (
                (tot_niter < maxiter_tot) 
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
        gtol_inner=gtol_inner, 
        fstop_inner=fstop_inner, 
        xstop_inner=xstop_inner
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
            solver_options={
                'maxiter': maxiter_inner,
                'fstop': fstop_inner,
                'xstop': xstop_inner,
                'gtol': gtol_inner,
            },
            max_linesearch_steps=max_linesearch_steps,
            verbose=verbose
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
                '    Iteration: {tot_niter}/{maxiter_tot}\n'\
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
                maxiter_tot=maxiter_tot,
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
        gtol_inner=gtol_inner_last, 
        fstop_inner=fstop_inner_last, 
        xstop_inner=xstop_inner_last
    )
    return(result_dict)# Changes in f, g, h between the kth and k-1th iteration

def _print_min_blank(a):
    return jnp.min(a) if a.size > 0 else jnp.nan

def _print_max_blank(a):
    return jnp.max(a) if a.size > 0 else jnp.nan


def stationarity_auglag_lbfgs_legacy(
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
    (Legacy) Build the augmented-Lagrangian stationarity condition and precompute the
    Hessian / preconditioner needed for implicit differentiation.

    Called once per ``quadcoil()`` invocation; the returned ``stationarity_data``
    dict is then passed into :func:`adjoint_auglag_lbfgs_legacy` once per metric.

    Parameters
    ----------
    constrained : bool
        Whether the problem has constraints.
    convex : bool
        Whether to assume the problem is convex.
    solve_results : dict
        Output of :func:`solve_constrained_auglag_lbfgs` or
        :func:`solve_unconstrained_auglag_lbfgs`.
    y_flat : ndarray
        Flattened problem parameters.
    f_g_ineq_h_eq_from_y : Callable
        ``(y_dict) -> (f_obj, g_ineq, h_eq, n_g, n_h, aux_dofs)``
    unravel_y : Callable
        ``(y_flat) -> y_dict``
    unravel_unscale_x : Callable
        ``(x_flat) -> dofs_dict``
    solver_options : dict
        Solver options; uses ``'svtol'`` for singular-value cut-off.
    verbose : int
        Verbosity level.

    Returns
    -------
    stationarity_data : dict
        Opaque state consumed by :func:`adjoint_auglag_lbfgs_legacy`.
    '''
    svtol = solver_options.get('svtol', 1e-6)
    x_flat_opt = solve_results['fin_x']
    x_flat_precond = x_flat_opt

    stationarity_data = {
        'constrained': constrained,
        'x_flat_precond': x_flat_precond,
    }

    if not constrained:
        # ----- Unconstrained stationarity -----
        def l_k(x, y):
            f_obj, _, _, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y))
            return f_obj(unravel_unscale_x(x))

        grad_x_l_k = jacrev(l_k, argnums=0)
        if convex:
            vihp_hess = lx.JacobianLinearOperator(
                grad_x_l_k,
                x_flat_opt, args=y_flat,
                tags=(lx.symmetric_tag, lx.positive_semidefinite_tag)
            )
        else:
            vihp_hess = lx.JacobianLinearOperator(
                grad_x_l_k,
                x_flat_opt, args=y_flat,
                tags=(lx.symmetric_tag,)
            )
        stationarity_data['vihp_hess'] = vihp_hess

        if verbose > 0:
            hess_l_k_mat = jacrev(grad_x_l_k)(x_flat_opt, y_flat)
            hess_cond = jnp.linalg.cond(hess_l_k_mat)
            stationarity_data['hess_cond'] = hess_cond
            debug.print('Unconstrained Hessian condition number: {x}', x=hess_cond)

    else:
        # ----- Constrained stationarity (augmented Lagrangian) -----
        c_k = solve_results['fin_c']
        mu_k = solve_results['fin_mu']
        lam_k = solve_results['fin_lam']

        def l_k_terms(x, y=y_flat, mu=mu_k, lam=lam_k, c=c_k):
            f_obj_temp, g_ineq_temp, h_eq_temp, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y))
            f_scaled_temp = lambda x_flat: f_obj_temp(unravel_unscale_x(x_flat))
            g_scaled_temp = lambda x_flat: g_ineq_temp(unravel_unscale_x(x_flat))
            h_scaled_temp = lambda x_flat: h_eq_temp(unravel_unscale_x(x_flat))
            gplus_temp = partial(gplus_hard, g_ineq=g_scaled_temp)
            return jnp.array([
                f_scaled_temp(x),
                lam@h_scaled_temp(x) + mu@gplus_temp(x, mu, c),
                c/2 * (
                    jnp.sum(h_scaled_temp(x)**2)
                    + jnp.sum(gplus_temp(x, mu, c)**2)
                )
            ])

        l_k = lambda x, y=y_flat, mu=mu_k, lam=lam_k, c=c_k: jnp.sum(
            l_k_terms(x=x, y=y, mu=mu, lam=lam, c=c)
        )

        # ----- Preconditioning -----
        # An important source of ill-conditioning in Hess(l_k) is the
        # difference in the three terms' orders of magnitude. We sort them
        # as A, B, C (ascending max singular value) and rescale each in
        # directions linearly independent from the larger terms.
        hess_l_k_terms_val = hessian(l_k_terms)(x_flat_precond)
        hess_l_k = jnp.sum(hess_l_k_terms_val, axis=0)
        hess_l_k = jnp.nan_to_num(hess_l_k, nan=0.0, posinf=0.0, neginf=0.0)
        hess_l_k_terms_val = 0.5 * (
            hess_l_k_terms_val
            + jnp.swapaxes(hess_l_k_terms_val, 1, 2)
        )
        U, s, VH = jnp.linalg.svd(hess_l_k_terms_val)
        s_max = jnp.max(s, axis=1)
        s_selection = s >= svtol * s_max[:, None]
        hess_order = jnp.argsort(s_max)

        A = hess_l_k_terms_val[hess_order[0]]
        B = hess_l_k_terms_val[hess_order[1]]
        C = hess_l_k_terms_val[hess_order[2]]
        VH_B = VH[hess_order[1]]
        VH_C = VH[hess_order[2]]
        s_selection_B = s_selection[hess_order[1]]
        s_selection_C = s_selection[hess_order[2]]

        proj_C  = VH_C.T @ (   s_selection_C[:, None] * VH_C)
        proj_B  = VH_B.T @ (   s_selection_B[:, None] * VH_B)
        annil_C = VH_C.T @ ((~s_selection_C)[:, None] * VH_C)

        U_BC, s_BC, VH_BC = jnp.linalg.svd(jnp.concatenate([proj_B, proj_C]))
        s_BC_selection = s_BC >= svtol * jnp.max(s_BC)
        proj_BC  = VH_BC.T @ (  s_BC_selection [:, None] * VH_BC)
        annil_BC = VH_BC.T @ ((~s_BC_selection)[:, None] * VH_BC)

        scale_AC = jnp.where(s_max[hess_order[0]] > 0, s_max[hess_order[2]] / s_max[hess_order[0]], 0)
        scale_BC = jnp.where(s_max[hess_order[1]] > 0, s_max[hess_order[2]] / s_max[hess_order[1]], 0)

        OC = C
        OB = scale_BC * proj_BC @ annil_C @ B + proj_C @ B
        OA = (
            scale_AC * annil_BC @ annil_C @ A
            + scale_BC * proj_BC @ annil_C @ A
            + proj_C @ A
        )
        Ohess = OA + OB + OC
        Ohess = jnp.nan_to_num(Ohess, nan=0.0, posinf=0.0, neginf=0.0)

        stationarity_data['vihp_A_raw'] = lx.MatrixLinearOperator(hess_l_k)
        stationarity_data['vihp_A_precond'] = lx.MatrixLinearOperator(Ohess)
        stationarity_data['hess_l_k'] = hess_l_k
        stationarity_data['scale_AC'] = scale_AC
        stationarity_data['scale_BC'] = scale_BC
        stationarity_data['proj_BC'] = proj_BC
        stationarity_data['annil_BC'] = annil_BC
        stationarity_data['proj_C'] = proj_C
        stationarity_data['annil_C'] = annil_C

        if verbose > 0:
            hess_rank = jnp.linalg.matrix_rank(A + B + C)
            Ohess_rank = jnp.linalg.matrix_rank(OA + OB + OC)
            hess_cond = jnp.linalg.cond(A + B + C)
            Ohess_cond = jnp.linalg.cond(OA + OB + OC)
            stationarity_data['hess_rank'] = hess_rank
            stationarity_data['Ohess_rank'] = Ohess_rank
            stationarity_data['hess_cond'] = hess_cond
            stationarity_data['Ohess_cond'] = Ohess_cond
            debug.print(
                'Info on Hessian terms (unsorted)\n'
                '    Rank of term 1, 2 and 3: {a1}\n'
                '    Max sv of 1, 2 and 3: {a2}\n'
                'Info on Hessian terms (sorted)\n'
                '    Rank of A, B and C: {a}\n'
                '    Max sv of A, B and C: {aa}\n'
                '    Rank of OA, OB and OC: {b}\n'
                '    scale_AC and scale_BC: {bb}\n'
                '    Rank of proj_BC and annil_BC: {c}\n'
                '    Rank of proj_C  and annil_C:  {d}\n'
                '    Constrained Hessian rank and condition number, before pre-conditioning: {x}, {x1}\n'
                '    Constrained Hessian rank and condition number, after pre-conditioning:  {y}, {y1}',
                a1=jnp.linalg.matrix_rank(hess_l_k_terms_val),
                a2=s_max,
                a=jnp.sum(s_selection[hess_order], axis=1),
                aa=s_max[hess_order],
                b=(jnp.linalg.matrix_rank(OA), jnp.linalg.matrix_rank(OB), jnp.linalg.matrix_rank(OC)),
                bb=(scale_AC, scale_BC),
                c=(jnp.linalg.matrix_rank(proj_BC), jnp.linalg.matrix_rank(annil_BC)),
                d=(jnp.linalg.matrix_rank(proj_C), jnp.linalg.matrix_rank(annil_C)),
                x=hess_rank,
                y=Ohess_rank,
                x1=hess_cond,
                y1=Ohess_cond
            )

    stationarity_data['grad_y_l_k'] = jacrev(l_k, argnums=1)
    return stationarity_data


def adjoint_auglag_lbfgs_legacy(
    f_metric,
    stationarity_data,
    y_flat,
    implicit_linear_solver,
    verbose,
):
    r'''
    (Legacy) Compute the total derivative ``dfdy_arr`` of a single metric with respect
    to all problem parameters, using the augmented-Lagrangian stationarity
    condition precomputed by :func:`stationarity_auglag_lbfgs_legacy`.

    Parameters
    ----------
    f_metric : Callable
        ``(x_flat, y_flat) -> scalar``.  The metric to differentiate.
    stationarity_data : dict
        Output of :func:`stationarity_auglag_lbfgs_legacy`.
    y_flat : ndarray
        Flattened problem parameters.
    implicit_linear_solver : lineax.AbstractLinearSolver
        Linear solver for the VIHP system.
    verbose : int
        Verbosity level.

    Returns
    -------
    metric_value : scalar
        Value of ``f_metric`` at the optimum.
    dfdy_arr : ndarray, shape ``(ny,)``
        Total derivative of the metric w.r.t. all problem parameters.
    debug_info : dict
        Empty when ``verbose == 0``. Otherwise contains ``'vihp'`` and,
        for constrained problems, ``'hess_err'`` and ``'Ohess_err'``.
    '''
    constrained = stationarity_data['constrained']
    x_flat_precond = stationarity_data['x_flat_precond']
    grad_y_l_k = stationarity_data['grad_y_l_k']
    grad_y_l_k_for_hess = lambda x, y_flat=y_flat: grad_y_l_k(x, y_flat)

    grad_x_f = jacrev(f_metric, argnums=0)(x_flat_precond, y_flat)
    grad_y_f = jacrev(f_metric, argnums=1)(x_flat_precond, y_flat)

    debug_info = {}
    if not constrained:
        vihp_hess = stationarity_data['vihp_hess']
        vihp = lx.linear_solve(vihp_hess, grad_x_f).value
    else:
        scale_AC = stationarity_data['scale_AC']
        scale_BC = stationarity_data['scale_BC']
        proj_BC = stationarity_data['proj_BC']
        annil_BC = stationarity_data['annil_BC']
        proj_C = stationarity_data['proj_C']
        annil_C = stationarity_data['annil_C']
        vihp_A_raw = stationarity_data['vihp_A_raw']
        vihp_A_precond = stationarity_data['vihp_A_precond']
        hess_l_k = stationarity_data['hess_l_k']

        vihp_b = (
            scale_AC * annil_BC @ annil_C @ grad_x_f
            + scale_BC * proj_BC @ annil_C @ grad_x_f
            + proj_C @ grad_x_f
        )
        grad_x_f = jnp.nan_to_num(grad_x_f, nan=0.0, posinf=0.0, neginf=0.0)
        vihp_b = jnp.nan_to_num(vihp_b, nan=0.0, posinf=0.0, neginf=0.0)

        # Try preconditioned solve first; only compute raw solve if
        # the preconditioned residual is poor (optimisation #7).
        vihp_precond = lx.linear_solve(
            vihp_A_precond, vihp_b, solver=implicit_linear_solver
        ).value
        Ohess_err = jnp.linalg.norm(hess_l_k @ vihp_precond - grad_x_f)

        precond_good = Ohess_err < 1e-4 * jnp.linalg.norm(grad_x_f)
        def _raw_solve():
            return lx.linear_solve(
                vihp_A_raw, grad_x_f, solver=implicit_linear_solver
            ).value
        vihp_raw = jax.lax.cond(
            precond_good,
            lambda: vihp_precond,
            _raw_solve,
        )
        hess_err = jnp.where(
            precond_good,
            Ohess_err + 1.0,  # ensure precond wins when we skipped raw
            jnp.linalg.norm(hess_l_k @ vihp_raw - grad_x_f),
        )
        if verbose > 0:
            debug.print('Solve error (raw vs precond): {x}, {y}', x=hess_err, y=Ohess_err)
        vihp = jnp.where(hess_err < Ohess_err, vihp_raw, vihp_precond)
        debug_info['hess_err'] = hess_err
        debug_info['Ohess_err'] = Ohess_err

    # df/dy = -grad_x(f) @ H(l_k)^{-1} @ grad_x(grad_y(l_k)) + grad_y(f)
    _, dfdy1 = jvp(grad_y_l_k_for_hess, primals=[x_flat_precond], tangents=[vihp])
    dfdy_arr = -dfdy1 + grad_y_f
    metric_value = f_metric(x_flat_precond, y_flat)

    if verbose > 0:
        debug_info['vihp'] = vihp
        if constrained:
            debug_info['hess_rank'] = stationarity_data.get('hess_rank', jnp.nan)
            debug_info['Ohess_rank'] = stationarity_data.get('Ohess_rank', jnp.nan)
            debug_info['hess_cond'] = stationarity_data.get('hess_cond', jnp.nan)
            debug_info['Ohess_cond'] = stationarity_data.get('Ohess_cond', jnp.nan)

    return metric_value, dfdy_arr, debug_info


# ── KKT-based stationarity and adjoint ───────────────────────────────────

def _recover_multipliers(x_opt, y_flat, f_g_ineq_h_eq_from_y, unravel_y,
                         unravel_unscale_x):
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
    unravel_unscale_x : Callable
        ``(x_flat) -> dofs_dict``

    Returns
    -------
    z_ineq : ndarray, shape (n_g,)
        Recovered inequality multipliers.
    z_eq : ndarray, shape (n_h,)
        Recovered equality multipliers.
    '''
    f_obj, g_ineq, h_eq, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y_flat))
    f_scaled = lambda x: f_obj(unravel_unscale_x(x))
    g_scaled = lambda x: g_ineq(unravel_unscale_x(x))
    h_scaled = lambda x: h_eq(unravel_unscale_x(x))

    grad_f = grad(f_scaled)(x_opt)
    J_g = jacrev(g_scaled)(x_opt)          # (n_g, n_x)
    J_h = jacrev(h_scaled)(x_opt)          # (n_h, n_x)
    A_T = jnp.concatenate([J_g.T, J_h.T], axis=1)  # (n_x, n_g + n_h)

    z_all = jnp.linalg.lstsq(A_T, -grad_f)[0]

    n_g = J_g.shape[0]
    return z_all[:n_g], z_all[n_g:]


def stationarity_auglag_lbfgs(
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
    augmented-Lagrangian solution.

    Recovers dual variables via least-squares on the KKT stationarity
    condition, then delegates to the shared :func:`kkt_adjoint.stationarity_kkt`.

    Called once per ``quadcoil()`` invocation; the returned ``stationarity_data``
    dict is then passed into :func:`adjoint_auglag_lbfgs` once per metric.

    Parameters
    ----------
    constrained : bool
        Whether the problem has constraints.
    convex : bool
        Whether to assume the problem is convex.
    solve_results : dict
        Output of :func:`solve_constrained_auglag_lbfgs` or
        :func:`solve_unconstrained_auglag_lbfgs`.
    y_flat : ndarray
        Flattened problem parameters.
    f_g_ineq_h_eq_from_y : Callable
        ``(y_dict) -> (f_obj, g_ineq, h_eq, n_g, n_h, aux_dofs)``
    unravel_y : Callable
        ``(y_flat) -> y_dict``
    unravel_unscale_x : Callable
        ``(x_flat) -> dofs_dict``
    solver_options : dict
        Solver options (passed through; unused by the KKT path).
    verbose : int
        Verbosity level.

    Returns
    -------
    stationarity_data : dict
        Opaque state consumed by :func:`adjoint_auglag_lbfgs`.
    '''
    x_opt = solve_results['fin_x']

    if not constrained:
        return stationarity_kkt(
            constrained=False,
            convex=convex,
            x_opt=x_opt,
            z_opt=jnp.zeros(0, dtype=x_opt.dtype),
            y_flat=y_flat,
            f_g_ineq_h_eq_from_y=f_g_ineq_h_eq_from_y,
            unravel_y=unravel_y,
            unravel_unscale_x=unravel_unscale_x,
            verbose=verbose,
        )

    z_ineq, z_eq = _recover_multipliers(
        x_opt, y_flat, f_g_ineq_h_eq_from_y, unravel_y, unravel_unscale_x,
    )
    z_combined = jnp.concatenate([z_ineq, z_eq])

    def f_g_combined_from_y(y_dict):
        f_obj, g_ineq, h_eq, n_g, n_h, aux = f_g_ineq_h_eq_from_y(y_dict)
        g_combined = lambda dofs: jnp.concatenate([g_ineq(dofs), h_eq(dofs)])
        h_empty = lambda dofs: jnp.zeros(0)
        return f_obj, g_combined, h_empty, n_g + n_h, 0, aux

    return stationarity_kkt(
        constrained=True,
        convex=convex,
        x_opt=x_opt,
        z_opt=z_combined,
        y_flat=y_flat,
        f_g_ineq_h_eq_from_y=f_g_combined_from_y,
        unravel_y=unravel_y,
        unravel_unscale_x=unravel_unscale_x,
        verbose=verbose,
    )


def adjoint_auglag_lbfgs(
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
    f_metric : Callable
        ``(x_flat, y_flat) -> scalar``.
    stationarity_data : dict
        Output of :func:`stationarity_auglag_lbfgs`.
    y_flat : ndarray
        Flattened problem parameters.
    implicit_linear_solver : lineax.AbstractLinearSolver
        Linear solver for the KKT system.
    verbose : int
        Verbosity level.

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