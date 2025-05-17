import warnings
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jax import jit, vmap, grad, jacrev
import jax
from jax.lax import while_loop

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

def run_opt_lbfgs(init_params, fun, maxiter, fstop, xstop, gtol, verbose):
    r'''
    A wrapper for performing unconstrained optimization using ``optax.lbfgs``.
    
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
    return run_opt_optax(init_params, fun, maxiter, fstop, xstop, gtol, opt=optax.lbfgs(), verbose=verbose)


def run_opt_optax(init_params, fun, maxiter, fstop, xstop, gtol, opt, verbose):
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
    init_carry = (
        init_params, 
        jnp.zeros_like(init_params),
        0, jnp.zeros_like(init_params), jnp.zeros_like(init_params), 0,
        # 0., 0., 0., 0.,
        opt.init(init_params)
    )
    g0 = grad(fun)(init_params)
    g0_norm = jnp.linalg.norm(g0)
    g0_max = jnp.max(jnp.abs(g0))
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
    if verbose>1:
        jax.debug.print('INNER: starting gradient L2 norm: {a}', a=g0_norm)
    # Carry is params, update, value, dx, du, df, state1
    def step(carry):
        params1, updates1, value1, _, _, _, state1 = carry
        value2, grad2 = value_and_grad_fun(params1, state=state1)
        updates2, state2 = opt.update(
            grad2, state1, params1, value=value2, grad=grad2, value_fn=fun
        )
        params2 = optax.apply_updates(params1, updates2)
        return(
            params2, updates2, value2, 
            jnp.abs(params2 - params1), # jnp.linalg.norm(params2 - params1), 
            jnp.abs(updates2 - updates1), # jnp.linalg.norm(updates2 - updates1), 
            jnp.abs(value2 - value1), 
            # jnp.linalg.norm(delta_normalized(params2, params1)), 
            # jnp.linalg.norm(delta_normalized(updates2, updates1)), 
            # delta_normalized(value2, value1), 
            state2
        )
  
    def continuing_criterion(carry):
        params, _, value, dx, du, df, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        # DEBUG 
        param2 = dx + params
        dx1 = param2 - params
        dx_norm = jnp.linalg.norm(dx)
        du_norm = jnp.linalg.norm(du)
        params_norm = jnp.linalg.norm(params)
        if verbose>2:
            jax.debug.print(
                'INNER: L: {l}, dx: {dx}, du: {du}, df: {df}, \n'\
                '    dx - (dx+x-x): {dxn},\n'\
                '    grad:{g}, grad/g0:{gnorm}\n'\
                '    Stopping criteria - (err > gtol): {a},(dx > xstop or du > xstop): {b},(df > fstop): {c}',
                a=(err > gtol), #  * g0_norm) # TOLERANCE SCALING! MAY NEED CHANGING!
                b=((dx_norm > xstop * params_norm) | (du_norm > xstop * params_norm)),
                c=(df > fstop),
                l=value,
                dx=jnp.max(dx),
                dxn=jnp.linalg.norm(dx - dx1),
                du=jnp.max(du),
                df=df,
                g=err,
                gnorm=err/g0_norm,
            )
        return (iter_num == 0) | (
            (iter_num < maxiter) 
            & (err > gtol) 
            & ((dx_norm > xstop * params_norm) | (du_norm > xstop * params_norm))
            & (df > fstop * value) 
        )
    final_params, final_updates, final_value, final_dx, final_du, final_df, final_state = while_loop(
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

def solve_constrained(
        x_init,
        # x_unit_init,
        f_obj,
        run_opt=run_opt_lbfgs,
        # No constraints by default
        c_init=1.,
        c_growth_rate=1.1,
        lam_init=jnp.zeros(0),
        h_eq=lambda x:jnp.zeros(0),
        mu_init=jnp.zeros(0),
        g_ineq=lambda x:jnp.zeros(0),
        xstop_outer=1e-7, # convergence rate tolerance
        # gtol_outer=1e-7, # gradient tolerance
        ctol_outer=1e-7, # constraint tolerance, used in multiplier update
        fstop_inner=1e-7,
        xstop_inner=1e-7,
        gtol_inner=1e-7,
        fstop_inner_last=1e-7,
        xstop_inner_last=1e-7,
        gtol_inner_last=1e-7,
        maxiter_tot=10000,
        maxiter_inner=500,
        # # Uses jax.lax.scan instead of while_loop.
        # # Enables history and forward diff but disables 
        # # convergence test.
        verbose=0,
        c_k_safe=1e9
    ):
    r'''
    Solves the constrained optimization problem:

    .. math::

        \min_x f(x) \\
        \text{subject to } \\
        h(x) = 0, \\
        g(x) \leq 0 \\
        
    Using the augmented Lagrangian method in 
    *Constrained Optimization and Lagrange Multiplier Methods* Chapter 3.
    Please refer to the chapter for notation.
    
    Parameters
    ----------  
    init_params : ndarray, shape (N,)
    fun : Callable
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
    
    x_init : ndarray, shape (Nx,)
        The initial condition.
    x_unit_init : ndarray, shape (Nx,)
        The initial x scale. This scaling factor ensures that x~1. Will be updated after every outer iteration.
    f_obj : Callable
        The objective function.
    run_opt : Callable, optional, default=run_opt_lbfgs
        The optimizer choice. Must be a wrapper with the 
        same signature as ``run_opt_lbfgs``.
    c_init : float, optional, default=1.
        The initial :math:`c` factor. Please see 
        *Constrained Optimization and Lagrange Multiplier Methods* 
        Chapter 3. 
    c_growth_rate : float, optional, default=1.1,
        The growth rate of the :math:`c` factor.
    lam_init : ndarray, shape (Nh), optional, default=jnp.zeros(1),
        The initial :math:`\lambda` multiplier for equality constraints.
        No constraints by default.
    h_eq : Callable, optional, default=lambda x:jnp.zeros(1),
        The equality constraint function. 
        Must map ``x`` to an ``ndarray`` with shape ``(Nh)``.
        No constraints by default.
    mu_init : ndarray, shape (Ng), optional, default=jnp.zeros(1),
        The initial :math:`\mu` multiplier for inequality constraints.
        No constraints by default.
    g_ineq : Callable, optional, default=lambda x:jnp.zeros(1),
        The equality constraint function. 
        Must map ``x`` to an ``ndarray`` with shape ``(Ng)``.
        No constraints by default.
    xstop_outer : float, optional, default=1e-7
        (Traced) ``x`` convergence rate of the outer augmented 
        Lagrangian loop. Terminates when ``dx`` falls below this. 
    gtol_outer : float, optional, default=1e-7
        (Traced) Tolerance of the :math:`\nabla L` KKT condition in
        the outer augmented Lagrangian loop. 
    ctol_outer : float, optional, default=1e-7
        (Traced) Tolerance of the constraint KKT conditions in the outer
        Lagrangian loop. 
    fstop_inner : float, optional, default=1e-7
        (Traced) ``f`` convergence rate of the inner LBFGS 
        Lagrangian loop. Terminates when ``df`` falls below this. 
    xstop_inner : float, optional, default=0
        (Traced) ``x`` convergence rate of the outer augmented 
        Lagrangian loop. Terminates when ``dx`` falls below this. 
    gtol_inner : float, optional, default=1e-7
        (Traced) Gradient tolerance of the inner LBFGS 
        iteration. Terminates when is satisfied. 
    maxiter_tot: int, optional, default=50
        (Static) The maximum of the outer iteration.
    verbose: int, optional, default=0
        (Static) The verbosity. When >1, outputs outer iteration convergence info.

    Returns
    -------
    status : dict
        The end state of the iteration. Contains the following entries:

        .. code-block:: python
        
            init_dict = {
                'tot_niter' : int, # The outer iteration number
                'outer_dx' : float, # The L2 norm of the change in x between the last 2 outer iterations
                'inner_fin_f' : float, # The value of f at the optimum
                'inner_fin_g' : ndarray, # The value of g at the optimum
                'inner_fin_h' : ndarray, # The value of h at the optimum
                'inner_fin_x' : ndarray, # The optimum
                'inner_fin_l_aug' : float, # The value of the augmented Lagrangian objective l_k at the optimum 
                'grad_l_k' : ndarray, # The gradient of the augmented Lagrangian objective l_k at the optimum 
                'inner_fin_c' : float, # The final value of c
                'inner_fin_lam' : ndarray, # The final value of lambda
                'inner_fin_mu' : ndarray, # The final value of mu
                'inner_fin_niter' : int, # The number of L-BFGS iterations in the last step
                'inner_fin_dx_scaled' : float, # The L2 norm of the change in x between the last 2 inner L-BFGS iteration
                'inner_fin_du' : float, # The L2 norm of the change in update between the last 2 inner L-BFGS iteration
                'inner_fin_dl' : float, # The L2 norm of the change in f between the last 2 inner L-BFGS iteration
            }
    '''
    # Has shape n_cons_ineq
    gplus = lambda x, mu, c: jnp.max(jnp.array([g_ineq(x), -mu/c]), axis=0)
    grad_f = grad(f_obj)
    grad_g = jacrev(g_ineq)
    grad_h = jacrev(h_eq)
    # True when non-convergent.
    # @jit
    def outer_convergence_criterion(dict_in):
        x_k = dict_in['inner_fin_x']
        x_norm = jnp.linalg.norm(x_k)
        # lam_k = dict_in['inner_fin_lam']
        # mu_k = dict_in['inner_fin_mu']
        # grad_l = dict_in['outer_grad_l']
        outer_dx = dict_in['outer_dx']
        tot_niter = dict_in['tot_niter']
        # outer_dgrad_l = dict_in['outer_dgrad_l']
        # outer_dg = dict_in['outer_dg']
        # outer_dh = dict_in['outer_dh']
        # f_k = dict_in['inner_fin_f']
        # This is the convergence condition (True when not converged yet)
        return(
            (tot_niter == 0) | (
                # Stop if max iter is exceeded
                (tot_niter < maxiter_tot) 
                # Only stop if reduction in all of f, g, or h
                # falls below the criterion
                # & (
                #     (outer_dgrad_l >= grad_l_stop_outer)
                #     # | (outer_dg >= grad_l_stop_outer)
                #     # | (outer_dh >= grad_l_stop_outer)
                # )
                # Stop if the iteration convergence 
                # rate falls below the stopping criterion
                & (outer_dx >= xstop_outer * x_norm)
                # KKT criterion, P321 of Nocedal
                # & (
                #     KKT1 
                #     | KKT2 
                #     | KKT3 
                #     | KKT4 
                #     | KKT5 
                #     | KKT6
                # )
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
        x_km1 = dict_in['inner_fin_x']
        c_k = dict_in['inner_fin_c']
        lam_k = dict_in['inner_fin_lam']
        mu_k = dict_in['inner_fin_mu']
        f_km1 = dict_in['inner_fin_f']
        g_km1 = dict_in['inner_fin_g']
        h_km1 = dict_in['inner_fin_h']
        # x_unit = dict_in['x_unit']
        # grad_l_val_km1 = dict_in['outer_grad_l']
        # Eq (10) on p160 of Constrained Optimization and Multiplier Method
        l_k = lambda x: (
            f_obj(x) 
            + lam_k@h_eq(x) 
            + mu_k@gplus(x, mu_k, c_k)
            + c_k/2 * (
                jnp.sum(h_eq(x)**2) 
                + jnp.sum(gplus(x, mu_k, c_k)**2)
            )
        ) 
        # Solving a stage of the problem
        x_k, val_l_k, grad_l_k, niter_inner_k, dx_k, du_k, dL_k = run_opt(
            x_km1, l_k, maxiter_inner, 
            fstop_inner, xstop_inner, gtol_inner,
            verbose
        )
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
        c_k_new = jnp.where(update_multiplier, c_k,               c_k * c_growth_rate) 
        lam_k =   lam_k + c_k * h_k # jnp.where(update_multiplier, lam_k + c_k * h_k, lam_k              )
        mu_k =    mu_k + c_k * gp_k # jnp.where(update_multiplier, mu_k + c_k * gp_k, mu_k               )
        # Calculating the gradient of the 
        # Actual lagrangian: 
        #   grad_x L
        # = grad_x f - lam @ grad_x h_eq + mu @ grad_x g_ineq_active
        # grad_f_val = grad_f(x_k)
        # g_active = jnp.where(g_k >= 0., 1., 0.)

        # grad_l_val = (
        #     grad_f_val 
        #     - lam_k @ grad_h(x_k)
        #     - (g_active * mu_k) @ grad_g(x_k)
        # )

        # If any of the following are True, KKT is NOT satisfied
        # Nocedal pg321, 12.34
        # lam_k_active = jnp.where(jnp.abs(h_k) >= ctol_outer, lam_k, 0)
        # mu_k_active = jnp.where(g_k >= ctol_outer, mu_k, 0)
        # d_grad_l_val = jnp.linalg.norm(grad_l_val - grad_l_val_km1)
        # KKT1 = jnp.linalg.norm(grad_l_val) >= gtol_outer # Nocedal 12.34a
        # KKT2 = jnp.any(jnp.abs(h_k) >= ctol_outer)       # Nocedal 12.34b
        # KKT3 = jnp.any(g_k >= ctol_outer)                # Nocedal 12.34c
        # KKT4 = jnp.any(mu_k <= -ctol_outer)              # Nocedal 12.34d
        # KKT5 = jnp.abs(lam_k @ h_k) >= ctol_outer        # Nocedal 12.34e
        # KKT6 = jnp.abs(mu_k @ g_k) >= ctol_outer         # Nocedal 12.34e
        if verbose>1:
            jax.debug.print(
                'OUTER: \n'\
                '    Iteration: {tot_niter}/{maxiter_tot}\n'\
                '        f  : {f}\n'\
                '        g  : {gmin}, {gmax}\n'\
                '        g+ : {gpmin}, {gpmax}\n'\
                '        h  : {hmin}, {hmax}\n'\
                '   |grad f|: {xx}\n'\
                '   |grad g|: {xg}\n'\
                '   |grad h|: {xh}\n'\
                '        mu : {mu1}, {mu2}\n'\
                '        lam: {lam1}, {lam2}\n'\
                '    Stopping criteria (False = satisfied)\n'\
                '    Stopping criterion 2: {b}\n'\
                '        outer_dx    = {outer_dx}\n'\
                '        xstop_outer = {xstop_outer}\n'\
                # '    grad_l_val: {x}, d_grad_l_val: {dx}\n'\
                '    inner iter #: {z}\n'\
                '    c_k: {c_k}',
                # ga=jnp.linalg.norm(grad_f_val),
                # gb=jnp.linalg.norm(lam_k @ grad_h(x_k)),
                # gc=jnp.linalg.norm((g_active * mu_k) @ grad_g(x_k)),
                # gtot=jnp.linalg.norm(grad_l_val),
                # gtol_outer=gtol_outer,
                # ctol_outer=ctol_outer,
                f=f_k,
                gmin=_print_min_blank(g_k),
                gmax=_print_max_blank(g_k),
                gpmin=_print_min_blank(gp_k),
                gpmax=_print_max_blank(gp_k),
                hmin=_print_min_blank(h_k),
                hmax=_print_max_blank(h_k),
                c_k=c_k,
                # x=jnp.linalg.norm(grad_l_val),
                mu1=_print_min_blank(mu_k),
                mu2=_print_max_blank(mu_k),
                lam1=_print_min_blank(lam_k),
                lam2=_print_max_blank(lam_k),
                # dx=jnp.linalg.norm(d_grad_l_val),
                xx=jnp.linalg.norm(grad_f(x_k)),
                xg=jnp.linalg.norm(grad_g(x_k)),
                xh=jnp.linalg.norm(grad_h(x_k)),
                z=niter_inner_k,
                tot_niter=dict_in['tot_niter']+niter_inner_k,
                maxiter_tot=maxiter_tot,
                outer_dx=jnp.linalg.norm(x_k - x_km1),
                xstop_outer=xstop_outer * jnp.linalg.norm(x_k),
                b=(jnp.linalg.norm(x_k - x_km1) >= xstop_outer),

            )
        dict_out = {
            'tot_niter': dict_in['tot_niter']+niter_inner_k,
            'outer_dx': jnp.linalg.norm(x_k - x_km1),
            # 'outer_dgrad_l': d_grad_l_val,
            # 'outer_dg': jnp.max(jnp.abs(g_k - g_km1)),
            # 'outer_dh': jnp.max(jnp.abs(h_k - h_km1)),
            # 'outer_grad_l': grad_l_val,
            'inner_fin_f': f_k,
            'inner_fin_g': g_k,
            'inner_fin_h': h_k,
            'inner_fin_x': x_k,
            'inner_fin_l_aug': val_l_k,
            'inner_fin_grad_l_aug': jnp.linalg.norm(grad_l_k),
            # 'inner_fin_grad_f': jnp.linalg.norm(grad_f_val),
            'inner_fin_c': c_k_new,
            'inner_fin_lam': lam_k,
            'inner_fin_mu': mu_k,
            'inner_fin_niter': niter_inner_k,
            'inner_fin_dx_scaled': dx_k,
            'inner_fin_du': du_k,
            'inner_fin_dl': dL_k,
            # The scaling factor for the next iteration
            # 'x_unit': jnp.average(jnp.abs(x_k)),
        }
        return(dict_out)
    init_dict = {
        'tot_niter': 0,       
        # Changes in x between the kth and k-1th iteration
        'outer_dx': 0.,
        # Changes in f, g, h between the kth and k-1th iteration
        # 'outer_dgrad_l': 0., 
        # 'outer_dg': 0.,
        # 'outer_dh': 0.,
        # 'outer_grad_l': jnp.zeros_like(x_init),
        'inner_fin_f': f_obj(x_init), # Value of f, g, h after the kth iteration
        'inner_fin_g': g_ineq(x_init),
        'inner_fin_h': h_eq(x_init),
        # 'x_unit': x_unit_init,
        'inner_fin_x': x_init,
        'inner_fin_l_aug': 0.,
        'inner_fin_grad_l_aug': 0.,
        # 'KKT1': True,
        # 'KKT2': True,
        # 'KKT3': True,
        # 'KKT4': True,
        # 'KKT5': True,
        # 'KKT6': True,
        # 'inner_fin_grad_f': 0.,
        'inner_fin_c': c_init,
        'inner_fin_lam': lam_init,
        'inner_fin_mu': mu_init,
        'inner_fin_niter': 0,
        'inner_fin_dx_scaled': 0.,
        'inner_fin_du': 0.,
        'inner_fin_dl': 0.,
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