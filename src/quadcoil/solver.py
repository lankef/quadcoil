import warnings
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jax import jit, vmap, grad
from jax.lax import while_loop

lstsq_vmap = vmap(jnp.linalg.lstsq)

# def wl_debug(cond_fun, body_fun, init_val):
#     val = init_val
#     iter_num_wl = 1
#     while cond_fun(val):
#         val = body_fun(val)
#     return val

def run_opt_lbfgs(init_params, fun, maxiter, fstop, xstop, gtol):
    r'''
    A wrapper for performing unconstrained optimization using ``optax.lbfgs``.
    
    Parameters
    ----------  
    init_params : ndarray, shape (N,)
        The initial condition.
    fun : callable
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
    return run_opt_optax(init_params, fun, maxiter, fstop, xstop, gtol, opt=optax.lbfgs())

# # Not robust. Does not have the up to date output signature. In here for backup purposes.
# def run_opt_bfgs(init_params, fun, maxiter, fstop, xstop, gtol): 
#     return jax.scipy.optimize.minimize(fun=fun, x0=init_params, method='BFGS', tol=gtol, options={'maxiter': maxiter,}).x

def run_opt_optax(init_params, fun, maxiter, fstop, xstop, gtol, opt):
    r'''
    A wrapper for performing unconstrained optimization using ``optax.base.GradientTransformationExtraArgs``.
    
    Parameters
    ----------  
    init_params : ndarray, shape (N,)
        The initial condition.
    fun : callable
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
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
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
            jnp.linalg.norm(params2 - params1), 
            jnp.linalg.norm(updates2 - updates1), 
            jnp.abs(value2 - value1), 
            state2
        )
  
    def continuing_criterion(carry):
        params, _, value, dx, du, df, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | (
            (iter_num < maxiter) 
            & (err >= gtol)
            & (dx >= xstop)
            & (du >= xstop)
            & (df/jnp.abs(value) >= fstop)
        )
    init_carry = (
        init_params, 
        jnp.zeros_like(init_params),
        0., 0., 0., 0.,
        opt.init(init_params)
    )
    final_params, final_updates, final_value, final_dx, final_du, final_df, final_state = while_loop(
        continuing_criterion, step, init_carry
    )
    return(
        final_params, 
        final_value,
        otu.tree_get(final_state, 'grad'), 
        otu.tree_get(final_state, 'count'),
        final_dx, # Changes in x
        final_du, # Changes in u
        final_df, # Changes in f
    )

# ''' Constrained optimization '''

# A simple augmented Lagrangian implementation
# This jit flag is temporary, because we want 
# derivatives wrt f and g's contents too.
# @partial(jit, static_argnames=[
#     'f_obj',
#     'h_eq',
#     'g_ineq',
#     'run_opt',
#     'c_growth_rate',
#     'fstop_outer',
#     'ctol_outer',
#     'xstop_outer',
#     'gtol_outer',
#     'fstop_inner',
#     'xstop_inner',
#     'gtol_inner',
#     'maxiter_inner',
#     'maxiter_outer',
#     # 'scan_mode',
# ])
def solve_constrained(
        x_init,
        f_obj,
        run_opt=run_opt_lbfgs,
        # No constraints by default
        c_init=1.,
        c_growth_rate=1.1,
        lam_init=jnp.zeros(1),
        h_eq=lambda x:jnp.zeros(1),
        mu_init=jnp.zeros(1),
        g_ineq=lambda x:jnp.zeros(1),
        fstop_outer=1e-7, # constraint tolerance
        xstop_outer=1e-7, # convergence rate tolerance
        gtol_outer=1e-7, # gradient tolerance
        ctol_outer=1e-7, # constraint tolerance
        fstop_inner=1e-7,
        xstop_inner=1e-7,
        gtol_inner=1e-7,
        maxiter_inner=1500,
        maxiter_outer=50,
        # # Uses jax.lax.scan instead of while_loop.
        # # Enables history and forward diff but disables 
        # # convergence test.
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
    fun : callable
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
    f_obj : callable
        The objective function.
    run_opt : callable, optional, default=run_opt_lbfgs
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
    h_eq : callable, optional, default=lambda x:jnp.zeros(1),
        The equality constraint function. 
        Must map ``x`` to an ``ndarray`` with shape ``(Nh)``.
        No constraints by default.
    mu_init : ndarray, shape (Ng), optional, default=jnp.zeros(1),
        The initial :math:`\mu` multiplier for inequality constraints.
        No constraints by default.
    g_ineq : callable, optional, default=lambda x:jnp.zeros(1),
        The equality constraint function. 
        Must map ``x`` to an ``ndarray`` with shape ``(Ng)``.
        No constraints by default.
    fstop_outer : float, optional, default=1e-7
        (Traced) ``f`` convergence rate of the outer augmented 
        Lagrangian loop. Terminates when ``df`` falls below this. 
    xstop_outer : float, optional, default=1e-7
        (Traced) ``x`` convergence rate of the outer augmented 
        Lagrangian loop. Terminates when ``dx`` falls below this. 
    gtol_outer : float, optional, default=1e-7
        (Traced) Gradient tolerance of the outer augmented 
        Lagrangian loop. Terminates when both tols are satisfied. 
    ctol_outer : float, optional, default=1e-7
        (Traced) Constraint tolerance of the outer augmented 
        Lagrangian loop. Terminates when both tols are satisfied. 
    fstop_inner : float, optional, default=1e-7
        (Traced) ``f`` convergence rate of the inner LBFGS 
        Lagrangian loop. Terminates when ``df`` falls below this. 
    xstop_inner : float, optional, default=0
        (Traced) ``x`` convergence rate of the outer augmented 
        Lagrangian loop. Terminates when ``dx`` falls below this. 
    gtol_inner : float, optional, default=1e-7
        (Traced) Gradient tolerance of the inner LBFGS 
        iteration. Terminates when is satisfied. 
    maxiter_outer: int, optional, default=50
        (Static) The maximum of the outer iteration.
    maxiter_inner: int, optional, default=1500
        (Static) The maximum of the inner iteration.

    Returns
    -------
    status : dict
        The end state of the iteration. Contains the following entries:

        .. code-block:: python
        
            init_dict = {
                'outer_niter' : int, # The outer iteration number
                'outer_dx' : float, # The L2 norm of the change in x between the last 2 outer iterations
                'outer_df' : float, # The L2 norm of the change in f between the last 2 outer iterations
                'outer_dg' : float, # The L2 norm of the change in g between the last 2 outer iterations
                'outer_dh' : float, # The L2 norm of the change in h between the last 2 outer iterations
                'inner_fin_f' : float, # The value of f at the optimum
                'inner_fin_g' : ndarray, # The value of g at the optimum
                'inner_fin_h' : ndarray, # The value of h at the optimum
                'inner_fin_x' : ndarray, # The optimum
                'inner_fin_l' : float, # The value of the augmented Lagrangian objective l_k at the optimum 
                'grad_l_k' : ndarray, # The gradient of the augmented Lagrangian objective l_k at the optimum 
                'inner_fin_c' : float, # The final value of c
                'inner_fin_lam' : ndarray, # The final value of lambda
                'inner_fin_mu' : ndarray, # The final value of mu
                'inner_fin_niter' : int, # The number of L-BFGS iterations in the last step
                'inner_fin_dx' : float, # The L2 norm of the change in x between the last 2 inner L-BFGS iteration
                'inner_fin_du' : float, # The L2 norm of the change in update between the last 2 inner L-BFGS iteration
                'inner_fin_dl' : float, # The L2 norm of the change in f between the last 2 inner L-BFGS iteration
            }
    '''
    # Has shape n_cons_ineq
    gplus = lambda x, mu, c: jnp.max(jnp.array([g_ineq(x), -mu/c]), axis=0)
    grad_f = grad(f_obj)
    # True when non-convergent.
    # @jit
    def outer_convergence_criterion(dict_in):
        x_k = dict_in['inner_fin_x']
        dx_outer = dict_in['outer_dx']
        grad_f_k = dict_in['inner_fin_grad_f']
        niter_outer = dict_in['outer_niter']
        df_outer = dict_in['outer_df']
        dg_outer = dict_in['outer_dg']
        dh_outer = dict_in['outer_dh']
        f_k = dict_in['inner_fin_f']
        g_k_mag = jnp.max(jnp.abs(dict_in['inner_fin_g']))
        h_k_mag = jnp.max(jnp.abs(dict_in['inner_fin_h']))
        # This is the convergence condition (True when not converged yet)
        return(
            (niter_outer == 0) | (
                # Stop if max iter is exceeded
                (niter_outer < maxiter_outer) 
                # Only stop if reduction in all of f, g, or h
                # falls below the criterion
                & (
                    (jnp.where(f_k>0, df_outer/jnp.abs(f_k), 0) >= fstop_outer)
                    | (jnp.where(g_k_mag>0, dg_outer/g_k_mag, 0) >= fstop_outer)
                    | (jnp.where(h_k_mag>0, dh_outer/h_k_mag, 0) >= fstop_outer)
                )
                # Stop if the iteration convergence 
                # rate falls below the stopping criterion
                & (dx_outer >= xstop_outer)
                # If no other stopping criteria are triggered,
                # only stop when both gradient and tolerance 
                # tlerance are reached.
                & (
                    (grad_f_k >= gtol_outer)
                    | jnp.any(h_eq(x_k) >= ctol_outer)
                    | jnp.any(h_eq(x_k) <= -ctol_outer)
                    | jnp.any(g_ineq(x_k) >= ctol_outer)
                )
            )
        )

    # Recursion
    # @jit
    def body_fun_augmented_lagrangian(dict_in, x_dummy=None):
        x_km1 = dict_in['inner_fin_x']
        c_k = dict_in['inner_fin_c']
        lam_k = dict_in['inner_fin_lam']
        mu_k = dict_in['inner_fin_mu']
        f_km1 = dict_in['inner_fin_f']
        g_km1 = dict_in['inner_fin_g']
        h_km1 = dict_in['inner_fin_h']

        l_k = lambda x: (
            f_obj(x) 
            + lam_k@h_eq(x) 
            + c_k/2 * (
                jnp.sum(h_eq(x)**2) 
                + jnp.sum(gplus(x, mu_k, c_k)**2)
            )
        ) 
        # Eq (10) on p160 of Constrained Optimization and Multiplier Method
        # Solving a stage of the problem
        
        # x, count, dx, du, df,
        x_k, val_l_k, grad_l_k, niter_inner_k, dx_k, du_k, dL_k = run_opt(x_km1, l_k, maxiter_inner, fstop_inner, xstop_inner, gtol_inner)

        lam_k_first_order = lam_k + c_k * h_eq(x_k)
        mu_k_first_order = mu_k + c_k * gplus(x_k, mu_k, c_k)
        
        f_k = f_obj(x_k)
        g_k = g_ineq(x_k)
        h_k = h_eq(x_k)
        dict_out = {
            'outer_niter': dict_in['outer_niter']+1,
            'outer_dx': jnp.linalg.norm(x_k - x_km1),
            'outer_df': jnp.max(jnp.abs(f_k - f_km1)),
            'outer_dg': jnp.max(jnp.abs(g_k - g_km1)),
            'outer_dh': jnp.max(jnp.abs(h_k - h_km1)),
            'inner_fin_f': f_k,
            'inner_fin_g': g_k,
            'inner_fin_h': h_k,
            'inner_fin_x': x_k,
            'inner_fin_l': val_l_k,
            'inner_fin_grad_f': jnp.linalg.norm(grad_f(x_k)),
            'inner_fin_c': c_k * c_growth_rate,
            'inner_fin_lam': lam_k_first_order,
            'inner_fin_mu': mu_k_first_order,
            'inner_fin_niter': niter_inner_k,
            'inner_fin_dx': dx_k,
            'inner_fin_du': du_k,
            'inner_fin_dl': dL_k,
        }
        return(dict_out)
    init_dict = {
        'outer_niter': 0,       
        # Changes in x between the kth and k-1th iteration
        'outer_dx': 0.,
        # Changes in f, g, h between the kth and k-1th iteration
        'outer_df': 0., 
        'outer_dg': 0.,
        'outer_dh': 0.,
        'inner_fin_f': f_obj(x_init), # Value of f, g, h after the kth iteration
        'inner_fin_g': g_ineq(x_init),
        'inner_fin_h': h_eq(x_init),
        'inner_fin_x': x_init,
        'inner_fin_l': 0.,
        'inner_fin_grad_f': 0.,
        'inner_fin_c': c_init,
        'inner_fin_lam': lam_init,
        'inner_fin_mu': mu_init,
        'inner_fin_niter': 0,
        'inner_fin_dx': 0.,
        'inner_fin_du': 0.,
        'inner_fin_dl': 0.,
    }
    result = while_loop(
        cond_fun=outer_convergence_criterion,
        body_fun=body_fun_augmented_lagrangian,
        init_val=init_dict,
    )
    return(result)# Changes in f, g, h between the kth and k-1th iteration

