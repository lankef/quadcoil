
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jax import jit, vmap
from jax.lax import while_loop

lstsq_vmap = vmap(jnp.linalg.lstsq)

def wl_debug(cond_fun, body_fun, init_val):
    val = init_val
    iter_num_wl = 1
    while cond_fun(val):
        val = body_fun(val)
    return val

run_opt_lbfgs = lambda init_params, fun, maxiter, ftol, xtol, gtol: \
    run_opt_optax(init_params, fun, maxiter, ftol, xtol, gtol, opt=optax.lbfgs())

# # Not robust. Does not have the up to date output signature. In here for backup purposes.
# def run_opt_bfgs(init_params, fun, maxiter, ftol, xtol, gtol): 
#     return jax.scipy.optimize.minimize(fun=fun, x0=init_params, method='BFGS', tol=gtol, options={'maxiter': maxiter,}).x

def run_opt_optax(init_params, fun, maxiter, ftol, xtol, gtol, opt):
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
        params, _, _, dx, du, df, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | (
            (iter_num < maxiter) 
            & (err >= gtol)
            & (dx >= xtol)
            & (du >= xtol)
            & (df >= ftol)
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

''' Constrained optimization '''

# A simple augmented Lagrangian implementation
# This jit flag is temporary, because we want 
# derivatives wrt f and g's contents too.
# @partial(jit, static_argnames=[
#     'f_obj',
#     'h_eq',
#     'g_ineq',
#     'run_opt',
#     'c_growth_rate',
#     'ftol_outer',
#     'ctol_outer',
#     'xtol_outer',
#     'gtol_outer',
#     'ftol_inner',
#     'xtol_inner',
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
        c_init=0.1,
        lam_init=jnp.zeros(1),
        h_eq=lambda x:jnp.zeros(1),
        mu_init=jnp.zeros(1),
        g_ineq=lambda x:jnp.zeros(1),
        c_growth_rate=1.1,
        ftol_outer=1e-7, # constraint tolerance
        ctol_outer=1e-7, # constraint tolerance
        xtol_outer=1e-7, # convergence rate tolerance
        gtol_outer=1e-7, # gradient tolerance
        ftol_inner=1e-7,
        xtol_inner=1e-7,
        gtol_inner=1e-7,
        maxiter_inner=1500,
        maxiter_outer=50,
        # # Uses jax.lax.scan instead of while_loop.
        # # Enables history and forward diff but disables 
        # # convergence test.
    ):
    '''
    Solves 
    min f(x)
    subject to 
    h(x) = 0, g(x) <= 0
    '''
    # Has shape n_cons_ineq
    gplus = lambda x, mu, c: jnp.max(jnp.array([g_ineq(x), -mu/c]), axis=0)

    # True when non-convergent.
    @jit
    def outer_convergence_criterion(dict_in):
        x_k = dict_in['x_k']
        dx_outer = dict_in['dx_outer']
        grad_l_k = dict_in['grad_l_k']
        niter_outer = dict_in['niter_outer']
        df_outer = dict_in['df_outer']
        dg_outer = dict_in['dg_outer']
        dh_outer = dict_in['dh_outer']
        # This is the convergence condition (True when not converged yet)
        return(
            (niter_outer == 0) | (
                # Only continue if max iter is not exceeded
                (niter_outer < maxiter_outer) 
                # Only continue if gradient is greater than tol
                & (grad_l_k >= gtol_outer)
                # Only continue if reduction in any of f, g, or h
                # is greater than tol.
                & (
                    (df_outer >= ftol_outer)
                    | (dg_outer >= ftol_outer)
                    | (dh_outer >= ftol_outer)
                )
                # Only continue if the iteration convergence 
                # rate is greater than tol
                & (dx_outer >= xtol_outer)
                # Only continue if all constraints are satisfied
                & (
                    jnp.any(h_eq(x_k) >= ctol_outer)
                    | jnp.any(h_eq(x_k) <= -ctol_outer)
                    | jnp.any(g_ineq(x_k) >= ctol_outer)
                )
            )
        )

    # Recursion
    @jit
    def body_fun_augmented_lagrangian(dict_in, x_dummy=None):
        x_km1 = dict_in['x_k']
        c_k = dict_in['c_k']
        lam_k = dict_in['lam_k']
        mu_k = dict_in['mu_k']
        f_km1 = dict_in['f_k']
        g_km1 = dict_in['g_k']
        h_km1 = dict_in['h_k']

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
        x_k, val_l_k, grad_l_k, niter_inner_k, dx_k, du_k, df_k = run_opt(x_km1, l_k, maxiter_inner, ftol_inner, xtol_inner, gtol_inner)

        lam_k_first_order = lam_k + c_k * h_eq(x_k)
        mu_k_first_order = mu_k + c_k * gplus(x_k, mu_k, c_k)
        
        f_k = f_obj(x_k)
        g_k = g_ineq(x_k)
        h_k = h_eq(x_k)
        dict_out = {
            'niter_outer': dict_in['niter_outer']+1,
            'dx_outer': jnp.linalg.norm(x_k - x_km1),
            'df_outer': jnp.max(jnp.abs(f_k - f_km1)),
            'dg_outer': jnp.max(jnp.abs(g_k - g_km1)),
            'dh_outer': jnp.max(jnp.abs(h_k - h_km1)),
            'f_k': f_k,
            'g_k': g_k,
            'h_k': h_k,
            'x_k': x_k,
            'val_l_k': val_l_k,
            'grad_l_k': jnp.linalg.norm(grad_l_k),
            'c_k': c_k * c_growth_rate,
            'lam_k': lam_k_first_order,
            'mu_k': mu_k_first_order,
            'niter_inner_k': niter_inner_k,
            'dx_k': dx_k,
            'du_k': du_k,
            'df_k': df_k,
        }
        return(dict_out)
    init_dict = {
        'niter_outer': 0,       
        # Changes in x between the kth and k-1th iteration
        'dx_outer': 0.,
        # Changes in f, g, h between the kth and k-1th iteration
        'df_outer': 0., 
        'dg_outer': 0.,
        'dh_outer': 0.,
        'f_k': f_obj(x_init), # Value of f, g, h after the kth iteration
        'g_k': g_ineq(x_init),
        'h_k': h_eq(x_init),
        'x_k': x_init,
        'val_l_k': 0.,
        'grad_l_k': 0.,
        'c_k': c_init,
        'lam_k': lam_init,
        'mu_k': mu_init,
        'niter_inner_k': 0,
        'dx_k': 0.,
        'du_k': 0.,
        'df_k': 0.,
    }
    result = while_loop(
        cond_fun=outer_convergence_criterion,
        body_fun=body_fun_augmented_lagrangian,
        init_val=init_dict,
    )
    return(result)# Changes in f, g, h between the kth and k-1th iteration

