from quadcoil import (
    parse_objectives, parse_constraints, get_objective,
    gen_winding_surface_atan, 
    SurfaceRZFourierJAX, QuadcoilParams, 
    solve_constrained
)
from functools import partial
from jax import jacrev, jvp, jit
import jax
import jax.numpy as jnp
import numpy as np

def is_ndarray(arr, n=1):
    return isinstance(arr, (np.ndarray, jnp.ndarray)) and arr.ndim == 1

# The list of all static arguments of 
# quadcoil. Also used in the DESC interface.
# All other vars are assumed traced. If you
# would like to add new static options, registering 
# them here will also register them in the DESC _Objective.
QUADCOIL_STATIC_ARGNAMES=[
    'nfp',
    'stellsym',
    'mpol',
    'ntor',
    # - Plasma options
    'plasma_mpol',
    'plasma_ntor',
    # - WS options
    'winding_surface_generator',
    'winding_mpol',
    'winding_ntor',
    # - Objectives
    'objective_name',
    # - Constraints 
    'constraint_name',
    'constraint_type',
    # - Solver options
    'metric_name',
    'maxiter_tot',
    'value_only',
    'verbose',
]

@partial(jit, static_argnames=QUADCOIL_STATIC_ARGNAMES)
def quadcoil(
    nfp:int,
    stellsym:bool,
    plasma_mpol:int,
    plasma_ntor:int,
    plasma_dofs,
    net_poloidal_current_amperes:float,
    
    # -- Defaults --
    
    # - Quadcoil parameters
    net_toroidal_current_amperes:float=0.,
    mpol:int=6,
    ntor:int=4,
    # Quadpoints to evaluate objectives at
    quadpoints_phi=None,
    quadpoints_theta=None,
    x_init=None, 
    # Current potential's normalization constant. 
    # By default will be generated from net total current.
    cp_mn_unit=None,
    
    # - Plasma parameters
    plasma_quadpoints_phi=None,
    plasma_quadpoints_theta=None,
    Bnormal_plasma=None,

    # - Winding parameters (offset)
    plasma_coil_distance:float=None,
    winding_surface_generator=gen_winding_surface_atan,

    # - Winding parameters (Providing surface)
    winding_dofs=None,
    winding_mpol:int=6,
    winding_ntor:int=5,
    winding_quadpoints_phi=None,
    winding_quadpoints_theta=None,

    # - Problem setup
    # Quadcoil objective terms, weights, and units
    # objective_unit differ in that they are not differentiated wrt.
    # They also exist to aid readability.
    objective_name='f_B',
    objective_weight=None,
    objective_unit=None,
    # - Quadcoil constraints
    constraint_name=(),
    constraint_type=(),
    constraint_unit=(),
    constraint_value=jnp.array([]),
    # - Metrics to study
    metric_name=('f_B', 'f_K'),

    # - Solver options
    c_init:float=1.,
    c_growth_rate:float=1.2,
    fstop_outer:float=1e-7, # convergence rate tolerance
    xstop_outer:float=1e-7, # convergence rate tolerance
    gtol_outer:float=1e-7, # gradient tolerance
    ctol_outer:float=1e-7, # constraint tolerance
    fstop_inner:float=1e-7,
    xstop_inner:float=0.,
    gtol_inner:float=1e-7,
    maxiter_tot:int=10000,
    maxiter_inner:int=500,
    value_only=False,
    verbose=False,
):
    r'''
    Solves a QUADCOIL problem.

    Parameters
    ----------
    nfp : int
        (Static) The number of field periods.
    stellsym : bool
        (Static) Stellarator symmetry.
    plasma_mpol : int
        (Static) The number of poloidal Fourier harmonics in the plasma boundary.
    plasma_ntor : int
        (Static) The number of toroidal Fourier harmonics in the plasma boundary.
    plasma_dofs : ndarray
        (Static) The plasma surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` convention.
    net_poloidal_current_amperes : float
        (Traced) The net poloidal current :math:`G`.
    net_toroidal_current_amperes : float, optional, default=0
        (Traced) The net toroidal current :math:`I`.
    mpol : int, optional, default=6
        (Static) The number of poloidal Fourier harmonics in the current potential :math:`\Phi_{sv}`.
    ntor : int, optional, default=4
        (Static) The number of toroidal Fourier harmonics in :math:`\Phi_{sv}`.
    quadpoints_phi : ndarray, shape (nphi,), optional, default=None
        (Traced) The poloidal quadrature points on the winding surface to evaluate the objectives at.
        Uses one period from the winding surface by default.
    quadpoints_theta : ndarray, shape (ntheta,), optional, default=None
        (Traced) The toroidal quadrature points on the winding surface to evaluate the objectives at.
        Uses one period from the winding surface by default.
    x_init : ndarray, optional, default=None
        (Traced) The initial guess. All zeros by default.
    cp_mn_unit : float, optional, default=None
        (Traced) Current potential's normalization constant.
        By default will be generated from total net current.
    plasma_quadpoints_phi : ndarray, shape (nphi_plasma,), optional, default=None
        (Traced) Will be set to ``jnp.linspace(0, 1/nfp, 32, endpoint=False)`` by default.
    plasma_quadpoints_theta : ndarray, shape (ntheta_plasma,), optional, default=None
        (Traced) Will be set to ``jnp.linspace(0, 1, 34, endpoint=False)`` by default.
    Bnormal_plasma : ndarray, shape (nphi, ntheta), optional, default=None
        (Traced) The magnetic field distribution on the plasma surface. Will be filled with zeros by default.
    plasma_coil_distance : float, optional, default=None
        (Traced) The coil-plasma distance. Is set to ``None`` by default, but a value must be provided if ``winding_dofs`` is not provided.
    winding_surface_generator : callable, optional, default=gen_winding_surface_atan
        (Static) The winding surface generator.
    winding_dofs : ndarray, shape (ndof_winding,)
        (Traced) The winding surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` convention.
        Will be generated using ``winding_surface_generator`` if ``plasma_coil_distance`` is provided. Must be provided otherwise.
    winding_mpol : int, optional, default=6
        (Static) The number of poloidal Fourier harmonics in the winding surface.
    winding_ntor : int, optional, default=5
        (Static) The number of toroidal Fourier harmonics in the winding surface.
    winding_quadpoints_phi : ndarray, shape (nphi_winding,), optional, default=None
        (Traced) Will be set to ``jnp.linspace(0, 1, 32*nfp, endpoint=False)`` by default.
    winding_quadpoints_theta : ndarray, shape (ntheta_winding,), optional, default=None
        (Traced) Will be set to ``jnp.linspace(0, 1, 34, endpoint=False)`` by default.
    objective_name : tuple, optional, default='f_B_normalized_by_Bnormal_IG'
        (Static) The names of the objective functions. Must be a member of ``quadcoil.objective`` that outputs a scalar.
    objective_weight : ndarray, optional, default=None
        (Traced) The weights of the objective functions. Derivatives will be calculated w.r.t. this quantity.
    objective_unit : tuple, optional, default=None
        (Traced) The normalization constants of the objective terms, so that ``f/objective_unit`` is :math:`O(1)`. May contain ``None``
    constraint_name : tuple, optional, default=()
        (Static) The names of the constraint functions. Must be a member of ``quadcoil.objective`` that outputs a scalar.
    constraint_type : tuple, optional, default=()
        (Static) The types of the constraints. Must consist of ``'>='``, ``'<='``, ``'=='`` only.
    constraint_unit : tuple, optional, default=()
        (Traced) The normalization constants of the constraints, so that ``f/constraint_unit`` is :math:`O(1)` May contain ``None``.
    constraint_value : ndarray, optional, default=()
        (Traced) The constraint thresholds. Derivatives will be calculated w.r.t. this quantity.
    metric_name : tuple, optional, default=('f_B', 'f_K')
        (Static) The names of the functions to diagnose the coil configurations with. Will be differentiated w.r.t. other input quantities.
    c_init : float, optional, default=1.
        (Traced) The initial :math:`c` factor. Please see *Constrained Optimization and Lagrange Multiplier Methods* Chapter 3.
    c_growth_rate : float, optional, default=1.2
        (Traced) The growth rate of the :math:`c` factor.
    fstop_outer : float, optional, default=1e-7
        (Traced) Constraint tolerance of the outer augmented Lagrangian loop. Terminates when any 3 of the outer conditions is satisfied.
    ctol_outer : float, optional, default=1e-7
        (Traced) Constraint tolerance of the outer augmented Lagrangian loop. Terminates when any 3 of the outer conditions is satisfied.
    xstop_outer : float, optional, default=1e-7
        (Traced) Convergence rate tolerance of the outer augmented Lagrangian loop. Terminates when any 3 of the outer conditions is satisfied.
    gtol_outer : float, optional, default=1e-7
        (Traced) Gradient tolerance of the outer augmented Lagrangian loop. Terminates when any is satisfied.
    fstop_inner : float, optional, default=1e-7
        (Traced) Gradient tolerance of the inner LBFGS iteration. Terminates when any is satisfied.
    xstop_inner : float, optional, default=0
        (Traced) Gradient tolerance of the inner LBFGS iteration. Terminates when any is satisfied.
    gtol_inner : float, optional, default=1e-7
        (Traced) Gradient tolerance of the inner LBFGS iteration. Terminates when any is satisfied.
    maxiter_toter : int, optional, default=50
        (Static) The maximum of the outer iteration.
    maxiter_tot : int, optional, default=500
        (Static) The maximum of the inner iteration.
    value_only : bool, optional, default=False
        (Static) When ``True``, skip gradient calculations.
    verbose : bool, optional, default=False
        (Static) Print things when ``True``.
    '''
    if verbose:
        jax.debug.print('Running QUADCOIL in verbose mode.')
        jax.debug.print('')
        jax.debug.print('----- Initializing ----- ')
    # ----- Default parameters -----
    if plasma_quadpoints_phi is None:
        plasma_quadpoints_phi = jnp.linspace(0, 1/nfp, 32, endpoint=False)
    if plasma_quadpoints_theta is None:
        plasma_quadpoints_theta = jnp.linspace(0, 1, 34, endpoint=False)
    if winding_quadpoints_phi is None:
        winding_quadpoints_phi = jnp.linspace(0, 1, 32*nfp, endpoint=False)
    if winding_quadpoints_theta is None:
        winding_quadpoints_theta = jnp.linspace(0, 1, 34, endpoint=False)
    if quadpoints_phi is None:
        len_phi = len(winding_quadpoints_phi)//nfp
        quadpoints_phi = winding_quadpoints_phi[:len_phi]
    else:
        quadpoints_phi = quadpoints_phi
    if quadpoints_theta is None:
        quadpoints_theta = winding_quadpoints_theta
    else:
        quadpoints_theta = quadpoints_theta
    if plasma_coil_distance is None and winding_dofs is None:
         raise ValueError('At least one of plasma_coil_distance and winding_dofs must be provided.')
    if plasma_coil_distance is not None and winding_dofs is not None:
         raise ValueError('Only one of plasma_coil_distance and winding_dofs can be provided.')
    # cp_mn need to be normalized to ~1 for the optimizer to behave well.
    # by default we do this using the net poloidal/toroidal current.
    if cp_mn_unit is None:
        # Scaling current potential dofs to ~1
        # Phi has the unit of Ampere (magnetic dipole density)
        # One potential choice of normalizing constant is the total net current.
        # We assume the current potential is on the same order with it.
        total_current = jnp.linalg.norm(jnp.array([
            net_poloidal_current_amperes,
            net_toroidal_current_amperes
        ]))
        if Bnormal_plasma is None:
            cp_mn_unit = total_current
        else:
            # Another choice is based on the Bnormal_plasma. 
            # B/mu0 * len has the unit of A.
            # Here, the first component of plasma_dofs
            # is close to the major radius. 
            # This is useful when the net currents are zero. 
            Bnormal_factor = jnp.max(jnp.abs(Bnormal_plasma * 1e7 * plasma_coil_distance))
            # Always select total_current unless it is zero.
            cp_mn_unit = jnp.where(total_current > 0, total_current, Bnormal_factor)
    # ----- Type checking -----
    if not isinstance(objective_name, str):
        if not isinstance(objective_name, tuple):
            raise TypeError('objective_name must be a tuple or string. It is:', type(objective_name))
        if not is_ndarray(objective_weight, 1):
            raise TypeError('objective_weight must be an 1d array. It is:', type(objective_weight))
        # if not isinstance(objective_unit, tuple):
        #     raise TypeError('objective_unit must be a tuple. It is:', type(objective_unit))
        if len(objective_name) != len(objective_weight) or len(objective_name) != len(objective_unit):
            raise ValueError('objective_name, objective_weight, and objective_unit must have the same len')
    if isinstance(metric_name, str):
        metric_name = (metric_name,)
    if not isinstance(constraint_name, tuple):
        raise TypeError('constraint_name must be a tuple. It is:', type(constraint_name))
    if not isinstance(constraint_type, tuple):
        raise TypeError('constraint_type must be a tuple. It is:', type(constraint_type))
    # if not isinstance(constraint_unit, tuple):
    #     raise TypeError('constraint_unit must be a tuple. It is:', type(constraint_unit))
    if (
        len(constraint_name) != len(constraint_type) 
        or len(constraint_name) != len(constraint_unit)
        or len(constraint_name) != len(constraint_value)
    ):
        raise ValueError('constraint_name, constraint_type, constraint_unit, '\
                     'and constraint_value must have the same len. They each '\
                     'are: '
                     + str(constraint_name) + ', ' 
                     + str(constraint_type) + ', ' 
                     + str(constraint_unit) + ', ' 
                     + str(constraint_value) + '.')    
   
    # A dictionary containing all parameters that the problem depends on.
    # These elements will always be in y.
    y_dict_current = {
        'plasma_dofs': plasma_dofs,
        'net_poloidal_current_amperes': net_poloidal_current_amperes,
        'net_toroidal_current_amperes': net_toroidal_current_amperes,
        'constraint_value': constraint_value, 
    }
    # Only differentiate wrt the weights when 
    # it's not None.
    if objective_weight is not None:
        y_dict_current['objective_weight'] = jnp.array(objective_weight)
    else:
        y_dict_current['objective_weight'] = None
    # Only differentiate wrt normal field when 
    # it's not zero.
    if Bnormal_plasma is not None:
        if verbose:
            jax.debug.print('Maximum Bnormal_plasma: {x}', x=jnp.max(jnp.abs(Bnormal_plasma)))
        y_dict_current['Bnormal_plasma'] = Bnormal_plasma
    # Include winding dofs when it's provided.
    if plasma_coil_distance is None:
        if verbose:
            jax.debug.print('Using custom winding surface.')
        y_dict_current['winding_dofs'] = winding_dofs
    else:
        if verbose:
            jax.debug.print('Plasma-coil distance (m): {x}', x=plasma_coil_distance)
        y_dict_current['plasma_coil_distance'] = plasma_coil_distance
    
    # ----- Printing inputs -----
    if verbose:
        jax.debug.print('Evaluation phi quadpoint num: {x}', x=len(quadpoints_phi))
        jax.debug.print('Evaluation theta quadpoint num: {x}', x=len(quadpoints_theta))
        jax.debug.print('Plasma phi quadpoint num: {x}', x=len(plasma_quadpoints_phi))
        jax.debug.print('Plasma theta quadpoint num: {x}', x=len(plasma_quadpoints_theta))
        jax.debug.print('Winding phi quadpoint num: {x}', x=len(winding_quadpoints_phi))
        jax.debug.print('Winding theta quadpoint num: {x}', x=len(winding_quadpoints_theta))
        jax.debug.print('Net poloidal current (A): {x}', x=net_poloidal_current_amperes)
        jax.debug.print('Net toroidal current (A): {x}', x=net_toroidal_current_amperes)
        jax.debug.print('Constraint names: {x}', x=constraint_name)
        jax.debug.print('Constraint types: {x}', x=constraint_type)
        jax.debug.print('Constraint units: {x}', x=constraint_unit)
        jax.debug.print('Constraint values: {x}', x=constraint_value)
        jax.debug.print('Objective names: {x}', x=objective_name)
        jax.debug.print('Objective units: {x}', x=objective_unit)
        jax.debug.print('Objective weights: {x}', x=objective_weight)
        jax.debug.print('Numerical parameters:')
        jax.debug.print('    c_init: {x}', x=c_init)
        jax.debug.print('    c_growth_rate: {x}', x=c_growth_rate)
        jax.debug.print('    fstop_outer: {x}', x=fstop_outer)
        jax.debug.print('    xstop_outer: {x}', x=xstop_outer)
        jax.debug.print('    gtol_outer: {x}', x=gtol_outer)
        jax.debug.print('    ctol_outer: {x}', x=ctol_outer)
        jax.debug.print('    fstop_inner: {x}', x=fstop_inner)
        jax.debug.print('    xstop_inner: {x}', x=xstop_inner)
        jax.debug.print('    gtol_inner: {x}', x=gtol_inner)
        jax.debug.print('    maxiter_tot: {x}', x=maxiter_tot)
        jax.debug.print('    maxiter_inner: {x}', x=maxiter_inner)
        jax.debug.print('')
        jax.debug.print('----- Solving ----- ')
    
    # ----- Helper functions -----
    def y_to_qp(y_dict):
        plasma_surface = SurfaceRZFourierJAX(
            nfp=nfp, stellsym=stellsym, 
            mpol=plasma_mpol, ntor=plasma_ntor, 
            quadpoints_phi=plasma_quadpoints_phi, 
            quadpoints_theta=plasma_quadpoints_theta,
            dofs=y_dict['plasma_dofs']
        )
        # winding surface is provided. 
        # Its dofs will be among x.
        if plasma_coil_distance is None:
            winding_surface = SurfaceRZFourierJAX(
                nfp=nfp, stellsym=stellsym, 
                mpol=winding_mpol, ntor=winding_ntor, 
                quadpoints_phi=winding_quadpoints_phi, 
                quadpoints_theta=winding_quadpoints_theta,
                dofs=y_dict['winding_dofs']
            )
        # winding surface is not provided. 
        # Its dofs will not be among x.
        else:
            winding_dofs_temp = winding_surface_generator(
                plasma_gamma=plasma_surface.gamma(), 
                d_expand=y_dict['plasma_coil_distance'], 
                nfp=plasma_surface.nfp, stellsym=plasma_surface.stellsym,
                mpol=winding_mpol,
                ntor=winding_ntor,
            )
            winding_surface = SurfaceRZFourierJAX(
                nfp=nfp,
                stellsym=stellsym,
                mpol=winding_mpol,
                ntor=winding_ntor,
                quadpoints_phi=winding_quadpoints_phi,
                quadpoints_theta=winding_quadpoints_theta,
                dofs=winding_dofs_temp
            )
        if Bnormal_plasma is None:
            Bnormal_plasma_temp = jnp.zeros((
                len(plasma_surface.quadpoints_phi), 
                len(plasma_surface.quadpoints_theta)
            ))
        else:
            Bnormal_plasma_temp = y_dict['Bnormal_plasma']
        qp_temp = QuadcoilParams(
            plasma_surface=plasma_surface, 
            winding_surface=winding_surface, 
            net_poloidal_current_amperes=y_dict['net_poloidal_current_amperes'], 
            net_toroidal_current_amperes=y_dict['net_toroidal_current_amperes'],
            Bnormal_plasma=Bnormal_plasma_temp,
            mpol=mpol, 
            ntor=ntor, 
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta, 
        )
        return qp_temp
    
    # A function that handles the parameter-dependence
    # of all objective functions. 
    # Maps parameters (dict) -> f, g, h, (callables, x -> scalar, arr, arr)
    def f_g_ineq_h_eq_from_y(
            y_dict,
            objective_name=objective_name,
            objective_unit=objective_unit,
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
            constraint_value=constraint_value,
            cp_mn_unit=cp_mn_unit,
        ):  
        qp_temp = y_to_qp(y_dict)
        f_obj = parse_objectives(
            objective_name=objective_name, 
            objective_unit=objective_unit,
            objective_weight=y_dict['objective_weight'], 
        )
        g_ineq, h_eq = parse_constraints(
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
            constraint_value=constraint_value,
        )
        # Scaling cp_mn to ~1 to make the optimizer behave better
        scaled_f_obj = lambda x: f_obj(qp_temp, x * cp_mn_unit)
        scaled_g_ineq = lambda x: g_ineq(qp_temp, x * cp_mn_unit)
        scaled_h_eq = lambda x: h_eq(qp_temp, x * cp_mn_unit)
        
        return scaled_f_obj, scaled_g_ineq, scaled_h_eq
    
    # ----- Initializing solver and values -----

    qp = y_to_qp(y_dict_current)
    if x_init is None:
        x_init_scaled = jnp.zeros(qp.ndofs)
    else:
        x_init_scaled = x_init / cp_mn_unit
    f_obj, g_ineq, h_eq = f_g_ineq_h_eq_from_y(y_dict_current)
    mu_init = jnp.zeros_like(g_ineq(x_init_scaled))
    lam_init = jnp.zeros_like(h_eq(x_init_scaled))

    # ----- Solving QUADCOIL -----
    
    # A dictionary containing augmented lagrangian info
    # and the last augmented lagrangian objective function for 
    # implicit differentiation.
    solve_results = solve_constrained(
        x_init=x_init_scaled,
        f_obj=f_obj,
        lam_init=lam_init,
        mu_init=mu_init,
        h_eq=h_eq,
        g_ineq=g_ineq,
        c_init=c_init,
        c_growth_rate=c_growth_rate,
        fstop_outer=fstop_outer,
        ctol_outer=ctol_outer,
        xstop_outer=xstop_outer,
        gtol_outer=gtol_outer,
        fstop_inner=fstop_inner,
        xstop_inner=xstop_inner,
        gtol_inner=gtol_inner,
        maxiter_tot=maxiter_tot,
        maxiter_inner=maxiter_inner,
    )
    # The optimum, unit-less.
    x_k = solve_results['inner_fin_x']
    cp_mn = x_k * cp_mn_unit
    
    # ----- Calculating metrics and gradients
    if value_only: 
        if verbose:
            jax.debug.print('')
            jax.debug.print('----- Calculating metrics only ----- ')
        out_dict = {}
        for metric_name_i in metric_name:
            f_metric_with_unit = get_objective(metric_name_i)
            f_metric = lambda x, y: f_metric_with_unit(y_to_qp(y), x * cp_mn_unit)
            metric_result_i = f_metric(x_k, y_dict_current)
            out_dict[metric_name_i] = {
                'value': metric_result_i
            }
            if verbose:
                jax.debug.print('Metric evaluated. {x} = {y}', x=metric_name_i, y=metric_result_i)
        return out_dict, qp, cp_mn, solve_results
    if verbose:
        jax.debug.print('')
        jax.debug.print('----- Calculating metrics and gradients ----- ')
    
    ''' Recover the l_k in the last iteration for dx_k/dy '''

    # First, we reproduce the augmented lagrangian objective l_k that 
    # led to the optimum.
    c_k = solve_results['inner_fin_c']
    lam_k = solve_results['inner_fin_lam']
    mu_k = solve_results['inner_fin_mu']
    # @jit
    def l_k(x, y): 
        f_obj, g_ineq, h_eq = f_g_ineq_h_eq_from_y(y)
        gplus = lambda x, mu, c: jnp.max(jnp.array([g_ineq(x), -mu/c]), axis=0)
        return(
            f_obj(x) 
            + lam_k@h_eq(x) 
            + c_k/2 * (
                jnp.sum(h_eq(x)**2) 
                + jnp.sum(gplus(x, mu_k, c_k)**2)
            )
        )
    nabla_x_l_k = jacrev(l_k, argnums=0)
    nabla_y_l_k = jacrev(l_k, argnums=1)
    nabla_y_l_k_for_hess = lambda x: nabla_y_l_k(x, y_dict_current)
    hess_l_k = jacrev(nabla_x_l_k)(x_k, y_dict_current)
    out_dict = {}
    for metric_name_i in metric_name:
        f_metric_with_unit = get_objective(metric_name_i)
        f_metric = lambda x, y: f_metric_with_unit(y_to_qp(y), x * cp_mn_unit)
        nabla_x_f = jacrev(f_metric, argnums=0)(x_k, y_dict_current)
        nabla_y_f = jacrev(f_metric, argnums=1)(x_k, y_dict_current)
        vihp = jnp.linalg.solve(hess_l_k, nabla_x_f)
        # Now we calculate df/dy using vjp
        # \nabla_{x_k} f [-H(l_k, x_k)^-1 \nabla_{x_k}\nabla_{y} l_k]
        # Primal and tangent must be the same shape
        _, dfdy1 = jvp(nabla_y_l_k_for_hess, primals=[x_k], tangents=[vihp])
        # \nabla_{y} f
        dfdy2 = nabla_y_f
        # This was -dfdy1 + dfdy2 in the old code where
        # y is an array. Now y is a dict, and 
        # dfdy1, dfdy2 are both dicts.
        dfdy = {}
        for key in dfdy1.keys():
            if dfdy1[key] is not None and dfdy2[key] is not None:
                dfdy['df_d' + key] = -jnp.array(dfdy1[key]) + jnp.array(dfdy2[key])
        metric_result_i = f_metric(x_k, y_dict_current)
        if verbose:
            jax.debug.print('Metric evaluated. {x} = {y}', x=metric_name_i, y=metric_result_i)
        out_dict[metric_name_i] = {
            'value': metric_result_i, 
            'grad': dfdy
        }
    return(out_dict, qp, cp_mn, solve_results)


