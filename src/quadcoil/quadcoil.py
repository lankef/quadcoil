from quadcoil import (
    parse_objectives, parse_constraints, get_objective,
    gen_winding_surface_atan, 
    SurfaceRZFourierJAX, QuadcoilParams, 
    solve_constrained)
from functools import partial
from jax import jacrev, jvp, jit
import jax.numpy as jnp

@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
    'mpol',
    'ntor',
    # 'plasma_dofs',
    'plasma_mpol',
    'plasma_ntor',
    # 'net_poloidal_current_amperes',
    # 'net_toroidal_current_amperes',
    # 'quadpoints_phi',
    # 'quadpoints_theta',
    # 'cp_mn_unit',
    # - Plasma options
    # 'plasma_quadpoints_phi',
    # 'plasma_quadpoints_theta',
    # 'Bnormal_plasma',
    # - WS options
    # 'plasma_coil_distance',
    'winding_surface_generator',
    'winding_surface_generator_args',
    # 'winding_dofs',
    'winding_mpol',
    'winding_ntor',
    # 'winding_quadpoints_phi',
    # 'winding_quadpoints_theta',
    # - Objectives
    'objective_name',
    # 'objective_weight',
    # 'objective_unit',
    # - Constraints 
    'constraint_name',
    'constraint_type',
    # 'constraint_unit',
    # 'constraint_value',
    # - Solver options
    'metric_name',
    # 'c_init',
    # 'c_growth_rate',
    # 'ftol_outer',
    # 'ctol_outer',
    # 'xtol_outer',
    # 'gtol_outer',
    # 'ftol_inner',
    # 'xtol_inner',
    # 'gtol_inner',
    'maxiter_inner',
    'maxiter_outer',
])
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
    mpol:int=4,
    ntor:int=4,
    # Quadpoints to evaluate objectives at
    quadpoints_phi=None,
    quadpoints_theta=None,
    # Current potential's normalization constant. 
    # By default will be generated from net total current.
    cp_mn_unit=None ,
    
    # - Plasma parameters
    plasma_quadpoints_phi=None,
    plasma_quadpoints_theta=None,
    Bnormal_plasma=None,

    # - Winding parameters (offset)
    plasma_coil_distance=None,
    winding_surface_generator=gen_winding_surface_atan,
    winding_surface_generator_args={'pol_interp': 1, 'lam_tikhonov': 0.05},

    # - Winding parameters (Providing surface)
    winding_dofs=None,
    winding_mpol=5,
    winding_ntor=5,
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
    constraint_value=(),
    # - Metrics to study
    metric_name=('f_B', 'f_K'),

    # - Solver options
    c_init=1.,
    c_growth_rate=1.1,
    ftol_outer=1e-7, # convergence rate tolerance
    ctol_outer=1e-7, # constraint tolerance
    xtol_outer=1e-7, # convergence rate tolerance
    gtol_outer=1e-7, # gradient tolerance
    ftol_inner=1e-7,
    xtol_inner=0.,
    gtol_inner=1e-7,
    maxiter_inner=1500,
    maxiter_outer=50,
):
    r'''
    Solves a QUADCOIL problem.

    Parameters
    ----------
    nfp : int
        (Static) The number of field periods.
    stellsym : bool
        (Static) Stellarator symmetry.
    plasma_mpol
        (Static) The number of poloidal Fourier harmonics in the plasma boundary.
    plasma_ntor
        (Static) The number of toroidal Fourier harmonics in the plasma boundary.
    plasma_dofs
        (Static) The plasma surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` covention.
    net_poloidal_current_amperes: float
        (Traced) The net poloidal current :math:`G`.
    net_toroidal_current_amperes: float, optional, default=0
        (Traced) The net toroidal current :math:`I`.
    mpol: int, optional, default=4
        (Static) The number of poloidal Fourier harmonics in the current potential :math:`\Phi_{sv}`,
        which represents the sheet current by 
        :math:`\mathbf{K} = \hat{\mathbf{n}}\times\nabla(\Phi_{sv} + \frac{G\phi'}{2\pi} + \frac{I\theta'}{2\pi})`,
        where :math:`G` is the net poloidal current, and :math:`I` is the net toroidal current.
    ntor: int, optional, default=4
        (Static) The number of toroidal Fourier harmonics in :math:`\Phi_{sv}`.
    quadpoints_phi : ndarray, shape (nphi,), optional, default=None
        (Traced) The poloidal quadrature points on the winding surface to evaluate the objectives at.
        Uses one period from the winding surfacee by default.
    quadpoints_theta : ndarray, shape (ntheta,), optional, default=None
        (Traced) The toroidal quadrature points on the winding surface to evaluate the objectives at.
        Uses one period from the winding surfacee by default.
    cp_mn_unit : floatm optional, default=None 
        (Traced) Current potential's normalization constant. 
        By default will be generated from total net current.
    plasma_quadpoints_phi : ndarray, shape (nphi_plasma,), optional, default=None
        (Traced) Will be set to ``jnp.linspace(0, 1/nfp, 32, endpoint=False)`` by default.
    plasma_quadpoints_theta : ndarray, shape (ntheta_plasma,), optional, default=None
        (Traced) Will be set to ``jnp.linspace(0, 1, 32, endpoint=False)`` by default.
    Bnormal_plasma : ndarray, shape (nphi, ntheta), optional, default=None
        (Traced) The magnetic field distribution on the plasma 
        surface. will be filled with zeros by default. 
    plasma_coil_distance : float, optional, default=None
        (Traced) The coil-plasma distance. Is set to ``None`` by default, 
        but a value must be provided if ``winding_dofs`` is not provided.
    winding_surface_generator : callable, optional, default=gen_winding_surface_atan
        (Static) The winding surface generator.
    winding_surface_generator_args : dict, optional, default={'pol_interp': 1, 'lam_tikhonov': 0.05}
        (Static) The arguments for the winding surface generator.
    winding_dofs : ndarray, shape (ndof_winding,)
        (Traced) The winding surface degrees of freedom. 
        Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` covention.
        Will be generated using ``winding_surface_generator`` if 
        ``plasma_coil_distance`` is provided. Must be provided otherwise.
    winding_mpol : int, optional, default=5
        (Static) The number of poloidal Fourier harmonics in the winding surface.
    winding_ntor : int, optional, default=5
        (Static) The number of toroidal Fourier harmonics in the winding surface.
    winding_quadpoints_phi : ndarray, shape (nphi_winding,), optional, default=None
        (Traced) Will be set to ``jnp.linspace(0, 1, 32*nfp, endpoint=False)`` by default.
    winding_quadpoints_theta : ndarray, shape (ntheta_winding,), optional, default=None
        (Traced) Will be set to ``jnp.linspace(0, 1, 32, endpoint=False)`` by default.
    objective_name: tuple, optional, default='f_B_normalized_by_Bnormal_IG',
        (Static) The names of the objective functions. Must be a member of ``quadcoil.objective`` that outputs a scalar.
    objective_weight: ndarray, optional, default=None,
        (Traced) The weights of the objective functions. Derivatives will be calculated w.r.t. this quantity.
    objective_unit: tuple, optional, default=None,
        (Traced) The normalization constants of the objective terms, so that ``f/objective_unit`` is :math:`O(1)`.
    constraint_name : tuple, optional, default=(),
        (Static) The names of the constraint functions. Must be a member of ``quadcoil.objective`` that outputs a scalar.
    constraint_type : tuple, optional, default=(),
        (Static) The types of the constraints. Must consists of ``'>=', '<=', '=='`` only.
    constraint_unit : tuple, optional, default=(),
        (Traced) The normalization constants of the constraints, so that ``f/constraint_unit`` is :math:`O(1)`.
    constraint_value : ndarray, optional, default=(),
        (Traced) The constraint thresholds. Derivatives will be calculated w.r.t. this quantity.
    metric_name=('f_B', 'f_K'),
        (Static) The names of the functions to diagnose the coil configurations with. Will be differentiated 
        w.r.t. other input quantities.  
    c_init : float, optional, default=1.
        (Traced) The initial :math:`c` factor. Please see
        *Constrained Optimization and Lagrange Multiplier Methods* Chapter 3.
    c_growth_rate : float, optional, default=1.1
        (Traced) The growth rate of the :math:`c` factor.
    ftol_outer : float, optional, default=1e-7
        (Traced) Constraint tolerance of the outer augmented 
        Lagrangian loop. Terminates when any is satisfied. 
    ctol_outer : float, optional, default=1e-7
        (Traced) Constraint tolerance of the outer augmented 
        Lagrangian loop. Terminates when any is satisfied. 
    xtol_outer : float, optional, default=1e-7
        (Traced) Convergence rate tolerance of the outer augmented 
        Lagrangian loop. Terminates when any is satisfied. 
    gtol_outer : float, optional, default=1e-7
        (Traced) Gradient tolerance of the outer augmented 
        Lagrangian loop. Terminates when any is satisfied. 
    ftol_inner : float, optional, default=1e-7
        (Traced) Gradient tolerance of the inner LBFGS 
        iteration. Terminates when any is satisfied. 
    xtol_inner : float, optional, default=0
        (Traced) Gradient tolerance of the inner LBFGS 
        iteration. Terminates when any is satisfied. 
    gtol_inner : float, optional, default=1e-7
        (Traced) Gradient tolerance of the inner LBFGS 
        iteration. Terminates when any is satisfied. 
    maxiter_outer: int, optional, default=50
        (Static) The maximum of the outer iteration.
    maxiter_inner: int, optional, default=1500
        (Static) The maximum of the inner iteration.
    '''
    ''' Default parameters '''
    
    if plasma_quadpoints_phi is None:
        plasma_quadpoints_phi = jnp.linspace(0, 1/nfp, 32, endpoint=False)
    if plasma_quadpoints_theta is None:
        plasma_quadpoints_theta = jnp.linspace(0, 1, 32, endpoint=False)
    if winding_quadpoints_phi is None:
        winding_quadpoints_phi = jnp.linspace(0, 1, 32*nfp, endpoint=False)
    if winding_quadpoints_theta is None:
        winding_quadpoints_theta = jnp.linspace(0, 1, 32, endpoint=False)
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
            Bnormal_factor = jnp.abs(Bnormal_plasma * 1e7 * plasma_coil_distance)
            # Always select total_current unless it is zero.
            cp_mn_unit = jnp.where(total_current > 0, total_current, Bnormal_factor)
    if not isinstance(objective_name, str):
        if not isinstance(objective_name, tuple):
            raise TypeError('objective_name must be a tuple or string') 
        if not isinstance(objective_weight, tuple):
            raise TypeError('objective_weight must be a tuple') 
        if not isinstance(objective_unit, tuple):
            raise TypeError('objective_unit must be a tuple') 
        if len(objective_name) != len(objective_weight) or len(objective_name) != len(objective_unit):
            raise ValueError('objective_name, objective_weight, and objective_unit must have the same len')
    if not isinstance(constraint_name, tuple):
        raise TypeError('constraint_name must be a tuple') 
    if not isinstance(constraint_type, tuple):
        raise TypeError('constraint_type must be a tuple') 
    if not isinstance(constraint_unit, tuple):
        raise TypeError('constraint_unit must be a tuple') 
    if (
        len(constraint_name) != len(constraint_type) 
        or len(constraint_name) != len(constraint_unit)
        or len(constraint_name) != len(constraint_value)
    ):
        raise ValueError('constraint_name, constraint_type, constraint_unit,'\
                         ' and constraint_value must have the same len')     
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
        y_dict_current['Bnormal_plasma'] = Bnormal_plasma
    # Include winding dofs when it's provided.
    if plasma_coil_distance is None:
        y_dict_current['winding_dofs'] = winding_dofs
    else:
        y_dict_current['plasma_coil_distance'] = plasma_coil_distance
    ''' Helper functions '''
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
                **winding_surface_generator_args
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
    
    ''' Initializing solver and values '''

    qp = y_to_qp(y_dict_current)
    x_init_scaled = jnp.zeros(qp.ndofs)
    f_obj, g_ineq, h_eq = f_g_ineq_h_eq_from_y(y_dict_current)
    mu_init = jnp.zeros_like(g_ineq(x_init_scaled))
    lam_init = jnp.zeros_like(h_eq(x_init_scaled))

    ''' Solving QUADCOIL '''
    
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
        ftol_outer=ftol_outer,
        ctol_outer=ctol_outer,
        xtol_outer=xtol_outer,
        gtol_outer=gtol_outer,
        ftol_inner=ftol_inner,
        xtol_inner=xtol_inner,
        gtol_inner=gtol_inner,
        maxiter_inner=maxiter_inner,
        maxiter_outer=maxiter_outer,
    )
    # The optimum, unit-less.
    x_k = solve_results['x_k']
    cp_mn = x_k * cp_mn_unit
    
    ''' Recover the l_k in the last iteration for dx_k/dy '''

    # First, we reproduce the augmented lagrangian objective l_k that 
    # led to the optimum.
    c_k = solve_results['c_k']
    lam_k = solve_results['lam_k']
    mu_k = solve_results['mu_k']
    # @jit
    def l_k(x, y): 
        f_obj, g_ineq, h_eq = f_g_ineq_h_eq_from_y(y)
        gplus = lambda x, mu, c: jnp.max(jnp.array([g_ineq(x), -mu/c]), axis=0)
        # l_k = lambda x: (
        #     f_obj(x) 
        #     + lam_k@h_eq(x) 
        #     + c_k/2 * (
        #         jnp.sum(h_eq(x)**2) 
        #         + jnp.sum(gplus(x, mu_k, c_k)**2)
        #     )
        # ) 
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
        out_dict[metric_name_i] = {
            'value': f_metric(x_k, y_dict_current), 
            'grad': dfdy
        }
    return(out_dict, qp, cp_mn, solve_results)