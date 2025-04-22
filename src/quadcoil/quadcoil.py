from quadcoil import (
    merge_callables, get_quantity,
    gen_winding_surface_atan, 
    SurfaceRZFourierJAX, QuadcoilParams, 
    solve_constrained, run_opt_lbfgs,
    is_ndarray
)
from quadcoil.wrapper import _parse_objectives, _parse_constraints
from functools import partial
from quadcoil.quantity import Bnormal
from jax import jacrev, jvp, jit
import jax
import jax.numpy as jnp

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
    phi_mn_init=None, 
    # Current potential's normalization constant. 
    # By default will be generated from net total current.
    phi_unit=None,
    
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
    c_growth_rate:float=1.5,
    grad_l_stop_outer:float=1e-7, # convergence rate tolerance
    xstop_outer:float=0., # convergence rate tolerance
    gtol_outer:float=1e-7, # gradient tolerance
    ctol_outer:float=1e-7, # constraint tolerance
    fstop_inner:float=0,
    xstop_inner:float=0.,
    gtol_inner:float=0.1,
    gtol_inner_fin:float=1e-7,
    maxiter_tot:int=10000,
    maxiter_inner:int=2000,
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
    phi_mn_init : ndarray, optional, default=None
        (Traced) The initial guess. All zeros by default.
    phi_unit : float, optional, default=None
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
    grad_l_stop_outer : float, optional, default=1e-7
        (Traced) Terminates when the improvement in the Lagrangian's gradient,
        :math:`\nabla L`, falls below this.  
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
    gtol_inner : float, optional, default=0.1
        (Traced) Gradient tolerance of the inner LBFGS iteration, normalized by the starting gradient.
    gtol_inner_fin : float, optional, default=1e-7
        (Traced) Gradient tolerance of the final inner LBFGS iteration, normalized by the starting gradient.
    maxiter_toter : int, optional, default=50
        (Static) The maximum of the outer iteration.
    maxiter_tot : int, optional, default=500
        (Static) The maximum of the inner iteration.
    value_only : bool, optional, default=False
        (Static) When ``True``, skip gradient calculations.
    verbose : int, optional, default=False
        (Static) Print general info when ``verbose==1``. 
        Print inside the outer iteration loop, too, when ``verbose==2``.
    '''
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
        if verbose>0:
            jax.debug.print('Maximum Bnormal_plasma: {x}', x=jnp.max(jnp.abs(Bnormal_plasma)))
        y_dict_current['Bnormal_plasma'] = Bnormal_plasma
    # Include winding dofs when it's provided.
    if plasma_coil_distance is None:
        if verbose>0:
            jax.debug.print('Using custom winding surface.')
        y_dict_current['winding_dofs'] = winding_dofs
    else:
        if verbose>0:
            jax.debug.print('Plasma-coil distance (m): {x}', x=plasma_coil_distance)
        y_dict_current['plasma_coil_distance'] = plasma_coil_distance
    
    # ----- Printing inputs -----
    if verbose>0:
        jax.debug.print(
            'Running QUADCOIL in verbose mode \n\n'\
            '----- Input summary ----- \n'\
            'Evaluation phi quadpoint num: {n_quadpoints_phi}\n'\
            'Evaluation theta quadpoint num: {n_quadpoints_theta}\n'\
            'Plasma phi quadpoint num: {n_plasma_quadpoints_phi}\n'\
            'Plasma theta quadpoint num: {n_plasma_quadpoints_theta}\n'\
            'Winding phi quadpoint num: {n_winding_quadpoints_phi}\n'\
            'Winding theta quadpoint num: {n_winding_quadpoints_theta}\n'\
            'Net poloidal current (A): {net_poloidal_current_amperes}\n'\
            'Net toroidal current (A): {net_toroidal_current_amperes}\n'\
            'Constraint names: {constraint_name}\n'\
            'Constraint types: {constraint_type}\n'\
            'Constraint units: {constraint_unit}\n'\
            'Constraint values: {constraint_value}\n'\
            'Objective names: {objective_name}\n'\
            'Objective units: {objective_unit}\n'\
            'Objective weights: {objective_weight}\n'\
            'Numerical parameters:\n'\
            '    c_init: {c_init}\n'\
            '    c_growth_rate: {c_growth_rate}\n'\
            '    grad_l_stop_outer: {grad_l_stop_outer}\n'\
            '    xstop_outer: {xstop_outer}\n'\
            '    gtol_outer: {gtol_outer}\n'\
            '    ctol_outer: {ctol_outer}\n'\
            '    fstop_inner: {fstop_inner}\n'\
            '    xstop_inner: {xstop_inner}\n'\
            '    gtol_inner: {gtol_inner}\n'\
            '    gtol_inner_fin: {gtol_inner_fin}\n'\
            '    maxiter_tot: {maxiter_tot}\n'\
            '    maxiter_inner: {maxiter_inner}',
            n_quadpoints_phi=len(quadpoints_phi),
            n_quadpoints_theta=len(quadpoints_theta),
            n_plasma_quadpoints_phi=len(plasma_quadpoints_phi),
            n_plasma_quadpoints_theta=len(plasma_quadpoints_theta),
            n_winding_quadpoints_phi=len(winding_quadpoints_phi),
            n_winding_quadpoints_theta=len(winding_quadpoints_theta),
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            net_toroidal_current_amperes=net_toroidal_current_amperes,
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
            constraint_value=constraint_value,
            objective_name=objective_name,
            objective_unit=objective_unit,
            objective_weight=objective_weight,
            c_init=c_init,
            c_growth_rate=c_growth_rate,
            grad_l_stop_outer=grad_l_stop_outer,
            xstop_outer=xstop_outer,
            gtol_outer=gtol_outer,
            ctol_outer=ctol_outer,
            fstop_inner=fstop_inner,
            xstop_inner=xstop_inner,
            gtol_inner=gtol_inner,
            gtol_inner_fin=gtol_inner_fin,
            maxiter_tot=maxiter_tot,
            maxiter_inner=maxiter_inner,
        )
    
    # ----- Helper functions -----
    # y, the plasma and problem parameters, is a dictionary with 
    # varying shape depenbding on the problem's setup. 
    # qp is a "struct" that contains all the standard problem setup 
    # in a simsopt format. This is a function that converts "y" dictionaries
    # into qp, which instances of "_Quantities" accept.
    # We only use "y" because JAX can take derivatives w.r.t. dicts, 
    # and I want quadcoil outputs to look like dict derivatives, rather
    # than an internal object of QUADCOIOL.
    # This hopefully achieves 2 things:
    # 1. Make it simpler to implement new quantities like one would in simsopt
    # 2. Also allow QUADCOIL to output a dict with dynamic structure 
    # based on problem setup. (For example, the output will not contain)
    # gradients wrt coil-plasma distances if the winding surface is given. 
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
                len(plasma_quadpoints_phi), 
                len(plasma_quadpoints_theta)
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

    # ----- Objective function generator -----
    # A function that handles the parameter-dependence
    # of all objective functions. 
    # Maps parameters (dict) -> f, g, h, (callables, x -> scalar, arr, arr)
    # Used during implicit differentiation.
    # It also evaluates some basic properties for initialization.
    def f_g_ineq_h_eq_from_y(
            y_dict,
            objective_name=objective_name,
            objective_unit=objective_unit,
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
        ):  
        # First, fetching all objectives and constraints
        qp_temp = y_to_qp(y_dict)
        f_obj, g_obj_list, h_obj_list, aux_dofs_obj = _parse_objectives(
            objective_name=objective_name, 
            objective_unit=objective_unit,
            objective_weight=y_dict['objective_weight'], 
        )
        g_cons_list, h_cons_list, aux_dofs_cons = _parse_constraints(
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
            constraint_value=y_dict['constraint_value'],
        )
        # Merging constraints and aux dofs from different sources
        g_list = g_obj_list + g_cons_list
        h_list = h_obj_list + h_cons_list
        aux_dofs_init = aux_dofs_obj | aux_dofs_cons

        f_obj_x = lambda x: f_obj(qp_temp, x)
        g_ineq_x = lambda x: merge_callables(g_list)(qp_temp, x)
        h_eq_x = lambda x: merge_callables(h_list)(qp_temp, x)
        unconstrained = ((len(g_list) == 0) and (len(h_list) == 0))
        return f_obj_x, g_ineq_x, h_eq_x, unconstrained, aux_dofs_init

    # ----- Creating init parameters -----
    # Defining a shared problem parameter object
    qp = y_to_qp(y_dict_current)
    # f, g, h are Callable(qp, {'phi':, ..., })
    # i.e., they accepts unscaled input
    f_obj, g_ineq, h_eq, unconstrained, aux_dofs_init = f_g_ineq_h_eq_from_y(y_dict_current)
    mu_init = jnp.zeros_like(g_ineq(phi_mn_init))
    lam_init = jnp.zeros_like(h_eq(phi_mn_init))
    if phi_mn_init is None:
        phi_mn_init = jnp.zeros(qp.ndofs)

    # ----- Calculating the unit of phi -----
    # phi need to be normalized to ~1 for the optimizer to behave well.
    # by default we do this using the initial value of Bnormal
    if phi_unit is None:
        # Scaling current potential dofs to ~1
        # By default, we use the Bnormal value when 
        # phi=0 to generate this scaling factor.
        B_normal_estimate = jnp.average(jnp.abs(Bnormal(qp, {'phi': phi_mn_init}))) # Unit: T
        if plasma_coil_distance is not None:
            phi_unit = B_normal_estimate * 1e7 * jnp.abs(plasma_coil_distance)
        else:
            # The minor radius can be estimated from the 
            # n=0, m=1 rc mode of the surface.
            plasma_minor = plasma_dofs[plasma_ntor*2 + 1]
            winding_minor = winding_dofs[winding_ntor*2 + 1]
            phi_unit = B_normal_estimate * 1e7 * jnp.abs(plasma_minor - winding_minor)

    # ----- Creating scaled, flattened dof, 'x_arr' -----
    # The dof that the ptimizer sees, "x", is a flattened version of this dictionary
    x_dict = {
       'phi_scaled': phi_mn_init/phi_unit,
       # And auxiliary vars. Because we have already implemented 
       # scaling for them in _add_quantity instances, we do not 
       # need to scale them here.
    }
    # Calculating the structure of auxiliary dofs from the problem setup (qp).
    # The current dictionary's items are either None (scalar), tuple (known shape), or 
    # Callable(QuadcoilParams) (shapes that depend on problem setup)
    for key in aux_dofs_init.keys():
        if callable(aux_dofs_init[key]): 
            # Callable(qp: QuadcoilParams, dofs: dict, f_unit: float)
            x_dict[key] = aux_dofs_init[key](qp, {'phi': phi_mn_init})
        else:
            try:
                x_dict[key] = jnp.array(aux_dofs_init[key])
            except:
                raise TypeError(
                    f'The auxiliary variable {key} is not a callable, '\
                    'and cannot be converted to an array. Its value is: '\
                    f'{str(aux_dofs_init[key])}. This is dur to improper '\
                    'implementation of the physical quantity. Please contact the developers.')
    # dofs_init is a dict for readability. However, for simple
    # implementation, we need to unravel it into a jax array. 
    # Here we perform the unraveling and 
    x_arr, unravel_fn = jax.flatten_util.ravel_pytree(x_dict)
    # This block prints out a summary on the auxiliary vars and 
    # phi degrees of freedom.
    def x_arr_to_dofs(x, unravel_fn=unravel_fn, phi_unit=phi_unit):
        d = unravel_fn(x)
        # Replace scaled phi with regular phi
        # after unraveling for passing into 
        # f_obj, g_ineq and h_eq.
        dofs_temp = {k: v for k, v in {**d, "phi": d["phi_scaled"] * phi_unit}.items() if k != "phi_scaled"}
        return(dofs_temp)
    if verbose>0:
        dofs_summary = []
        for key, value in x_dict.items():
            dofs_summary.append(f"    {key}: {jnp.atleast_1d(value).shape}")
        final_str = "\n".join(dofs_summary)
        jax.debug.print(
            '----- DOF summary ----- \n'\
            'After converting non-smooth terms (such as |f|) into\n'\
            'smooth terms, auxiliary vars and constraints, the dofs are:\n{s}\n'\
            'Total # dofs (including auxiliary): {t}',
            s = final_str, t = len(x_arr) 
        )
    
    # ----- Scaling f, g, h -----
    # f, g and h should take x_arr, the flattened, scaled dofs.
    f_scaled = lambda x_scaled, f_obj=f_obj: f_obj(x_arr_to_dofs(x_scaled))
    g_scaled = lambda x_scaled, g_ineq=g_ineq: g_ineq(x_arr_to_dofs(x_scaled))
    h_scaled = lambda x_scaled, h_eq=h_eq: h_eq(x_arr_to_dofs(x_scaled))

    # ----- Solving QUADCOIL -----
    # A dictionary containing augmented lagrangian info
    # and the last augmented lagrangian objective function for 
    # implicit differentiation.
    # When unconstrained, this function instead serves the 
    # purpose of "zooming in" when iteration step lengths
    # are small.
    if unconstrained:
        x_opt, val_l_k, grad_l_k, niter_inner_k, dx_k, du_k, dL_k = run_opt_lbfgs(
            x_arr, 
            f_scaled, 
            maxiter_inner, 
            fstop_inner, 
            xstop_inner, 
            gtol_inner_fin,
            verbose
        )
        dofs_opt = x_arr_to_dofs(x_opt)
        solve_results = {
            'inner_fin_f': val_l_k,
            'inner_fin_x': x_opt,
            'inner_fin_grad_f': jnp.linalg.norm(grad_l_k),
            'inner_fin_niter': niter_inner_k,
            'inner_fin_dx_scaled': dx_k,
            'inner_fin_du': du_k,
            'inner_fin_df': dL_k,
            # The scaling factor for the next iteration
            # 'x_unit': jnp.average(jnp.abs(x_k)),
        }
        if verbose>0:       
            jax.debug.print(
                '----- Solver status summary -----\n'\
                'Final value of objective f: {f}\n'\
                'Final Max current potential (dipole density): {max_cp} (A)\n'\
                'Final Avg current potential (dipole density): {avg_cp} (A)\n'\
                '* Total L-BFGS iteration number: {niter}\n'\
                '    Phi scaling constant:  {x_unit_init}(A)\n'\
                # '    Final value of phi scaling constant: {x_unit}(A)\n'\
                '    Final value of grad f: {grad_f}\n'\
                '    Inner convergence rate in x (scaled): {inner_dx}, {inner_du}\n'\
                '    Inner convergence rate in f: {df}\n',
                f=val_l_k,
                niter=niter_inner_k,
                grad_f=jax.block_until_ready(solve_results['inner_fin_grad_f']),
                x_unit_init=phi_unit,
                inner_dx=jax.block_until_ready(solve_results['inner_fin_dx_scaled']),
                inner_du=jax.block_until_ready(solve_results['inner_fin_du']),
                df=jax.block_until_ready(solve_results['inner_fin_df']),
                max_cp=jnp.max(jnp.abs(dofs_opt['phi'])),
                avg_cp=jnp.average(jnp.abs(dofs_opt['phi'])),
            )
    else:
        solve_results = solve_constrained(
            x_init=x_arr,
            f_obj=f_scaled,
            lam_init=lam_init,
            mu_init=mu_init,
            h_eq=h_scaled,
            g_ineq=g_scaled,
            c_init=c_init,
            c_growth_rate=c_growth_rate,
            grad_l_stop_outer=grad_l_stop_outer,
            ctol_outer=ctol_outer,
            xstop_outer=xstop_outer,
            gtol_outer=gtol_outer,
            fstop_inner=fstop_inner,
            xstop_inner=xstop_inner,
            gtol_inner=gtol_inner,
            gtol_inner_fin=gtol_inner_fin,
            maxiter_tot=maxiter_tot,
            maxiter_inner=maxiter_inner,
            verbose=verbose
        )
        # The optimum, unit-less.
        x_opt = solve_results['inner_fin_x']
        dofs_opt = x_arr_to_dofs(x_opt)
        if verbose>0:       
            jax.debug.print(
                '----- Solver status summary -----\n'\
                'Final value of objective f: {f}\n'\
                'Final Max current potential (dipole density): {max_cp} (A)\n'\
                'Final Avg current potential (dipole density): {avg_cp} (A)\n'\
                '* Total L-BFGS iteration number: {niter}\n'\
                '    Phi scaling constant:  {x_unit_init}(A)\n'\
                # '    Final value of phi scaling constant: {x_unit}(A)\n'\
                '    Final value of grad f: {grad_f}\n'\
                '    Final value of constraint g: {g}\n'\
                '    Final value of constraint h: {h}\n'\
                '    Outer convergence rate in x (scaled): {dx}\n'\
                '    Outer convergence rate in g, h: {dg}, {dh}\n'\
                '    L-2 norm of the gradient of the Lagrangian:    {outer_grad_l}\n'\
                '    Change rate of the gradient of the Lagrangian: {outer_dgrad_l}\n'\
                '    Outer KKT condition 1: {KKT1}\n'\
                '    Outer KKT condition 2: {KKT2}\n'\
                '    Outer KKT condition 3: {KKT3}\n'\
                '    Outer KKT condition 4: {KKT4}\n'\
                '    Outer KKT condition 5: {KKT5}\n'\
                '    Outer KKT condition 6: {KKT6}\n'\
                '* Last inner L_BFGS iteration number: {inner_niter}\n'\
                '    Inner convergence rate in x (scaled): {inner_dx}, {inner_du}\n'\
                '    Inner convergence rate in l: {dl}\n',
                f=jax.block_until_ready(solve_results['inner_fin_f']),
                niter=jax.block_until_ready(solve_results['tot_niter']),
                grad_f=jax.block_until_ready(solve_results['inner_fin_grad_f']),
                g=jax.block_until_ready(solve_results['inner_fin_g']),
                h=jax.block_until_ready(solve_results['inner_fin_h']),
                x_unit_init=phi_unit,
                # x_unit=jax.block_until_ready(solve_results['x_unit']),
                dx=jax.block_until_ready(solve_results['outer_dx']),
                dg=jax.block_until_ready(solve_results['outer_dg']),
                dh=jax.block_until_ready(solve_results['outer_dh']),
                outer_dgrad_l=jax.block_until_ready(solve_results['outer_dgrad_l']),
                outer_grad_l=jax.block_until_ready(jnp.linalg.norm(solve_results['outer_grad_l'])),
                KKT1=jax.block_until_ready(solve_results['KKT1']),
                KKT2=jax.block_until_ready(solve_results['KKT2']),
                KKT3=jax.block_until_ready(solve_results['KKT3']),
                KKT4=jax.block_until_ready(solve_results['KKT4']),
                KKT5=jax.block_until_ready(solve_results['KKT5']),
                KKT6=jax.block_until_ready(solve_results['KKT6']),
                inner_niter=jax.block_until_ready(solve_results['inner_fin_niter']),
                inner_dx=jax.block_until_ready(solve_results['inner_fin_dx_scaled']),
                inner_du=jax.block_until_ready(solve_results['inner_fin_du']),
                dl=jax.block_until_ready(solve_results['inner_fin_dl']),
                max_cp=jnp.max(jnp.abs(dofs_opt['phi'])),
                avg_cp=jnp.average(jnp.abs(dofs_opt['phi'])),
            )
    # ----- Calculating metrics and gradients
    if value_only: 
        out_dict = {}
        for metric_name_i in metric_name:
            f_metric = lambda x, y: get_quantity(metric_name_i)(dofs, y_to_qp(y))
            metric_result_i = f_metric(dofs_opt, y_dict_current)
            out_dict[metric_name_i] = {
                'value': metric_result_i
            }
            if verbose>0:
                jax.debug.print('Metric evaluated. {x} = {y}', x=metric_name_i, y=metric_result_i)
        return out_dict, qp, dofs_opt, solve_results
    
    # ''' Recover the l_k in the last iteration for dx_k/dy '''

    # First, we reproduce the augmented lagrangian objective l_k that 
    # led to the optimum.
    if unconstrained:
        def l_k(x, y): 
            f_obj, _, _, _, _ = f_g_ineq_h_eq_from_y(y)
            return f_obj(x_arr_to_dofs(x)) 
    else:  
        c_k = solve_results['inner_fin_c']
        lam_k = solve_results['inner_fin_lam']
        mu_k = solve_results['inner_fin_mu']
        # @jit
        def l_k(x, y): 
            f_obj, g_ineq, h_eq, _, _ = f_g_ineq_h_eq_from_y(y)
            f_scaled = lambda x_scaled, f_obj=f_obj: f_obj(x_arr_to_dofs(x_scaled))
            g_scaled = lambda x_scaled, g_ineq=g_ineq: g_ineq(x_arr_to_dofs(x_scaled))
            h_scaled = lambda x_scaled, h_eq=h_eq: h_eq(x_arr_to_dofs(x_scaled))
            gplus = lambda x, mu, c: jnp.max(jnp.array([g_scaled(x), -mu/c]), axis=0)
            return(
                f_scaled(x) 
                + lam_k@h_scaled(x) 
                + c_k/2 * (
                    jnp.sum(h_scaled(x)**2) 
                    + jnp.sum(gplus(x, mu_k, c_k)**2)
                )
            )
    nabla_x_l_k = jacrev(l_k, argnums=0)
    nabla_y_l_k = jacrev(l_k, argnums=1)
    nabla_y_l_k_for_hess = lambda x: nabla_y_l_k(x, y_dict_current)
    hess_l_k = jacrev(nabla_x_l_k)(x_opt, y_dict_current)
    out_dict = {}
    for metric_name_i in metric_name:
        f_metric = lambda x, y: get_quantity(metric_name_i)(y_to_qp(y), x_arr_to_dofs(x))
        nabla_x_f = jacrev(f_metric, argnums=0)(x_opt, y_dict_current)
        nabla_y_f = jacrev(f_metric, argnums=1)(x_opt, y_dict_current)
        vihp = jnp.linalg.solve(hess_l_k, nabla_x_f)
        # Now we calculate df/dy using vjp
        # \nabla_{x_k} f [-H(l_k, x_k)^-1 \nabla_{x_k}\nabla_{y} l_k]
        # Primal and tangent must be the same shape
        _, dfdy1 = jvp(nabla_y_l_k_for_hess, primals=[x_opt], tangents=[vihp])
        # \nabla_{y} f
        dfdy2 = nabla_y_f
        # This was -dfdy1 + dfdy2 in the old code where
        # y is an array. Now y is a dict, and 
        # dfdy1, dfdy2 are both dicts.
        dfdy = {}
        for key in dfdy1.keys():
            if dfdy1[key] is not None and dfdy2[key] is not None:
                dfdy['df_d' + key] = -jnp.array(dfdy1[key]) + jnp.array(dfdy2[key])
                dfdy['df_d' + key + '_1'] = -jnp.array(dfdy1[key])
                dfdy['df_d' + key + '_vihp'] = vihp
                dfdy['df_d' + key + '_nabla_x_f'] = nabla_x_f
                dfdy['df_d' + key + '_nabla_y_f'] = nabla_y_f[key]
                dfdy['df_d' + key + '_hess_l_k'] = hess_l_k
                dfdy['df_d' + key + '_2'] = jnp.array(dfdy2[key])
        metric_result_i = f_metric(x_opt, y_dict_current)
        if verbose>0:
            jax.debug.print('Metric evaluated. {x} = {y}', x=metric_name_i, y=metric_result_i)
        out_dict[metric_name_i] = {
            'value': metric_result_i, 
            'grad': dfdy
        }

    # # # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # # # TESTING IMPLICIT FUNC THM ON THE ACTUAL LAGRANGIAN
    
    # # # Implicit derivative for l_full
    # # nabla_x_l_full = jacrev(l_full, argnums=0)
    # # nabla_y_l_full = jacrev(l_full, argnums=1)
    # # nabla_y_l_full_for_hess = lambda x: nabla_y_l_full(x, y_dict_current)
    # # hess_l_full = jacrev(nabla_x_l_full)(x_eff, y_dict_current)
    # # # out_dict = {}
    # # # x_eff consists of (x, z, mu, lam)
    # # x_eff = cp_mnprint
    # # if g_num > 0:
    # #     x_eff = jnp.concatenate([x_eff, jnp.zeros(g_num), mu_k]) # Adding z and mu
    # # if h_num > 0:
    # #     x_eff = jnp.concatenate([x_eff, lam_k]) # Adding lam
    # # for metric_name_i in metric_name:
    # #     f_metric = lambda x, y: get_quantity(metric_name_i)(y_to_qp(y), x)
    # #     nabla_x_f = jacrev(f_metric, argnums=0)(x_eff, y_dict_current)
    # #     nabla_y_f = jacrev(f_metric, argnums=1)(x_eff, y_dict_current)
    # #     vihp = jnp.linalg.solve(hess_l_full, nabla_x_f)
    # #     # Now we calculate df/dy using vjp
    # #     # \nabla_{x_k} f [-H(l_full, x_k)^-1 \nabla_{x_k}\nabla_{y} l_full]
    # #     # Primal and tangent must be the same shape
    # #     _, dfdy1 = jvp(nabla_y_l_full_for_hess, primals=[x_eff], tangents=[vihp])
    # #     # \nabla_{y} f
    # #     dfdy2 = nabla_y_f
    # #     # This was -dfdy1 + dfdy2 in the old code where
    # #     # y is an array. Now y is a dict, and 
    # #     # dfdy1, dfdy2 are both dicts.
    # #     dfdy = {}
    # #     for key in dfdy1.keys():
    # #         if dfdy1[key] is not None and dfdy2[key] is not None:
    # #             dfdy['df_d' + key] = -jnp.array(dfdy1[key]) + jnp.array(dfdy2[key])
    # #     metric_result_i = f_metric(x_eff, y_dict_current)
    # #     if verbose>0:
    # #         jax.debug.print('Metric evaluated. {x} = {y}', x=metric_name_i, y=metric_result_i)
    # #         jax.debug.print('Metric evaluated2. {x} = {y}', x=metric_name_i, y=get_quantity(metric_name_i)(qp, x_eff))
    # #     out_dict[metric_name_i]['grad2'] = dfdy
    # #     # out_dict[metric_name_i] = {
    # #     #     'value': metric_result_i, 
    # #     #     'grad': dfdy
    # #     # }
    
    return(out_dict, qp, dofs_opt, solve_results)


