from quadcoil import (
    merge_callables, get_quantity,
    gen_winding_surface_arc, 
    gen_winding_surface_atan,
    gen_winding_surface_offset,
    SurfaceRZFourierJAX, QuadcoilParams, 
    solve_constrained, run_opt_lbfgs,
    is_ndarray, tree_len,
    gplus_hard, gplus_elu, gplus_softplus
)
from quadcoil.wrapper import _parse_objectives, _parse_constraints
from functools import partial
from quadcoil.quantity import Bnormal
from jax import jacfwd, jacrev, jvp, jit, hessian, block_until_ready, config, debug, flatten_util, eval_shape
import jax.numpy as jnp
import lineax as lx
config.update('jax_enable_x64', True)

tol_default = 1e-6
tol_default_last = 1e-10
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
    # - Metrics
    'metric_name',
    # - Solver options
    'convex',
    'maxiter_tot',
    'maxiter_inner',
    # 'maxiter_inner_last',
    'gplus_mask',
    'implicit_linear_solver',
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
    phi_init=None, 
    # Current potential's normalization constant. 
    # By default will be generated from net total current.
    phi_unit=None,
    
    # - Plasma parameters
    plasma_quadpoints_phi=None,
    plasma_quadpoints_theta=None,
    Bnormal_plasma=None,

    # - Winding parameters (offset)
    plasma_coil_distance:float=None,
    winding_surface_generator=gen_winding_surface_offset, # gen_winding_surface_arc,

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
    objective_weight=1.,
    objective_unit=None,
    # - Quadcoil constraints
    constraint_name=(),
    constraint_type=(),
    constraint_unit=(),
    constraint_value=jnp.array([]),
    # - Metrics to study
    metric_name=('f_B', 'f_K'),

    # - Solver options
    convex=False,
    c_init:float=1.,
    c_growth_rate:float=2.,
    xstop_outer:float=tol_default, # convergence rate tolerance
    # gtol_outer:float=1e-7, # gradient tolerance
    ctol_outer:float=tol_default, # constraint tolerance
    # was 0., but we change this because we changed the logic to req x, u, g 
    # convergence rate to all be smaller than the thres, because sometimes small x results in 
    # large change in f.
    fstop_inner:float=tol_default,
    xstop_inner:float=tol_default,
    gtol_inner:float=tol_default,
    fstop_inner_last:float=0.,
    xstop_inner_last:float=tol_default_last,
    gtol_inner_last:float=tol_default_last,
    svtol:float=tol_default,
    maxiter_tot:int=10000,
    maxiter_inner:int=1000,
    gplus_mask=gplus_hard, # gplus_elu,
    implicit_linear_solver=lx.AutoLinearSolver(well_posed=True),
    value_only=False,
    verbose=0,
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
    phi_init : ndarray, optional, default=None
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
    convex : bool, optional, default=False
        (Static) Whether to assume the problem is convex. When ``True``, QUADCOIL will apply some limited simplifications.
    c_init : float, optional, default=1.
        (Traced) The initial :math:`c` factor. Please see *Constrained Optimization and Lagrange Multiplier Methods* Chapter 3.
    c_growth_rate : float, optional, default=1.2
        (Traced) The growth rate of the :math:`c` factor.
    xstop_outer : float, optional, default=1e-7
        (Traced) ``x`` convergence rate of the outer augmented 
        Lagrangian loop. Terminates when ``dx`` falls below this. 
    ctol_outer : float, optional, default=1e-7
        (Traced) Tolerance of the constraint KKT conditions in the outer
        Lagrangian loop. 
    fstop_inner, fstop_inner_last : float, optional, default=1e-7
        (Traced) ``f`` convergence rate of the inner LBFGS 
        Lagrangian loop. Terminates when ``df`` falls below this. 
    xstop_inner, xstop_inner_last : float, optional, default=0
        (Traced) ``x`` convergence rate of the outer augmented 
        Lagrangian loop. Terminates when ``dx`` falls below this. 
    gtol_inner, gtol_inner_last : float, optional, default=0.1
        (Traced) Gradient tolerance of the inner LBFGS iteration, normalized by the starting gradient.
    svtol : float, optional, default=0.1
        (Traced) Singular-value cut-off threshold during the pre-conditioning. Will treat 
        singular values smaller than ``svtol * jnp.max(s)`` as 0
    maxiter_tot : int, optional, default=50.
        (Static) The maximum of the outer iteration.
    maxiter_inner, maxiter_inner_last : int, optional, default=500
        (Static) The maximum of the inner iteration.
    gplus_mask : Callable, optional, default=quadcoil.gplus_hard
        (Static) The form of g+. Soft thresholding may improve derivative effectiveness.
    implicit_linear_solver : lineax.AbstractLinearSolver, optional, default=lineax.AutoLinearSolver(well_posed=True)
        (Static) The lineax linear solver choice for implicit differentiation.
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
    if isinstance(metric_name, str):
        metric_name = (metric_name,)
    # Type checking and error throwing
    _input_checking(
        objective_name=objective_name,
        objective_weight=objective_weight,
        objective_unit=objective_unit,
        constraint_name=constraint_name,
        constraint_type=constraint_type,
        constraint_unit=constraint_unit,
        constraint_value=constraint_value,
    )
    # A dictionary containing all parameters that the problem depends on.
    # These elements will always be in y.
    y_dict_current = {
        'plasma_dofs': plasma_dofs,
        'net_poloidal_current_amperes': net_poloidal_current_amperes,
        'net_toroidal_current_amperes': net_toroidal_current_amperes,
    }
    if not isinstance(objective_name, str):
        y_dict_current['objective_weight'] = jnp.array(objective_weight)
    if len(constraint_name) > 0:
        y_dict_current['constraint_value'] = constraint_value
    # Only differentiate wrt normal field when 
    # it's not zero.
    if Bnormal_plasma is not None:
        if verbose>0:
            debug.print('Maximum Bnormal_plasma: {x}', x=jnp.max(jnp.abs(Bnormal_plasma)))
        y_dict_current['Bnormal_plasma'] = Bnormal_plasma
    # Include winding dofs when it's provided.
    if plasma_coil_distance is None:
        if verbose>0:
            debug.print('Using custom winding surface.')
        y_dict_current['winding_dofs'] = winding_dofs
    else:
        if verbose>0:
            debug.print('Plasma-coil distance (m): {x}', x=plasma_coil_distance)
        y_dict_current['plasma_coil_distance'] = plasma_coil_distance
    
    # ----- Printing inputs -----
    if verbose>0:
        debug.print(
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
            '    xstop_outer: {xstop_outer}\n'\
            # '    gtol_outer: {gtol_outer}\n'\
            '    ctol_outer: {ctol_outer}\n'\
            '    fstop_inner: {fstop_inner}\n'\
            '    xstop_inner: {xstop_inner}\n'\
            '    gtol_inner: {gtol_inner}\n'\
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
            xstop_outer=xstop_outer,
            # gtol_outer=gtol_outer,
            ctol_outer=ctol_outer,
            fstop_inner=fstop_inner,
            xstop_inner=xstop_inner,
            gtol_inner=gtol_inner,
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
        if 'objective_weight' in y_dict:
            objective_weight_temp = y_dict['objective_weight']
        else:
            objective_weight_temp = 1.
        if 'constraint_value' in y_dict:
            constraint_value_temp = y_dict['constraint_value']
        else:
            constraint_value_temp = []
        f_obj, g_obj_list, h_obj_list, aux_dofs_obj = _parse_objectives(
            objective_name=objective_name, 
            objective_unit=objective_unit,
            objective_weight=objective_weight_temp, 
        )
        g_cons_list, h_cons_list, aux_dofs_cons = _parse_constraints(
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
            constraint_value=constraint_value_temp,
        )
        # Merging constraints and aux dofs from different sources
        g_list = g_obj_list + g_cons_list
        h_list = h_obj_list + h_cons_list
        aux_dofs_init = aux_dofs_obj | aux_dofs_cons

        f_obj_x = lambda x, qp_temp=qp_temp, f_obj=f_obj: f_obj(qp_temp, x)
        g_ineq_x = lambda x, qp_temp=qp_temp, g_list=g_list: merge_callables(g_list)(qp_temp, x)
        h_eq_x = lambda x, qp_temp=qp_temp, h_list=h_list: merge_callables(h_list)(qp_temp, x)
        n_g = len(g_list)
        n_h = len(h_list)
        return f_obj_x, g_ineq_x, h_eq_x, n_g, n_h, aux_dofs_init

    
    # ----- Creating Initializing phi -----
    # Defining a shared problem parameter object
    qp = y_to_qp(y_dict_current)
    # f, g, h are Callable(qp, {'phi':, ..., })
    # i.e., they accepts unscaled input
    f_obj, g_ineq, h_eq, n_g, n_h, aux_dofs_init = f_g_ineq_h_eq_from_y(y_dict_current)
    unconstrained = ((n_g == 0) and (n_h == 0))

    if phi_init is None:
        phi_init = jnp.zeros(qp.ndofs)
    # not really used in initialization, but used 
    # to calculate phi scaling, the initial value 
    # of lam and mu, and the initial value of aux 
    # variables.
    dofs_dict_init = {'phi': phi_init}
    # ----- Calculating the unit of phi -----
    # phi need to be normalized to ~1 for the optimizer to behave well.
    # by default we do this using the initial value of Bnormal
    if phi_unit is None:
        # Scaling current potential dofs to ~1
        # By default, we use the Bnormal value when 
        # phi=0 to generate this scaling factor.
        B_normal_estimate = jnp.average(jnp.abs(Bnormal(qp, dofs_dict_init))) # Unit: T
        if plasma_coil_distance is not None:
            phi_unit = B_normal_estimate * 1e7 * jnp.abs(plasma_coil_distance)
        else:
            # The minor radius can be estimated from the 
            # n=0, m=1 rc mode of the surface.
            plasma_minor = plasma_dofs[plasma_ntor*2 + 1]
            winding_minor = winding_dofs[winding_ntor*2 + 1]
            phi_unit = B_normal_estimate * 1e7 * jnp.abs(plasma_minor - winding_minor)

    # ----- Creating scaled, flattened dof, 'x_flat_init' -----
    # The actual, unit-free, variable used for initialization,
    # and by the optimizer. The dof that the optimizer operates on is a
    # flattened version of this dictionary.
    x_dict = {
       'phi_scaled': phi_init/phi_unit,
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
            x_dict[key] = aux_dofs_init[key](qp, {'phi': phi_init})
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
    # Here we perform the unraveling. 
    # *** x_flat_init is the actual dof manipulated by the optimizers! ***
    x_flat_init, unravel_x = flatten_util.ravel_pytree(x_dict)
    ndofs_tot = len(x_flat_init) # This counts the aux vars too
    ny = tree_len(y_dict_current)
    # This block prints out a summary on the auxiliary vars and 
    # phi degrees of freedom.
    def unravel_unscale_x(x, unravel_x=unravel_x, phi_unit=phi_unit):
        d = unravel_x(x)
        # Replace scaled phi with regular phi
        # after unraveling for passing into 
        # f_obj, g_ineq and h_eq.
        dofs_temp = {k: v for k, v in {**d, "phi": d["phi_scaled"] * phi_unit}.items() if k != "phi_scaled"}
        return(dofs_temp)
    # ----- Scaling f, g, h and initializing mu and lam -----
    # f, g and h should take x_flat_init, the flattened, scaled dofs.
    # *** f_scaled, g_scaled and h_scaled are the actual functions 
    # seen by the optimizer! ***
    f_scaled = lambda x_scaled, f_obj=f_obj: f_obj(unravel_unscale_x(x_scaled))
    g_scaled = lambda x_scaled, g_ineq=g_ineq: g_ineq(unravel_unscale_x(x_scaled))
    h_scaled = lambda x_scaled, h_eq=h_eq: h_eq(unravel_unscale_x(x_scaled))
    
    mu_init = jnp.zeros(eval_shape(g_scaled, x_flat_init).shape)
    lam_init = jnp.zeros(eval_shape(h_scaled, x_flat_init).shape)
    
    # ----- Summarizing initialization -----
    if verbose>0:
        dofs_summary = []
        for key, value in x_dict.items():
            dofs_summary.append(f"    {key}: {jnp.atleast_1d(value).shape}")
        final_str = "\n".join(dofs_summary)
        debug.print(
            '----- DOF summary ----- \n'\
            'After converting non-smooth terms (such as |f|) into\n'\
            'smooth terms, auxiliary vars and constraints, the dofs are:\n{s}\n'\
            'Total # dofs (including auxiliary): {t}\n'\
            'Shape of mu, lam: {mu}, {lam}\n'\
            'Total # of ineq constraint quantities (can have array output): {n_g}\n'\
            'Total # of eq constraint quantities (can have array output): {n_h}\n'\
            'Total # problem parameters: {u}',
            mu=mu_init.shape, lam=lam_init.shape,
            s=final_str, t=ndofs_tot, u=ny, n_g=n_g, n_h=n_h
        )
    
    # ----- Solving QUADCOIL -----
    # A dictionary containing augmented lagrangian info
    # and the last augmented lagrangian objective function for 
    # implicit differentiation.
    # When unconstrained, this function instead serves the 
    # purpose of "zooming in" when iteration step lengths
    # are small.
    if unconstrained:
        x_flat_opt, val_l_k, grad_l_k, niter_inner_k, dx_k, du_k, dL_k = run_opt_lbfgs(
            init_params=x_flat_init,
            fun=f_scaled,
            maxiter=maxiter_tot,
            fstop=fstop_inner_last,
            xstop=xstop_inner_last,
            gtol=gtol_inner,
            verbose=verbose,
        )
        dofs_opt = unravel_unscale_x(x_flat_opt)
        solve_results = {
            'inner_fin_f': val_l_k,
            'inner_fin_x': x_flat_opt,
            'inner_fin_niter': niter_inner_k,
            'inner_fin_dx_scaled': dx_k,
            'inner_fin_du': du_k,
            'inner_fin_df': dL_k,
            # The scaling factor for the next iteration
            # 'x_unit': jnp.average(jnp.abs(x_k)),
        }
        if verbose>0:       
            debug.print(
                '----- Solver status summary -----\n'\
                'Final value of objective f: {f}\n'\
                'Final Max current potential (dipole density): {max_cp} (A)\n'\
                'Final Avg current potential (dipole density): {avg_cp} (A)\n'\
                '* Total L-BFGS iteration number: {niter}\n'\
                '    Phi scaling constant:  {x_unit_init}(A)\n'\
                '    Inner convergence rate in x (scaled): {inner_dx}, {inner_du}\n'\
                '    Inner convergence rate in f: {df}\n',
                f=val_l_k,
                niter=niter_inner_k,
                x_unit_init=phi_unit,
                inner_dx=block_until_ready(solve_results['inner_fin_dx_scaled']),
                inner_du=block_until_ready(solve_results['inner_fin_du']),
                df=block_until_ready(solve_results['inner_fin_df']),
                max_cp=jnp.max(jnp.abs(dofs_opt['phi'])),
                avg_cp=jnp.average(jnp.abs(dofs_opt['phi'])),
            )
    else:
        solve_results = solve_constrained(
            x_init=x_flat_init,
            f_obj=f_scaled,
            lam_init=lam_init,
            mu_init=mu_init,
            h_eq=h_scaled,
            g_ineq=g_scaled,
            c_init=c_init,
            c_growth_rate=c_growth_rate,
            ctol_outer=ctol_outer,
            xstop_outer=xstop_outer,
            # gtol_outer=gtol_outer,
            fstop_inner=fstop_inner,
            xstop_inner=xstop_inner,
            gtol_inner=gtol_inner,
            fstop_inner_last=fstop_inner_last,
            xstop_inner_last=xstop_inner_last,
            gtol_inner_last=gtol_inner_last,
            maxiter_tot=maxiter_tot,
            maxiter_inner=maxiter_inner,
            verbose=verbose
        )
        # The optimum, unit-less.
        x_flat_opt = solve_results['inner_fin_x']
        dofs_opt = unravel_unscale_x(x_flat_opt)
        if verbose>0:       
            debug.print(
                '----- Solver status summary -----\n'\
                'Final value of f(scaled): {fs}\n'\
                'Final Max current potential (dipole density): {max_cp} (A)\n'\
                'Final Avg current potential (dipole density): {avg_cp} (A)\n'\
                '* Total L-BFGS iteration number: {niter}\n'\
                '    Phi scaling constant:  {x_unit_init}(A)\n'\
                '    Final max constraint g violation(scaled): {g}\n'\
                '    Final max constraint h violation(scaled): {h}\n'\
                '    Outer convergence rate in x (scaled): {dx}\n'\
                '* Last inner L_BFGS iteration number: {inner_niter}\n'\
                '    Inner convergence rate in x (scaled): {inner_dx}, {inner_du}\n'\
                '    Inner convergence rate in l: {dl}\n',
                fs=block_until_ready(solve_results['inner_fin_f']),
                niter=block_until_ready(solve_results['tot_niter']),
                g=block_until_ready(_print_max_blank(solve_results['inner_fin_g'])),
                h=block_until_ready(_print_max_blank(jnp.abs(solve_results['inner_fin_h']))),
                x_unit_init=phi_unit,
                dx=block_until_ready(solve_results['outer_dx']),
                inner_niter=block_until_ready(solve_results['inner_fin_niter']),
                inner_dx=block_until_ready(solve_results['inner_fin_dx_scaled']),
                inner_du=block_until_ready(solve_results['inner_fin_du']),
                dl=block_until_ready(solve_results['inner_fin_dl']),
                max_cp=jnp.max(jnp.abs(dofs_opt['phi'])),
                avg_cp=jnp.average(jnp.abs(dofs_opt['phi'])),
            )
    # ----- Calculating metrics and gradients
    if value_only: 
        out_dict = {}
        for metric_name_i in metric_name:
            metric_result_i = get_quantity(metric_name_i)(qp, dofs_opt)
            out_dict[metric_name_i] = {
                'value': metric_result_i
            }
            if verbose>0:
                debug.print('Metric evaluated. {x} = {y}', x=metric_name_i, y=metric_result_i)
        return out_dict, qp, dofs_opt, solve_results
    # flatten the y dictionary. This will simplify the code structure a bit
    y_flat, unravel_y = flatten_util.ravel_pytree(y_dict_current)
    out_dict = {}
    
    # ----- Stationarity conditions -----
    # It will be prohibitively expensive to solve the KKT condition. 
    # Therefore, we use the Jacobian of the unconstrained objective instead.
    if unconstrained:
        def l_k(x, y): 
            f_obj, _, _, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y))
            return f_obj(unravel_unscale_x(x)) 
        # No need in preconditioning x.
        # for more detail, see Step-1 preconditioning.
        x_flat_precond = x_flat_opt
        xp_to_x = lambda xp: xp
        # 
        grad_x_l_k = jacrev(l_k, argnums=0)
        # When the problem is unconstrained, 
        # we can avoid materializing the full Hessian.
        if convex:
            vihp_A_precond = lx.JacobianLinearOperator(
                grad_x_l_k, 
                x_flat_opt, args=y_flat, 
                tags=(lx.symmetric_tag, lx.positive_semidefinite_tag)
            )
        else:
            vihp_A_precond = lx.JacobianLinearOperator(
                grad_x_l_k, 
                x_flat_opt, args=y_flat,
                tags=(lx.symmetric_tag)
            )
        if verbose>0:
            hess_l_k = jacrev(grad_x_l_k)(x_flat_opt, y_flat)
            hess_cond = jnp.linalg.cond(hess_l_k)
            out_dict['hess_cond'] = hess_cond
            debug.print('Unconstrained Hessian condition number: {x}', x=hess_cond)
    else:  
        # When solving a constrained optimization problem, an important source of 
        # ill-conditioning is that the three terms in l_k can have drastically different
        # orders of magnitudes. This block of code performs the pre-conditioning.
        c_k = solve_results['inner_fin_c']
        mu_k = solve_results['inner_fin_mu']
        lam_k = solve_results['inner_fin_lam']
        # The pre-conditioning requires that us treat the three 
        # terms in l_k separately.
        def l_k_terms(x, y=y_flat, mu=mu_k, lam=lam_k, c=c_k): 
        # def l_k_terms_raw(x, y=y_flat, mu=mu_k, lam=lam_k, c=c_k): 
            f_obj_temp, g_ineq_temp, h_eq_temp, _, _, _ = f_g_ineq_h_eq_from_y(unravel_y(y))
            f_scaled_temp = lambda x_flat: f_obj_temp(unravel_unscale_x(x_flat))
            g_scaled_temp = lambda x_flat: g_ineq_temp(unravel_unscale_x(x_flat))
            h_scaled_temp = lambda x_flat: h_eq_temp(unravel_unscale_x(x_flat))
            gplus_temp = partial(gplus_mask, g_ineq=g_scaled_temp)
            # gplus_temp = lambda x, mu, c: jnp.max(jnp.array([g_scaled_temp(x), -mu/c]), axis=0)
            # gplus = lambda x, mu, c: g_scaled_temp(x)
            return jnp.array([
                f_scaled_temp(x),
                lam@h_scaled_temp(x) + mu@gplus_temp(x, mu, c),
                c/2 * (
                    jnp.sum(h_scaled_temp(x)**2) 
                    + jnp.sum(gplus_temp(x, mu, c)**2)
                )
            ])
        # For calculating grad_y_l_k
        l_k = lambda x, y=y_flat, mu=mu_k, lam=lam_k, c=c_k: jnp.sum(l_k_terms(x=x, y=y, mu=mu, lam=lam, c=c))
        # hess_l_k = hessian(l_k)(x_flat_opt)
        x_flat_precond = x_flat_opt
        xp_to_x = lambda xp: xp
        # # ----- Step-1 preconditioning -----
        # l_k_raw = lambda x, y=y_flat, mu=mu_k, lam=lam_k, c=c_k: jnp.sum(l_k_terms_raw(x=x, y=y, mu=mu_k, lam=lam_k, c=c_k))
        # hess_l_k_raw = hessian(l_k_raw)(x_flat_opt)
        # # As the first step of the pre-conditioning, we re-define 
        # # l_k and l_k_terms under a changed coordinate based 
        # # on the SVD of l_k's hessian. This reduce rounding error 
        # # during the autodiff process, and improve the conditioning of 
        # # all three terms in l_k. 
        # # First, we generate the coordinate transform.
        # x_to_xp, xp_to_x = _precondition_coordinate_by_matrix(hess_l_k_raw)
        # # We replace x_flat with its new definition after pre-conditioning.
        # # We've already unraveled it before this point, so it's okay to replace
        # # the variable.
        # x_flat_precond = x_to_xp(x_flat_opt)
        # # Redefining l_k and l_k_terms. All autodiff will be done 
        # # with these instead.
        # l_k_terms = lambda xp, y=y_flat: l_k_terms_raw(xp_to_x(xp), y=y)
        # l_k = lambda xp, y: jnp.sum(l_k_terms(xp=xp, y=y))
        # ----- Step-2 preconditioning -----
        # An important source of ill-conditioning in Hess(l_k)
        # is the difference in the three terms' orders of magnitude.
        # Often, each of these terms are singular by themselves, 
        # but adds up to a non-singular Hess(l_k).
        # The goal of pre-conditioning is to 
        # 1. Sort the three Hessians based on the magnitude of their 
        # SV's, in ascending order as A, B and C.
        # 2. Stretch B in directions where it's linearly indep from C.
        # 3. Stretch A in directions where it's linearly indep from B and C.
        # Because these are symmetric matrices, 
        hess_l_k_terms_val = hessian(l_k_terms)(x_flat_precond)
        hess_l_k = jnp.sum(hess_l_k_terms_val, axis=0)
        # Symmetrizing
        hess_l_k_terms_val = 0.5 * (
            hess_l_k_terms_val
            + jnp.swapaxes(hess_l_k_terms_val, 1, 2)
        )
        # U_i = V_i.
        # (or U - VH.T = 0)
        U, s, VH = jnp.linalg.svd(hess_l_k_terms_val)
        # We use the maximum SV as an estimate of the 
        # order of magnitude of a matrix
        s_max = jnp.max(s, axis=1)
        # A 3 x ndofs boolean array that 
        # selects singular values bigger than machine 
        # precision * s_max.
        s_selection = s >= svtol * s_max[:, None]
        # We now sort the matrices by their orders of magnitude
        # We'll refer to the matrices in ascending order as 
        # A, B, C
        hess_order = jnp.argsort(s_max)
        # We first project B's basis' onto C's basis and then remove the projection
        # from B's basis'. This gives us the "component" of B that are impossible 
        # to represent with C's basis.
        A = hess_l_k_terms_val[hess_order[0]]
        B = hess_l_k_terms_val[hess_order[1]]
        C = hess_l_k_terms_val[hess_order[2]]
        VH_A = VH[hess_order[0]]
        VH_B = VH[hess_order[1]]
        VH_C = VH[hess_order[2]]
        s_selection_B = s_selection[hess_order[1]]
        s_selection_C = s_selection[hess_order[2]]
        proj_C  = VH_C.T @ (   s_selection_C[:, None] * VH_C)
        proj_B  = VH_B.T @ (   s_selection_B[:, None] * VH_B)
        annil_C = VH_C.T @ ((~s_selection_C)[:, None] * VH_C)
        # annil_C = jnp.identity(len(x_flat_precond)) - proj_C
        # We now calculate the basis spanned by B abd C combined
        U_BC, s_BC, VH_BC = jnp.linalg.svd(jnp.concatenate([proj_B, proj_C]))
        s_BC_selection = s_BC >= svtol * jnp.max(s_BC)
        # annil_BC and annil_C removes the components spanned by BC and C's basis
        # proj_BC and proj_C projects a vector in BC and C's basis
        proj_BC  = VH_BC.T @ (  s_BC_selection [:, None] * VH_BC)
        annil_BC = VH_BC.T @ ((~s_BC_selection)[:, None] * VH_BC)
        # This where statement is here in case A (or A and B)'s Hessian is rank-0
        scale_AC = jnp.where(s_max[hess_order[0]]>0, s_max[hess_order[2]] / s_max[hess_order[0]], 0)
        scale_BC = jnp.where(s_max[hess_order[1]]>0, s_max[hess_order[2]] / s_max[hess_order[1]], 0)
        # The appropriate pre-conditioner is:
        # O \equiv \[\epsilon^-2 (I-P_{BC}) + \epsilon^-1 P_{BC}\](I-P_C) +P_C
        OC = C
        OB = scale_BC * proj_BC @ annil_C @ B + proj_C @ B 
        OA = (
            scale_AC * annil_BC @ annil_C @ A
            + scale_BC * proj_BC @ annil_C @ A
            + proj_C @ A
        )
        Ohess = OA + OB + OC
        
        vihp_A_raw = lx.MatrixLinearOperator(hess_l_k)
        vihp_A_precond = lx.MatrixLinearOperator(Ohess)
        if verbose>0:
            hess_rank = jnp.linalg.matrix_rank(A + B + C)
            Ohess_rank = jnp.linalg.matrix_rank(OA + OB + OC)
            hess_cond = jnp.linalg.cond(A + B + C)
            Ohess_cond = jnp.linalg.cond(OA + OB + OC)
            # out_dict['hess_cond'] = hess_cond
            # out_dict['hess_cond_preconditioned'] = Ohess_cond
            debug.print(
                'Info on Hessian terms (unsorted)\n'\
                '    Rank of term 1, 2 and 3: {a1}\n'\
                '    Max sv of 1, 2 and 3: {a2}\n'\
                'Info on Hessian terms (sorted)\n'\
                '    Rank of A, B and C: {a}\n'\
                '    Max sv of A, B and C: {aa}\n'\
                '    Rank of OA, OB and OC: {b}\n'\
                '    scale_AC and scale_BC: {bb}\n'\
                '    Rank of proj_BC and annil_BC: {c}\n'\
                '    Rank of proj_C  and annil_C:  {d}\n'\
                '    Constrained Hessian rank and condition number, before pre-conditioning: {x}, {x1}\n'\
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
    grad_y_l_k = jacrev(l_k, argnums=1)
    grad_y_l_k_for_hess = lambda x, y_flat=y_flat: grad_y_l_k(x, y_flat)
    for metric_name_i in metric_name:
        f_metric = lambda xp, y: get_quantity(metric_name_i)(
            y_to_qp(unravel_y(y)), 
            unravel_unscale_x(xp_to_x(xp))
        )
        grad_x_f = jacrev(f_metric, argnums=0)(x_flat_precond, y_flat)
        grad_y_f = jacrev(f_metric, argnums=1)(x_flat_precond, y_flat)
        if unconstrained:
            vihp = lx.linear_solve(
                vihp_A_precond, # should be .T but hessian is symmetric 
                grad_x_f,
            ).value
        else:
            vihp_b = (
                scale_AC * annil_BC @ annil_C @ grad_x_f
                + scale_BC * proj_BC @ annil_C @ grad_x_f
                + proj_C @ grad_x_f
            )
            # TODO
            # It is somewhat hard to tell whether pre-conditioning 
            # improves accuracy or introduces additional error 
            # just from A and b. Therefore, we compute both with
            # and without pre-conditioning, and pick the option with 
            # the less error. Is there a way to improve this?
            vihp_raw = lx.linear_solve(
                vihp_A_raw, # should be .T but hessian is symmetric 
                grad_x_f,
                solver=implicit_linear_solver
            ).value
            vihp_precond = lx.linear_solve(
                vihp_A_precond, # should be .T but hessian is symmetric 
                vihp_b,
                solver=implicit_linear_solver
            ).value
            hess_err = jnp.linalg.norm(hess_l_k @ vihp_raw - grad_x_f)
            Ohess_err = jnp.linalg.norm(hess_l_k @ vihp_precond - grad_x_f)
            vihp = jnp.where(hess_err < Ohess_err, vihp_raw, vihp_precond)
            
        # Now we calculate df/dy using vjp
        # \grad_{x_k} f [-H(l_k, x_k)^-1 \grad_{x_k}\grad_{y} l_k]
        # Primal and tangent must be the same shape
        _, dfdy1 = jvp(grad_y_l_k_for_hess, primals=[x_flat_precond], tangents=[vihp])
        # \grad_{y} f
        dfdy2 = grad_y_f
        dfdy_arr = -dfdy1 + dfdy2
        dfdy_dict = {f"df_d{key}": value for key, value in unravel_y(dfdy_arr).items()}
        metric_result_i = f_metric(x_flat_precond, y_flat)
        if verbose>0:
            grad_y_l_k_val = grad_y_l_k(x_flat_precond, y_flat)
            grad_x_grad_y_l_k = jacfwd(grad_y_l_k_for_hess, argnums=0)(x_flat_precond, y_flat)
            grad_keys = dfdy_dict.keys()
            grad_avgs = {}
            for k in grad_keys:
                item_k = jnp.atleast_1d(dfdy_dict[k])
                grad_avgs[k] = (
                    jnp.min(item_k),
                    jnp.max(item_k)
                )
            debug.print(
                '* Metric evaluated.\n'
                '    {x} = {y}\n'
                '    VIHP        min max: {v1}, {v2}\n'
                '    df/dy       min max: {gy1}, {gy2}\n'
                '    df/dx       min max: {gx1}, {gx2}\n'
                '    df/dx dx/dy min max: {gxy1}, {gxy2}\n'
                '    dLk/dy min max: {a1}, {a2}\n'
                # '    d2Lk/dxdy min max: {b1}, {b2}\n'
                # '    VIHP error without pre-conditioning: {a}\n'
                # '    VIHP error with pre-conditioning:    {b}\n'
                '    Gradient min, max: {g}',
                v1=jnp.min(vihp),
                v2=jnp.max(vihp),
                gy1=jnp.min(grad_y_f),
                gy2=jnp.max(grad_y_f),
                gx1=jnp.min(grad_x_f),
                gx2=jnp.max(grad_x_f),
                gxy1=jnp.min(dfdy1),
                gxy2=jnp.max(dfdy1),
                a1=jnp.min(grad_y_l_k_val),
                a2=jnp.max(grad_y_l_k_val),
                # b1=jnp.min(grad_x_grad_y_l_k),
                # b2=jnp.max(grad_x_grad_y_l_k),
                x=metric_name_i, 
                y=metric_result_i,
                # a=hess_err,
                # b=Ohess_err,
                g=grad_avgs,
            )
        out_dict[metric_name_i] = {
            'value': metric_result_i, 
            'grad': dfdy_dict,
        }     
        if verbose>0:
            out_dict[metric_name_i]['vihp'] = vihp
            if not unconstrained:
                out_dict[metric_name_i]['hess_rank'] = hess_rank
                out_dict[metric_name_i]['Ohess_rank'] = Ohess_rank
                out_dict[metric_name_i]['hess_cond'] = hess_cond
                out_dict[metric_name_i]['Ohess_cond'] = Ohess_cond
                out_dict[metric_name_i]['hess_err'] = hess_err
                out_dict[metric_name_i]['Ohess_err'] = Ohess_err
    return(out_dict, qp, dofs_opt, solve_results)

def _choose_fwd_rev(func, n_in, n_out, argnums):
    '''
    Choosing forward or reverse-mode AD based on the input and 
    output size of a function.
    '''
    if n_out > n_in:
        out = jacfwd(func, argnums=argnums)
    else:
        out = jacrev(func, argnums=argnums)
    return out

def _precondition_coordinate_by_matrix(hess):
    '''
    Takes a symmetric matrix hess, calculates its SVD, 
    and returns two coordinate transform function, 
    x_to_xp and xp_to_x, so that 
    hess(f(x')) is more well-behaved than hess(f(x)).
    '''
    _, sv, basis = jnp.linalg.svd(hess)
    scale = jnp.sqrt(sv)
    x_to_xp = lambda x: (basis @ x) * scale
    xp_to_x = lambda xp: basis.T @ (xp / scale)
    return x_to_xp, xp_to_x


def _print_min_blank(a):
    return jnp.min(a) if a.size > 0 else jnp.nan

def _print_max_blank(a):
    return jnp.max(a) if a.size > 0 else jnp.nan

def _input_checking(
    objective_name,
    objective_weight,
    objective_unit,
    constraint_name,
    constraint_type,
    constraint_unit,
    constraint_value,
):
    
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
    else:
        objective_weight = 1.
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