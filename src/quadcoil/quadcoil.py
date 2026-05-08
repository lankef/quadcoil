from quadcoil import (
    merge_callables, get_quantity,
    SurfaceRZFourierJAX, SurfaceXYZTensorFourierJAX, SurfaceXYZFourierJAX,
    QuadcoilParams, 
    solve_constrained_auglag_lbfgs, solve_unconstrained_auglag_lbfgs,
    stationarity_auglag_lbfgs, adjoint_auglag_lbfgs,
    solve_constrained_ipm, solve_unconstrained_ipm,
    stationarity_ipm, adjoint_ipm,
    solve_constrained_slsqp, solve_unconstrained_slsqp,
    stationarity_slsqp, adjoint_slsqp,
    is_ndarray, tree_len,
)

from quadcoil.wrapper import _parse_objectives, _parse_constraints, _resolve_quadpoints
from functools import partial
from quadcoil.quantity import Bnormal
from jax import jacfwd, jacrev, jit, block_until_ready, debug, flatten_util, eval_shape
from jax import config as config_jax
import jax.numpy as jnp
import lineax as lx
import warnings
config_jax.update('jax_enable_x64', True)


tol_default = 1e-6
tol_default_last = 1e-10

SURFACE_TYPE_MAP = {
    'SurfaceRZFourier':        SurfaceRZFourierJAX,
    'SurfaceXYZTensorFourier': SurfaceXYZTensorFourierJAX,
    'SurfaceXYZFourier':       SurfaceXYZFourierJAX,
}

# Stores the default options for each solver.
SOLVER_OPTIONS_DEFAULT_DICT = {
    'auglag-lbfgs': {
        'c_init':           1.,
        'c_growth_rate':    2.,
        'xstop_outer':      tol_default,
        'ctol_outer':       tol_default,
        'atol_inner':       tol_default,
        'rtol_inner':       tol_default,
        'atol_inner_last':  tol_default_last,
        'rtol_inner_last':  tol_default_last,
        'svtol':            tol_default,
        'maxiter_tot':      10000,
        'maxiter_inner':    1000,
    },
    'ipm': {
        'tol_kkt': 1e-6,
        'max_ipm_iter': 100,
        'tau': 0.995,
        'delta_init': 1e-6,
        'delta_min': 1e-10,
        'delta_max': 1e-2,
    },
    'slsqp': {
        'max_steps': 200,
        'atol': 1e-7,
        'rtol': 1e-7,
    },
}

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
    'plasma_stellsym',
    # - WS options
    'surface_type',
    'offset_smoothing',
    'winding_mpol',
    'winding_ntor',
    'winding_stellsym',
    # - Objectives
    'objective_name',
    # - Constraints 
    'constraint_name',
    'constraint_type',
    # - Metrics
    'metric_name',
    # - Constraint handling and adjoint
    'value_only',
    'convex',
    'merge_constraints',
    'implicit_linear_solver',
    # - Solver options
    'solver',
    'lbfgs_memory',
    # - Other options
    'verbose',
    # Smoothing parameters:
    'smoothing',
]
@partial(jit, static_argnames=QUADCOIL_STATIC_ARGNAMES)
def quadcoil(
    # Now, the regular arguments.
    nfp:int, # Documented
    stellsym:bool, # Documented
    plasma_mpol:int, # Documented
    plasma_ntor:int, # Documented
    plasma_dofs, # Documented
    net_poloidal_current_amperes:float, # Documented
    
    # -- Defaults --
    
    # - Quadcoil parameters
    net_toroidal_current_amperes:float=0., # Documented
    mpol:int=6, # Documented
    ntor:int=4, # Documented
    # Quadpoints to evaluate objectives at
    quadpoints_phi=None, # Documented
    quadpoints_theta=None, # Documented
    phi_init=None,  # Documented
    # Current potential's normalization constant. 
    # By default will be generated from net total current.
    phi_unit=None, # Documented
    
    # - Plasma parameters
    plasma_stellsym=True, # Documented
    plasma_quadpoints_phi=None, # Documented
    plasma_quadpoints_theta=None, # Documented
    Bnormal_plasma=None, # Documented

    # - Winding parameters (offset)
    plasma_coil_distance:float=None, # Documented
    surface_type:str='SurfaceRZFourier', # Documented
    offset_smoothing:str='intersection', # Documented

    # - Winding parameters (Providing surface)
    winding_dofs=None, # Documented
    winding_mpol:int=6, # Documented
    winding_ntor:int=5, # Documented
    winding_quadpoints_phi=None, # Documented
    winding_quadpoints_theta=None, # Documented
    winding_stellsym=True, # Documented

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
    
    # - Constraint handling and adjoint
    value_only=False,
    smoothing='slack',
    smoothing_params={'lse_epsilon': 1e-3},
    convex:bool=False,

    # - Solver options
    # ess_alpha=1., # ESS factor, see Algorithm 2 of arxiv 2509.16320    
    verbose:int=0,
    merge_constraints:bool=False,

    # - Auglag options (traced dict; merged with SOLVER_OPTIONS_DEFAULT)
    solver:str='auglag-lbfgs',
    solver_options=None,
    lbfgs_memory:int=10,

    # - Experimental
    implicit_linear_solver=None,
):
    r'''
    Solves a QUADCOIL problem.

    Parameters
    ----------
    nfp : int
        (Static) The number of field periods.
    stellsym : bool
        (Static) Whether the coils have stellarator symmetry.
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
    plasma_stellsym : bool, default=True
        (Static) Whether the plasma has stellarator symmetry.
    plasma_quadpoints_phi : ndarray, shape (nphi_plasma,), optional, default=None
        (Traced) Will be set based on the shape of ``Bnormal_plasma`` if it's provided, 
        or default to ``jnp.linspace(0, 1/nfp, 32, endpoint=False)`` otherwise.
    plasma_quadpoints_theta : ndarray, shape (ntheta_plasma,), optional, default=None
        (Traced) Will be set based on the shape of ``Bnormal_plasma`` if it's provided, 
        or default to ``jnp.linspace(0, 1, 34, endpoint=False)`` otherwise.
    Bnormal_plasma : ndarray, shape (nphi, ntheta), optional, default=None
        (Traced) The magnetic field distribution on the plasma surface. Will be filled with zeros by default.
    plasma_coil_distance : float, optional, default=None
        (Traced) The coil-plasma distance. Is set to ``None`` by default, but a value must be provided if ``winding_dofs`` is not provided.
    surface_type : str, optional, default='SurfaceRZFourier'
        (Static) The surface parametrization. One of ``'SurfaceRZFourier'``,
        ``'SurfaceXYZTensorFourier'``, ``'SurfaceXYZFourier'``.
        Auto-offset (``plasma_coil_distance``) only works with ``'SurfaceRZFourier'``.
    offset_smoothing : str, optional, default='intersection'
        (Static) Self-intersection removal strategy when auto-generating the winding surface.
        One of ``'none'``, ``'intersection'``, or ``'hull'``.
    winding_dofs : ndarray, shape (ndof_winding,)
        (Traced) The winding surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` convention.
        Will be generated using ``offset_smoothing`` if ``plasma_coil_distance`` is provided. Must be provided otherwise.
    winding_mpol : int, optional, default=6
        (Static) The number of poloidal Fourier harmonics in the winding surface.
    winding_ntor : int, optional, default=5
        (Static) The number of toroidal Fourier harmonics in the winding surface.
    winding_quadpoints_phi : ndarray, shape (nphi_winding,), optional, default=None
        (Traced) Will be set to ``jnp.linspace(0, 1, 32*nfp, endpoint=False)`` by default.
    winding_quadpoints_theta : ndarray, shape (ntheta_winding,), optional, default=None
        (Traced) Will be set to ``jnp.linspace(0, 1, 34, endpoint=False)`` by default.
    winding_stellsym : bool, default=True
        (Static) Whether the winding surface has stellarator symmetry.
    objective_name : tuple, optional, default='f_B_normalized_by_Bnormal_IG'
        (Static) The names of the objective functions. Must be a member of ``quadcoil.objective`` that outputs a scalar.
    objective_weight : ndarray, optional, default=None
        (Traced) The weights of the objective functions. Derivatives will be calculated w.r.t. this quantity.
    objective_unit : ndarray, optional, default=None
        (Traced) The normalization constants of the objective terms, so that ``f/objective_unit`` is :math:`O(1)`. May contain ``None``
    constraint_name : tuple, optional, default=()
        (Static) The names of the constraint functions. Must be a member of ``quadcoil.objective`` that outputs a scalar.
    constraint_type : tuple, optional, default=()
        (Static) The types of the constraints. Must consist of ``'>='``, ``'<='``, ``'=='`` only.
    constraint_unit : ndarray, optional, default=()
        (Traced) The normalization constants of the constraints, so that ``f/constraint_unit`` is :math:`O(1)` May contain ``None``.
    constraint_value : ndarray, optional, default=()
        (Traced) The constraint thresholds. Derivatives will be calculated w.r.t. this quantity.
    metric_name : tuple, optional, default=('f_B', 'f_K')
        (Static) The names of the functions to diagnose the coil configurations with. Will be differentiated w.r.t. other input quantities.
    convex : bool, optional, default=False
        (Static) Whether to assume the problem is convex. When ``True``, QUADCOIL will apply some limited simplifications.
    solver_options : dict, optional, default=None
        (Traced) Augmented-Lagrangian and inner-solver options. Merged with
        ``SOLVER_OPTIONS_DEFAULT``; unspecified keys take their defaults.
        Recognised keys and defaults:
        - ``'c_init'`` (``1.``) — initial penalty :math:`c` factor.
        - ``'c_growth_rate'`` (``2.``) — multiplicative growth of :math:`c` each outer step.
        - ``'xstop_outer'`` (``1e-6``) — outer-loop ``x`` convergence rate tolerance.
        - ``'ctol_outer'`` (``1e-6``) — outer-loop constraint-violation tolerance.
        - ``'atol_inner'`` (``1e-6``) — absolute gradient tolerance for inner L-BFGS solves.
        - ``'rtol_inner'`` (``1e-6``) — relative gradient tolerance for inner L-BFGS solves.
        - ``'atol_inner_last'`` (``1e-10``) — absolute gradient tolerance for the final inner solve.
        - ``'rtol_inner_last'`` (``1e-10``) — relative gradient tolerance for the final inner solve.
        - ``'svtol'`` (``1e-6``) — singular-value cut-off for pre-conditioning.
        - ``'maxiter_tot'`` (``10000``) — maximum outer-loop iterations.
        - ``'maxiter_inner'`` (``1000``) — maximum inner L-BFGS iterations per outer step.
    lbfgs_memory : int, optional, default=10
        (Static) L-BFGS history length for the inner solver.
    implicit_linear_solver : lineax.AbstractLinearSolver, optional, default=lineax.AutoLinearSolver(well_posed=True)
        (Static) The lineax linear solver choice for implicit differentiation.
    value_only : bool, optional, default=False
        (Static) When ``True``, skip gradient calculations.
    verbose : int, optional, default=False
        (Static) Print general info when ``verbose==1``. 
        Print inside the outer iteration loop, too, when ``verbose==2``.
    '''
    # ----- Solver options unpacking -----
    if solver_options is None:
        solver_options = SOLVER_OPTIONS_DEFAULT_DICT[solver]

    # ----- Default parameters -----
    # ess_alpha = jnp.abs(ess_alpha)
    (
        plasma_quadpoints_phi, plasma_quadpoints_theta,
        winding_quadpoints_phi, winding_quadpoints_theta,
        quadpoints_phi, quadpoints_theta,
    ) = _resolve_quadpoints(
        nfp=nfp,
        Bnormal_plasma=Bnormal_plasma,
        plasma_quadpoints_phi=plasma_quadpoints_phi,
        plasma_quadpoints_theta=plasma_quadpoints_theta,
        winding_quadpoints_phi=winding_quadpoints_phi,
        winding_quadpoints_theta=winding_quadpoints_theta,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        plasma_coil_distance=plasma_coil_distance,
        winding_dofs=winding_dofs,
    )
    if isinstance(metric_name, str):
        metric_name = (metric_name,)
    if implicit_linear_solver is None:
        # The bool conversion here is to make sure that stellsym is bool instead 
        # of numpy bools. This is because desc.Equilibrium.sym is a numpy.bool
        # and can cause issues with lineax.
        implicit_linear_solver = lx.AutoLinearSolver(well_posed=False) # bool(stellsym))
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
            'Smoothing mode for non-smooth functions: {smoothing}\n'\
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
            'Solver type: {solver}\n'\
            'Solver options:\n'\
            '{solver_options}\n',
            smoothing=smoothing,
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
            solver=solver,
            solver_options=solver_options,
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
        surface_cls = SURFACE_TYPE_MAP[surface_type]
        plasma_surface = surface_cls(
            nfp=nfp, stellsym=plasma_stellsym, 
            mpol=plasma_mpol, ntor=plasma_ntor, 
            quadpoints_phi=plasma_quadpoints_phi, 
            quadpoints_theta=plasma_quadpoints_theta,
            dofs=y_dict['plasma_dofs']
        )
        # winding surface is provided. 
        # Its dofs will be among x.
        if plasma_coil_distance is None:
            winding_surface = surface_cls(
                nfp=nfp, stellsym=winding_stellsym, 
                mpol=winding_mpol, ntor=winding_ntor, 
                quadpoints_phi=winding_quadpoints_phi, 
                quadpoints_theta=winding_quadpoints_theta,
                dofs=y_dict['winding_dofs']
            )
        # winding surface is not provided. 
        # Its dofs will not be among x.
        # gen_offset_dofs is only implemented for SurfaceRZFourierJAX;
        # other surface types will raise NotImplementedError here.
        else:
            winding_dofs_temp = plasma_surface.gen_offset_dofs(
                d_expand=y_dict['plasma_coil_distance'],
                mpol=winding_mpol,
                ntor=winding_ntor,
                smoothing=offset_smoothing,
            )
            winding_surface = surface_cls(
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
            stellsym=stellsym,
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
            smoothing=smoothing,
            smoothing_params=smoothing_params,
        )
        g_cons_list, h_cons_list, aux_dofs_cons = _parse_constraints(
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
            constraint_value=constraint_value_temp,
            smoothing=smoothing,
            smoothing_params=smoothing_params,
        )
        # Merging constraints and aux dofs from different sources
        g_list = g_obj_list + g_cons_list
        h_list = h_obj_list + h_cons_list
        aux_dofs_init = aux_dofs_obj | aux_dofs_cons

        f_obj_x = lambda x, qp_temp=qp_temp, f_obj=f_obj: f_obj(qp_temp, x)
        g_ineq_x = lambda x, qp_temp=qp_temp, g_list=g_list: merge_callables(
            g_list, 
            merge_constraints=merge_constraints,
            smoothing=smoothing,
            smoothing_params=smoothing_params,
        )(qp_temp, x)
        # Merging equality constraints is not supported yet    
        h_eq_x = lambda x, qp_temp=qp_temp, h_list=h_list: merge_callables(
            h_list
        )(qp_temp, x)
        if merge_constraints:
            n_g = 1
        else:
            n_g = len(g_list)
        # Merging equality constraints is not supported yet    
        n_h = len(h_list)
        return f_obj_x, g_ineq_x, h_eq_x, n_g, n_h, aux_dofs_init

    # ----- Creating Initializing phi -----
    # Defining a shared problem parameter object
    qp = y_to_qp(y_dict_current)
    # f, g, h are Callable(qp, {'phi':, ..., })
    # i.e., they accepts unscaled input
    f_obj, g_ineq, h_eq, n_g, n_h, aux_dofs_init = f_g_ineq_h_eq_from_y(y_dict_current)
    constrained = not ((n_g == 0) and (n_h == 0))

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
        Bnormal_estimate = jnp.average(jnp.abs(Bnormal(qp, dofs_dict_init))) # Unit: T
        if plasma_coil_distance is not None:
            phi_unit = Bnormal_estimate * 1e7 * jnp.abs(plasma_coil_distance)
        else:
            # The minor radius can be estimated from the 
            # n=0, m=1 rc mode of the surface.
            plasma_minor = plasma_dofs[plasma_ntor*2 + 1]
            winding_minor = winding_dofs[winding_ntor*2 + 1]
            phi_unit = Bnormal_estimate * 1e7 * jnp.abs(plasma_minor - winding_minor)
    # # ESS scaling (See Chris Jang's paper at https://arxiv.org/pdf/2509.16320) 
    # ess_factor = jnp.exp(
    #     -ess_alpha 
    #     * jnp.linalg.norm(jnp.array(qp.make_mn()), ord=1, axis=0)
    # )
    # ess_factor = ess_factor / jnp.average(ess_factor)
    # phi_unit = phi_unit * ess_factor
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
    if not constrained:
        if solver == 'auglag-lbfgs':
            solve_results = solve_unconstrained_auglag_lbfgs(
                init_params=x_flat_init,
                fun=f_scaled,
                convex=convex,
                solver_options={
                    'maxiter': solver_options['maxiter_tot'],
                    'atol': solver_options['atol_inner_last'],
                    'rtol': solver_options['rtol_inner_last'],
                },
                verbose=verbose,
                lbfgs_memory=lbfgs_memory,
            )
        elif solver == 'ipm':
            solve_results = solve_unconstrained_ipm(
                init_params=x_flat_init,
                fun=f_scaled,
                convex=convex,
                solver_options=solver_options,
                verbose=verbose,
            )
        elif solver == 'slsqp':
            solve_results = solve_unconstrained_slsqp(
                init_params=x_flat_init,
                fun=f_scaled,
                convex=convex,
                solver_options=solver_options,
                verbose=verbose,
                lbfgs_memory=lbfgs_memory,
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")
        x_flat_opt = solve_results['fin_x']
        dofs_opt = unravel_unscale_x(x_flat_opt)
        if verbose>0:       
            debug.print(
                '----- Solver status summary -----\n'\
                'Final value of objective f: {f}\n'\
                'Total iteration number: {niter}\n'\
                'Solve results:\n'\
                '{solve_results}',
                f=solve_results['fin_f'],
                niter=solve_results['niter'],
                solve_results=solve_results
            )
    else:
        if solver == 'auglag-lbfgs':
            solve_results = solve_constrained_auglag_lbfgs(
                x_init=x_flat_init,
                f_obj=f_scaled,
                h_eq=h_scaled,
                g_ineq=g_scaled,
                solver_options={**solver_options, 'lam_init': lam_init, 'mu_init': mu_init},
                verbose=verbose,
                lbfgs_memory=lbfgs_memory,
            )
        elif solver == 'ipm':
            solve_results = solve_constrained_ipm(
                x_init=x_flat_init,
                f_obj=f_scaled,
                h_eq=h_scaled,
                g_ineq=g_scaled,
                convex=convex,
                solver_options=solver_options,
                verbose=verbose,
            )
        elif solver == 'slsqp':
            solve_results = solve_constrained_slsqp(
                x_init=x_flat_init,
                f_obj=f_scaled,
                h_eq=h_scaled,
                g_ineq=g_scaled,
                convex=convex,
                solver_options=solver_options,
                verbose=verbose,
                lbfgs_memory=lbfgs_memory,
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")
        # The optimum, unit-less.
        x_flat_opt = solve_results['fin_x']
        dofs_opt = unravel_unscale_x(x_flat_opt)
        if verbose>0:       
            debug.print(
                '----- Solver status summary -----\n'\
                'Final value of objective f: {f}\n'\
                'Total iteration number: {niter}\n'\
                'Solve results:\n'\
                '{solve_results}',
                f=solve_results['fin_f'],
                niter=solve_results['niter'],
                solve_results=solve_results
            )
    # ----- Calculating metrics and gradients
    if value_only: 
        out_dict = {}
        for metric_name_i in metric_name:
            if metric_name_i == 'f_obj':
                metric_result_i = f_obj(qp_temp=qp, x=dofs_opt)
            else:
                metric_result_i = get_quantity(metric_name_i)(qp, dofs_opt)
                out_dict[metric_name_i] = {
                    'value': metric_result_i
                }
            if verbose>0:
                debug.print('Metric evaluated. {x} = {y}', x=metric_name_i, y=metric_result_i)
        return out_dict, qp, dofs_opt, solve_results
    # flatten the y dictionary. This will simplify the code structure a bit
    y_flat, unravel_y = flatten_util.ravel_pytree(y_dict_current)

    # ----- Stationarity conditions -----
    if solver == 'auglag-lbfgs':
        stationarity_data = stationarity_auglag_lbfgs(
            constrained=constrained,
            convex=convex,
            solve_results=solve_results,
            y_flat=y_flat,
            f_g_ineq_h_eq_from_y=f_g_ineq_h_eq_from_y,
            unravel_y=unravel_y,
            unravel_unscale_x=unravel_unscale_x,
            solver_options=solver_options,
            verbose=verbose,
        )
    elif solver == 'ipm':
        stationarity_data = stationarity_ipm(
            constrained=constrained,
            convex=convex,
            solve_results=solve_results,
            y_flat=y_flat,
            f_g_ineq_h_eq_from_y=f_g_ineq_h_eq_from_y,
            unravel_y=unravel_y,
            unravel_unscale_x=unravel_unscale_x,
            solver_options=solver_options,
            verbose=verbose,
        )
    elif solver == 'slsqp':
        stationarity_data = stationarity_slsqp(
            constrained=constrained,
            convex=convex,
            solve_results=solve_results,
            y_flat=y_flat,
            f_g_ineq_h_eq_from_y=f_g_ineq_h_eq_from_y,
            unravel_y=unravel_y,
            unravel_unscale_x=unravel_unscale_x,
            solver_options=solver_options,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")

    out_dict = {}

    for metric_name_i in metric_name:
        if metric_name_i == 'f_obj':
            f_metric = lambda xp, y: f_obj(
                qp_temp=y_to_qp(unravel_y(y)),
                x=unravel_unscale_x(xp)
            )
        else:
            f_metric = lambda xp, y: get_quantity(metric_name_i)(
                y_to_qp(unravel_y(y)),
                unravel_unscale_x(xp)
            )
        if solver == 'auglag-lbfgs':
            metric_result_i, dfdy_arr, debug_info_i = adjoint_auglag_lbfgs(
                f_metric=f_metric,
                stationarity_data=stationarity_data,
                y_flat=y_flat,
                implicit_linear_solver=implicit_linear_solver,
                verbose=verbose,
            )
        elif solver == 'ipm':
            metric_result_i, dfdy_arr, debug_info_i = adjoint_ipm(
                f_metric=f_metric,
                stationarity_data=stationarity_data,
                y_flat=y_flat,
                implicit_linear_solver=implicit_linear_solver,
                verbose=verbose,
            )
        elif solver == 'slsqp':
            metric_result_i, dfdy_arr, debug_info_i = adjoint_slsqp(
                f_metric=f_metric,
                stationarity_data=stationarity_data,
                y_flat=y_flat,
                implicit_linear_solver=implicit_linear_solver,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")
        dfdy_dict = {f"df_d{key}": value for key, value in unravel_y(dfdy_arr).items()}
        if verbose > 0:
            grad_avgs = {}
            for k in dfdy_dict:
                item_k = jnp.atleast_1d(dfdy_dict[k])
                grad_avgs[k] = (jnp.min(item_k), jnp.max(item_k))
            debug.print(
                '* Metric evaluated.\n'
                '    {x} = {y}\n'
                '    Gradient min, max: {g}',
                x=metric_name_i,
                y=metric_result_i,
                g=grad_avgs,
            )
        out_dict[metric_name_i] = {
            'value': metric_result_i,
            'grad': dfdy_dict,
        }
        if verbose > 0:
            out_dict[metric_name_i].update(debug_info_i)
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
        if len(objective_name) != len(objective_weight) or len(objective_name) != len(objective_unit):
            raise ValueError('objective_name, objective_weight, and objective_unit must have the same len')
    else:
        objective_weight = 1.
    if not isinstance(constraint_name, tuple):
        raise TypeError('constraint_name must be a tuple. It is:', type(constraint_name))
    if not isinstance(constraint_type, tuple):
        raise TypeError('constraint_type must be a tuple. It is:', type(constraint_type))
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