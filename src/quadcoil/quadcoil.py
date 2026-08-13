from quadcoil import (
    merge_callables, get_quantity,
    SurfaceRZFourierJAX, SurfaceXYZTensorFourierJAX, SurfaceXYZFourierJAX,
    QuadcoilParams, 
    is_ndarray, tree_len,
    qp_nescoil,
)

from quadcoil.solver import (
    solve_constrained_auglag_lbfgs, 
    solve_constrained_ipm,
    solve_constrained_slsqp,
    solve_unconstrained_auglag_lbfgs,
    solve_unconstrained_ipm,
    solve_unconstrained_slsqp,
    recover_multipliers,
    stationarity_kkt,
    adjoint_kkt,
)

from quadcoil.wrapper import _parse_objectives, _parse_constraints, _resolve_quadpoints
from functools import partial
from quadcoil.quantity import Bnormal, K_cyl
from jax import jacfwd, jacrev, jit, block_until_ready, debug, flatten_util, eval_shape, grad, vmap, tree_util
from jax import config as config_jax
import jax.numpy as jnp
import lineax as lx
import warnings
config_jax.update('jax_enable_x64', True)


tol_default = 1e-6
tol_default_last = 1e-10

surface_type_MAP = {
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
    },
    'ipm': {
        'tol_kkt': 1e-6,
        'tau': 0.995,
        'delta_init': 1e-6,
        'delta_min': 1e-10,
        'delta_max': 1e-2,
    },
    'slsqp': {
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
    'phi_init_with_nescoil',
    # - Plasma options
    'plasma_mpol',
    'plasma_ntor',
    'plasma_stellsym',
    # - WS options
    'surface_type',
    'winding_stellsym',
    'winding_mpol',
    'winding_ntor',
    'winding_phi_interp',
    'winding_theta_interp',
    'winding_theta_rule_subsample',
    'winding_surface_mode',
    'winding_theta_mode',
    # - Objectives
    'objective_name',
    # - Constraints 
    'constraint_name',
    'constraint_type',
    # - Metrics
    'metric_name',
    # - Preconditioning options
    'precond',
    'precond_dims',
    # - Constraint handling and adjoint
    'value_only',
    'convex',
    'merge_constraints',
    # - Solver options
    'solver',
    'lbfgs_memory',
    'maxiter',
    'maxiter_inner',
    # - Other options
    'verbose',
    # Smoothing parameters:
    'smoothing',
]
@partial(jit, static_argnames=QUADCOIL_STATIC_ARGNAMES)
def _quadcoil_pure(
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
    # Whether to initalize with nescoil sln. Will override phi_init and phi_unit.
    phi_init_with_nescoil=True, 
    # Current potential's normalization constant. 
    # By default will be generated from net total current.
    phi_unit=None, # Documented
    
    # - Plasma parameters
    plasma_stellsym=True, # Documented
    plasma_quadpoints_phi=None, # Documented
    plasma_quadpoints_theta=None, # Documented
    Bnormal_plasma=None, # Documented
    surface_type:str='SurfaceRZFourier', # Documented

    # - Winding parameters (offset)
    plasma_coil_distance:float=None, # Documented
    winding_phi_interp:int=2,
    winding_theta_interp:int=2,
    winding_theta_rule_subsample:int=2,
    winding_lam_tikhonov=1e-5,
    winding_surface_mode='self-intersection',
    winding_theta_mode='arclen',

    # - Winding parameters (known surface)
    winding_dofs=None, # Documented
    winding_stellsym=True,

    # - Winding parameters (shared)
    winding_mpol:int=6, # Documented
    winding_ntor:int=5, # Documented
    winding_quadpoints_phi=None, # Documented
    winding_quadpoints_theta=None, # Documented
    
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

    # - Preconditioning options
    precond='svd', # Supported options are 'ess', 'svd', 'svd_K' and None
    precond_dims=None,
    precond_options={
        'svd_safe_thres': 0.,
        'ess_alpha': 1.,
        'ess_p': 2.,
    },
    
    # - Constraint handling and adjoint
    value_only=False,
    smoothing='approx',
    smoothing_params={'lse_epsilon': 1e-3},
    convex:bool=False,

    # - Solver options
    verbose:int=0,
    merge_constraints:bool=False,
    solver:str='auglag-lbfgs', # 'auglag-lbfgs',
    solver_options=None,
    lbfgs_memory:int=10, # applicable for 'slsqp' only
    maxiter:int=None,
    maxiter_inner:int=None, # applicable for 'auglag-lbfgs' only
):
    r'''The jitted part of quadcoil().
    '''
    # ----- Solver options unpacking -----
    if solver_options is None:
        solver_options = SOLVER_OPTIONS_DEFAULT_DICT[solver]

    # ----- maxiter defaults per solver -----
    if maxiter is None:
        if solver == 'auglag-lbfgs':
            maxiter = 10000
        elif solver == 'ipm':
            maxiter = 100
        else:  # slsqp
            maxiter = 500
    if maxiter_inner is None:
        maxiter_inner = 500

    # ----- Warning for constraint treatment -----
    if smoothing == 'slack' and solver != 'auglag-lbfgs':
        warnings.warn(
            'We only recommend using smoothing=\'slack\' '
            'with solver=\'auglag-lbfgs\'. The current value of '
            'solver is ' + solver
        )

    # ----- Default parameters -----
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
    if metric_name is None:
        metric_name = ()
    elif isinstance(metric_name, str):
        metric_name = (metric_name,)
    else:
        metric_name = tuple(metric_name)
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
        surface_cls = surface_type_MAP[surface_type]
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
            winding_surface = plasma_surface.gen_winding_surface(
                d_expand=y_dict['plasma_coil_distance'],
                mpol=winding_mpol,
                ntor=winding_ntor,
                phi_interp=winding_phi_interp,
                theta_interp=winding_theta_interp,
                theta_rule_subsample=winding_theta_rule_subsample,
                quadpoints_phi=winding_quadpoints_phi,
                quadpoints_theta=winding_quadpoints_theta,
                lam_tikhonov=winding_lam_tikhonov,
                winding_surface_mode=winding_surface_mode,
                theta_mode=winding_theta_mode,
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

    if phi_init_with_nescoil:
        if phi_init or phi_unit is not None:
            warnings.warn(
                'phi_init_with_nescoil is True, but '
                'phi_init or phi_unit are not None, and '
                'they will be replaced with the NESCOIL values.'
            )
        phi_init = qp_nescoil(qp)
        phi_unit = jnp.max(jnp.abs(phi_init))
        dofs_init = {'phi': phi_init}
    else:
        if phi_init is None:
            phi_init = jnp.zeros(qp.ndofs)
        dofs_init = {'phi': phi_init}
        # ----- Calculating the unit of phi -----
        # phi need to be normalized to ~1 for the optimizer to behave well.
        # by default we do this using the initial value of Bnormal
        if phi_unit is None:
            # Scaling current potential dofs to ~1
            # By default, we use the Bnormal value when 
            # phi=0 to generate this scaling factor.
            Bnormal_estimate = jnp.average(jnp.abs(Bnormal(qp, dofs_init))) # Unit: T
            if plasma_coil_distance is not None:
                phi_unit = Bnormal_estimate * 1e7 * jnp.abs(plasma_coil_distance)
            else:
                # The minor radius can be estimated from the 
                # n=0, m=1 rc mode of the surface.
                plasma_minor = plasma_dofs[plasma_ntor*2 + 1]
                winding_minor = winding_dofs[winding_ntor*2 + 1]
                phi_unit = Bnormal_estimate * 1e7 * jnp.abs(plasma_minor - winding_minor)
                    


            
    # ----- Preconditioning -----
    
    # ----- Creating scaled, flattened dof, 'x_flat_init' -----
    # The actual, unit-free, variable used for initialization,
    # and by the optimizer. The dof that the optimizer operates on is a
    # flattened version of this dictionary.
    if precond is None:
        _precond_phi = lambda phi: phi / phi_unit
        _recover_phi = lambda x: x * phi_unit
        
    elif precond in ('svd', 'svd_K'):
        # Preparing quantities for SVD-based pre-conditioning
        # This is the maximum difference in order-of-magnitude 
        # across all singular values
        svd_safe_thres = precond_options['svd_safe_thres']
        # Materializing the Bnormal matrix using lineax, since 
        # unlike REGCOIL, our Bnormal is a direct implementation of 
        # the Biot-Savart law in QUADCOIL.
        if precond == 'svd':
            precond_f = lambda x, qp=qp: Bnormal(qp, {'phi': x}) - Bnormal(qp, {'phi': jnp.zeros_like(phi_init)})
        else:
            precond_f = lambda x, qp=qp: K_cyl(qp, {'phi': x}) - K_cyl(qp, {'phi': jnp.zeros_like(phi_init)})
        precond_op = lx.FunctionLinearOperator(
            fn=precond_f, 
            input_structure=phi_init
        ).as_matrix()
        # Performing SVD
        precond_U, precond_s, precond_Vh = jnp.linalg.svd(precond_op, full_matrices=False)
        # In case there are very small singular values, use 
        # svd_safe_thres as a thresold to prevent divide 
        # by zero.
        svd_scale = jnp.where(
            precond_s<jnp.max(precond_s)*svd_safe_thres, 
            jnp.max(precond_s)*svd_safe_thres, 
            precond_s
        )
        # Default: no SVD truncation
        if precond_dims is None:
            precond_dims = qp.ndofs
            
        def _precond_phi(phi):
            phi_projected = precond_Vh @ phi
            x_padded = phi_projected * svd_scale
            return x_padded[:precond_dims]
        
        def _recover_phi(x):
            x_padded = jnp.concatenate([x, jnp.zeros(precond_Vh.shape[0] - len(x))])
            x_scaled = x_padded / svd_scale
            phi_recovered = precond_Vh.T @ x_scaled
            return phi_recovered
            
    elif precond == 'ess':
        # Preparing quantities for 
        ess_alpha = jnp.abs(precond_options['ess_alpha'])
        # p for Lp norm has to be > 1
        ess_p = jnp.where(precond_options['ess_p'] < 1, 1, precond_options['ess_p'])
        # Calculating the ESS factor
        ess_mn = jnp.array(qp.make_mn()) 
        ess_Lp_norm = jnp.sum(jnp.abs(ess_mn) ** ess_p, axis=0) ** (1/ess_p)
        ess_factor = jnp.exp(ess_alpha * ess_Lp_norm)
        
        def _precond_phi(phi):
            return phi / phi_unit * ess_factor
        
        def _recover_phi(x):
            return x / ess_factor * phi_unit
            
    else:
        raise ValueError(f"Unknown preconditioner: {precond}")

    # Converting the preconditioned dofs into a flattened array
    # by first replacing the phi element with a 'phi_precond' 
    # element, and then unraveling.
    x_dict = {
       'phi_precond': _precond_phi(phi_init),
       # And auxiliary vars. Because we have already implemented 
       # scaling for them in _add_quantity instances, we do not 
       # need to precondition them here.
    }
    # Calculating the structure of auxiliary dofs from the problem setup (qp).
    # The current dictionary's items are either None (scalar), tuple (known shape), or 
    # Callable(QuadcoilParams) (shapes that depend on problem setup)
    for key in aux_dofs_init.keys():
        if callable(aux_dofs_init[key]): 
            # Callable(qp: QuadcoilParams, dofs: dict, f_unit: float)
            x_dict[key] = aux_dofs_init[key](qp, dofs_init)
        else:
            try:
                x_dict[key] = jnp.array(aux_dofs_init[key])
            except:
                raise TypeError(
                    f'The auxiliary variable {key} is not a callable, '\
                    'and cannot be converted to an array. Its value is: '\
                    f'{str(aux_dofs_init[key])}. This is dur to improper '\
                    'implementation of the physical quantity. Please contact the developers.')
                
    # We now flatten x into a jax array so that we can pass it into a solver.
    x_flat_init, unravel_x = flatten_util.ravel_pytree(x_dict)
    

    def flat_x_to_dofs(x, unravel_x=unravel_x, phi_unit=phi_unit):
        d = unravel_x(x)
        # Replace scaled phi with regular phi
        # after unraveling for passing into 
        # f_obj, g_ineq and h_eq.
        dofs_temp = {
            k: v for k, v in {**d, "phi": _recover_phi(d["phi_precond"])}.items() if k != "phi_precond"
        }
        return(dofs_temp)

    # Versions of f, g, h that takes in the flat, preconditioned x array.
    f_precond = lambda x_precond, f_obj=f_obj: f_obj(flat_x_to_dofs(x_precond))
    g_precond = lambda x_precond, g_ineq=g_ineq: g_ineq(flat_x_to_dofs(x_precond))
    h_precond = lambda x_precond, h_eq=h_eq: h_eq(flat_x_to_dofs(x_precond))
    
    mu_init = jnp.zeros(eval_shape(g_precond, x_flat_init).shape)
    lam_init = jnp.zeros(eval_shape(h_precond, x_flat_init).shape)
    
    # ----- Summarizing initialization -----
    # This block prints out a summary on the auxiliary vars and 
    # phi degrees of freedom.
    if verbose>0:
        ny = tree_len(y_dict_current)
        ndofs_tot = len(x_flat_init) # This counts the aux vars too
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
                fun=f_precond,
                convex=convex,
                maxiter=maxiter,
                solver_options={
                    'atol': solver_options.get('atol_inner_last', tol_default_last),
                    'rtol': solver_options.get('rtol_inner_last', tol_default_last),
                },
                verbose=verbose,
                lbfgs_memory=lbfgs_memory,
            )
        elif solver == 'ipm':
            solve_results = solve_unconstrained_ipm(
                init_params=x_flat_init,
                fun=f_precond,
                convex=convex,
                maxiter=maxiter,
                solver_options=solver_options,
                verbose=verbose,
            )
        elif solver == 'slsqp':
            solve_results = solve_unconstrained_slsqp(
                init_params=x_flat_init,
                fun=f_precond,
                convex=convex,
                maxiter=maxiter,
                solver_options=solver_options,
                verbose=verbose,
                lbfgs_memory=lbfgs_memory,
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")
        x_flat_opt = solve_results['fin_x']
        dofs_opt = flat_x_to_dofs(x_flat_opt)
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
                f_obj=f_precond,
                h_eq=h_precond,
                g_ineq=g_precond,
                maxiter=maxiter,
                maxiter_inner=maxiter_inner,
                solver_options={**solver_options, 'lam_init': lam_init, 'mu_init': mu_init},
                verbose=verbose,
                lbfgs_memory=lbfgs_memory,
            )
        elif solver == 'ipm':
            solve_results = solve_constrained_ipm(
                x_init=x_flat_init,
                f_obj=f_precond,
                h_eq=h_precond,
                g_ineq=g_precond,
                convex=convex,
                maxiter=maxiter,
                solver_options=solver_options,
                verbose=verbose,
            )
        elif solver == 'slsqp':
            solve_results = solve_constrained_slsqp(
                x_init=x_flat_init,
                f_obj=f_precond,
                h_eq=h_precond,
                g_ineq=g_precond,
                convex=convex,
                maxiter=maxiter,
                solver_options=solver_options,
                verbose=verbose,
                lbfgs_memory=lbfgs_memory,
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")
        # The optimum, unit-less.
        x_flat_opt = solve_results['fin_x']
        dofs_opt = flat_x_to_dofs(x_flat_opt)
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
            
    # ----- Calculating metrics and gradients -----
    # If nescoil initial conditions are used, output it
    # solution too.
    if phi_init_with_nescoil:
        solve_results['dofs_nescoil'] = {'phi': phi_init}

    # value_only or empty metric_name: skip KKT / adjoint work.
    if value_only or len(metric_name) == 0:
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
    
    # Each solver has slightly different stationarity condition.
    x_opt = solve_results['fin_x']
    if solver == 'auglag-lbfgs':
        z_ineq, z_eq = recover_multipliers(
            x_opt, y_flat, f_g_ineq_h_eq_from_y, unravel_y, flat_x_to_dofs,
        )
        z_opt = jnp.concatenate([z_ineq, z_eq])

        def f_g_combined_from_y(y_dict):
            f_obj_i, g_ineq_i, h_eq_i, n_g_i, n_h_i, aux = f_g_ineq_h_eq_from_y(y_dict)
            g_combined = lambda dofs: jnp.concatenate([g_ineq_i(dofs), h_eq_i(dofs)])
            h_empty = lambda dofs: jnp.zeros(0)
            return f_obj_i, g_combined, h_empty, n_g_i + n_h_i, 0, aux

        f_g_for_kkt = f_g_combined_from_y
    elif solver in ('ipm', 'slsqp'):
        z_opt = solve_results.get('fin_z', jnp.zeros(0, dtype=x_opt.dtype))
        f_g_for_kkt = f_g_ineq_h_eq_from_y
    else:
        raise ValueError(f"Unknown solver: {solver}")

    if not constrained:
        z_opt = jnp.zeros(0, dtype=x_opt.dtype)

    stationarity_data = stationarity_kkt(
        constrained=constrained,
        x_opt=x_opt,
        z_opt=z_opt,
        y_flat=y_flat,
        f_g_ineq_h_eq_from_y=f_g_for_kkt,
        unravel_y=unravel_y,
        flat_x_to_dofs=flat_x_to_dofs,
        verbose=verbose,
    )

    # Evaluate metrics once for shapes / sizes, then batch into one adjoint.
    metric_shapes = []
    metric_K_list = []
    for metric_name_i in metric_name:
        if metric_name_i == 'f_obj':
            v = f_obj(qp_temp=qp, x=dofs_opt)
        else:
            v = get_quantity(metric_name_i)(qp, dofs_opt)
        metric_shapes.append(jnp.shape(v))
        metric_K_list.append(int(jnp.size(v)))
    K_tot = sum(metric_K_list)

    def f_metrics_flat(xp, y):
        qp_temp = y_to_qp(unravel_y(y))
        dofs_temp = flat_x_to_dofs(xp)
        parts = []
        for n in metric_name:
            if n == 'f_obj':
                v = f_obj(qp_temp=qp_temp, x=dofs_temp)
            else:
                v = get_quantity(n)(qp_temp, dofs_temp)
            parts.append(jnp.atleast_1d(v).ravel())
        return jnp.concatenate(parts)

    all_values, dfdy, debug_info = adjoint_kkt(
        f_metrics_flat, K_tot, stationarity_data, y_flat, verbose,
    )

    out_dict = {}
    offset = 0
    for i, metric_name_i in enumerate(metric_name):
        k = metric_K_list[i]
        shape_i = metric_shapes[i]
        val_flat = all_values[offset:offset + k]
        dfdy_rows = dfdy[offset:offset + k]  # (k, n_y)
        if k == 1 and shape_i == ():
            metric_result_i = val_flat[0]
            dfdy_pytree = unravel_y(dfdy_rows[0])
        else:
            metric_result_i = val_flat.reshape(shape_i)
            dfdy_pytree = vmap(unravel_y)(dfdy_rows)
            # Leading axis k -> original metric shape.
            dfdy_pytree = tree_util.tree_map(
                lambda g, s=shape_i: g.reshape(s + g.shape[1:]),
                dfdy_pytree,
            )
        dfdy_dict = {f"df_d{key}": value for key, value in dfdy_pytree.items()}
        if verbose > 0:
            grad_avgs = {}
            for key_g in dfdy_dict:
                item_k = jnp.atleast_1d(dfdy_dict[key_g])
                grad_avgs[key_g] = (jnp.min(item_k), jnp.max(item_k))
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
            out_dict[metric_name_i].update(debug_info)
        offset += k
    return out_dict, qp, dofs_opt, solve_results


def quadcoil(**kwargs):
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
        (Traced) The plasma surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` convention.
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
        (Traced) The initial guess. All zeros by default (unless ``phi_init_with_nescoil`` is ``True``).
    phi_init_with_nescoil : bool, optional, default=True
        (Static) When ``True``, initialize :math:`\Phi_{sv}` from a NESCOIL solve
        (overrides ``phi_init`` / ``phi_unit``).
    phi_unit : float, optional, default=None
        (Traced) Current potential's normalization constant. Only applies when ``precond!='svd'``.
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
    winding_surface_mode : str, optional, default='self-intersection'
        (Static) Winding-surface generation mode when auto-generating from
        ``plasma_coil_distance``. One of ``'self-intersection'``, ``'hull'``,
        or ``'uniform'``.
    winding_theta_mode : str, optional, default='arclen'
        (Static) Poloidal reparameterization for the offset-surface fit.
        One of ``'arclen'`` or ``'arctan'``.
    winding_phi_interp : int, optional, default=2
        (Static) Toroidal oversampling factor during the Fourier fit.
    winding_theta_interp : int, optional, default=2
        (Static) Poloidal oversampling factor during the Fourier fit.
    winding_theta_rule_subsample : int, optional, default=2
        (Static) Poloidal subsampling stride for the self-intersection check.
    winding_lam_tikhonov : float, optional, default=1e-5
        (Traced) Tikhonov regularization weight for the least-squares surface fit.
    winding_dofs : ndarray, shape (ndof_winding,), optional, default=None
        (Traced) The winding surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` convention.
        Will be generated using ``winding_surface_mode`` if ``plasma_coil_distance`` is provided. Must be provided otherwise.
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
    objective_name : str or tuple, optional, default='f_B'
        (Static) The names of the objective functions. Must be a member of ``quadcoil.quantity`` that outputs a scalar.
    objective_weight : ndarray, optional, default=1.
        (Traced) The weights of the objective functions. Derivatives will be calculated w.r.t. this quantity.
    objective_unit : ndarray, optional, default=None
        (Traced) The normalization constants of the objective terms, so that ``f/objective_unit`` is :math:`O(1)`. May contain ``None``
    constraint_name : tuple, optional, default=()
        (Static) The names of the constraint functions. Must be a member of ``quadcoil.quantity`` that outputs a scalar.
    constraint_type : tuple, optional, default=()
        (Static) The types of the constraints. Must consist of ``'>='``, ``'<='``, ``'=='`` only.
    constraint_unit : ndarray, optional, default=()
        (Traced) The normalization constants of the constraints, so that ``f/constraint_unit`` is :math:`O(1)` May contain ``None``.
    constraint_value : ndarray, optional, default=()
        (Traced) The constraint thresholds. Derivatives will be calculated w.r.t. this quantity.
    metric_name : None, str, list, or tuple, optional, default=('f_B', 'f_K')
        (Static) The names of the functions to diagnose the coil configurations
        with. Will be differentiated w.r.t. other input quantities.
        ``None`` or ``()`` skips metric evaluation (and KKT adjoint work).
        A single ``str`` is treated as a one-element tuple. Lists are converted
        to tuples before JIT (static args must be hashable).
        Include ``'phi_dofs'`` to obtain the solution DOFs and their Jacobians.
    precond : str or None, optional, default='svd'
        (Static) Current-potential preconditioner. One of ``'svd'``, ``'ess'``,
        ``'svd_K'``, or ``None``.
    precond_dims : tuple or None, optional, default=None
        (Static) Optional dimensions used by some preconditioners.
    precond_options : dict, optional
        (Traced) Preconditioner hyperparameters. Defaults to
        ``{'svd_safe_thres': 0., 'ess_alpha': 1., 'ess_p': 2.}``.
    convex : bool, optional, default=False
        (Static) Whether to assume the problem is convex for supported solvers
        (``'ipm'``, ``'slsqp'``). The KKT adjoint path does not use this flag.
    solver : str, optional, default='auglag-lbfgs'
        (Static) Optimizer backend. One of ``'auglag-lbfgs'``, ``'ipm'``, ``'slsqp'``.
    solver_options : dict, optional, default=None
        (Traced) Solver-specific options. Merged with
        ``SOLVER_OPTIONS_DEFAULT_DICT[solver]``; unspecified keys take their defaults.

        For ``'auglag-lbfgs'``:
        - ``'c_init'`` (``1.``) — initial penalty :math:`c` factor.
        - ``'c_growth_rate'`` (``2.``) — multiplicative growth of :math:`c` each outer step.
        - ``'xstop_outer'`` (``1e-6``) — outer-loop ``x`` convergence rate tolerance.
        - ``'ctol_outer'`` (``1e-6``) — outer-loop constraint-violation tolerance.
        - ``'atol_inner'`` (``1e-6``) — absolute gradient tolerance for inner L-BFGS solves.
        - ``'rtol_inner'`` (``1e-6``) — relative gradient tolerance for inner L-BFGS solves.
        - ``'atol_inner_last'`` (``1e-10``) — absolute gradient tolerance for the final inner solve.
        - ``'rtol_inner_last'`` (``1e-10``) — relative gradient tolerance for the final inner solve.
        - ``'svtol'`` (``1e-6``) — singular-value cut-off for pre-conditioning.

        For ``'ipm'``: ``'tol_kkt'``, ``'tau'``, ``'delta_init'``, ``'delta_min'``, ``'delta_max'``.

        For ``'slsqp'``: ``'atol'``, ``'rtol'``.
    lbfgs_memory : int, optional, default=10
        (Static) L-BFGS history length for solvers that use L-BFGS.
    maxiter : int, optional, default=None
        (Static) Maximum solver iterations. Defaults to 10000 for
        ``'auglag-lbfgs'``, 100 for ``'ipm'``, and 500 for ``'slsqp'``.
        For ``'auglag-lbfgs'`` this is the outer-loop iteration limit;
        for ``'ipm'`` and ``'slsqp'`` it is the total iteration limit.
    maxiter_inner : int, optional, default=None
        (Static) Maximum inner L-BFGS iterations per outer step (default 500).
        Only used by ``'auglag-lbfgs'``.
    merge_constraints : bool, optional, default=False
        (Static) When ``True``, combines compatible constraint evaluations before solving.
    value_only : bool, optional, default=False
        (Static) When ``True``, skip gradient calculations.
    verbose : int, optional, default=0
        (Static) Print general info when ``verbose==1``.
        Print inside the outer iteration loop, too, when ``verbose==2``.
    '''
    # Normalize metric_name before JIT: lists are unhashable static args.
    if 'metric_name' in kwargs:
        metric_name = kwargs['metric_name']
        if metric_name is None:
            kwargs['metric_name'] = ()
        elif isinstance(metric_name, str):
            kwargs['metric_name'] = (metric_name,)
        else:
            kwargs['metric_name'] = tuple(metric_name)
    try:
        out_dict, qp, dofs_opt, solve_results = _quadcoil_pure(**kwargs)
    except RuntimeError as e:
        # Catch some common Equinox errors due to improper 
        # parameter choices.
        # More to be added. 
        if "_EquinoxRuntimeError: The root is not contained in [lower, upper]" in str(e):
            raise ValueError(
                'Equinox rootfinder failed during winding surface '
                'generation (`SurfaceJAX.gen_winding_surface_dofs`). '
                'Your plasma_coil_distance may have the wrong sign or be too large. '
                'It can also be caused by a self-intersecting plasma surface.'
                'As a visual check, please create a `SurfaceJAX` object for your '
                'plasma surface and call '
                '`SurfaceJAX.uniform_offset(plasma_coil_distance).plot()`.'
                'If this is not caused by improper parameter choices, '
                'please contact the developers.'
            )
        elif "_EquinoxRuntimeError: The root is not contained in [lower, upper]" in str(e):
            raise NotImplementedError(
                'You have encountered an undocumented Equinox error. '
                'These errors are commonly caused by imporper parameter '
                'choices. Please report this error to the developers '
                'so that we can add a troubleshooting guide.'
            )
        else:
            raise e
    return out_dict, qp, dofs_opt, solve_results

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