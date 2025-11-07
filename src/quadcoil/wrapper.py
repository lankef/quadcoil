import quadcoil.quantity
from quadcoil.quantity.quantity import _Quantity
import jax.numpy as jnp
from jax import jit
from functools import partial

def get_quantity(func_name: str):
    r'''
    Takes a string as input and returns the function with the 
    same name in ``quadcoil.quantity``.
    throws an error if a function with the same name cannot be found.
    Used to parse ``str`` in ``quadcoil.quadcoil``.

    Parameters
    ----------  
    func_name : str
        Name of the function to find. 
    
    Returns
    -------
    callable
        A callable with the same name in ``quadcoil.quantity``.
    '''
    if hasattr(quadcoil.quantity, func_name):
        func = getattr(quadcoil.quantity, func_name)
        if isinstance(func, _Quantity):
            return func
        else:
            raise ValueError(
                f'\'{func_name}\' exists in quadcoil.quantity but is '\
                'not properly implemented as an instance of _Quantity. '\
                f'Instead, it\'s of type: {str(func)}')
    else:
        raise ValueError(f'\'{func_name}\' not found in quadcoil.quantity.')

def merge_callables(callables):
    r'''
    Merge a tuple of ``callable``s into one that 
    takes 2 arguments (all functions in the ``quadcoil.objective`` do),
    by flattening and concatenating their outputs into an 1D ``array``. 
    Used to construct constraints.

    Parameters
    ----------  
    callables : tuple of callables
        The callables to merge. 
    
    Returns
    -------
    callable
        A callable that returns a 1D ``array``
    '''
    @partial(jit, static_argnums=(2,))
    def merged_fn(qp, dofs, callables=callables):
        outputs = []
        for fn in callables:
            if fn is not None:
                outputs.append(fn(qp, dofs))
        # Convert scalars to 1D arrays
        outputs = [jnp.atleast_1d(out) for out in outputs]
        # Flatten any array outputs
        outputs = [out.ravel() for out in outputs]
        # Concatenate into a single 1D array
        if len(outputs) == 0:
            return jnp.zeros(0)
        return jnp.concatenate(outputs, axis=0)
    
    return merged_fn

def _add_quantity(name, unit, use_case, smoothing, smoothing_params):
    '''
    Finds a quantity from quadcoil.quantity, unpacks and scales it. 
    Also checks compatibility.

    Parameters
    ----------  
    objective_name : str 
        The name of the quantity to find
    unit : scalar or None 
        The unit of the quantity
    use_case : str
        The current type of use case (``'f'``, ``'=='``, ``'<='`` or ``'>='``).

    Returns
    -------
    scaled_f_impl : Callable
        The "under-the-hood" implementation of the quantity
    g_ineq, h_eq : List[Callable]
        The list of inequality and equality constraints.
    unit_callable : Callable(qp: QuadcoilParams)
        The unit as a Callable, in case the scaling factor of the quantity 
        need to be used later, and the scaling mode is set to ``None``.
        The only place where this is currently used is constraint value scaling. 
    scaled_slack_dofs : dict{str: None, Tuple or Callable}
        The accumulator after adding auxiliary variables.
    smoothing : str
        Smoothing mode.
    smoothing_params
        Smoothing parameters.
    '''
    quantity = get_quantity(name)
    scaled_f_impl = quantity.scaled_f_impl(smoothing=smoothing, smoothing_params=smoothing_params)
    scaled_g_ineq_impl = quantity.scaled_g_ineq_impl(smoothing=smoothing)
    scaled_h_eq_impl = quantity.scaled_h_eq_impl(smoothing=smoothing)
    scaled_slack_dofs = quantity.scaled_slack_dofs_init(smoothing=smoothing)
    compatibility = quantity.compatibility
    # Checking compatibility
    if use_case not in compatibility:
        if use_case == 'f':
            raise ValueError(f'{name} cannot be used as an objective term.')
        elif use_case in ['<=', '==', '>=']:
            raise ValueError(f'{name} cannot be used in a {use_case} constraint.')
        else:
            raise ValueError(f'{use_case} is not a valid type of constraint.')
    # Perform scaling
    # When the unit of a quantity is left blank,
    # automatically scale that quantity by its value
    # with only net poloidal/toroidal currents.
    # To accommodate this with the shortest amount of code,
    # we make unit a callable regardless it's a scalar or None.
    if unit is None:
        c0_impl = quantity.__call__
        unit_callable = lambda qp: c0_impl(qp, {'phi': jnp.zeros(qp.ndofs)})
    elif jnp.isscalar(unit):
        unit_callable = lambda qp: unit
    else:
        raise TypeError(
            f'Unit for {name} has incorrect type. The supported '\
            f'types are scalar and None. The provided value is a {type(unit)}.'
        )
    # Apply units (now known) to a function with signature 
    # Callable(qp: QuadcoilParams, dofs: dict, unit: float)
    def apply_unit(fun, unit_callable=unit_callable):
        return lambda qp, dofs, fun=fun, unit_callable=unit_callable:\
            fun(qp, dofs, unit=unit_callable(qp))
        
    # Scaling value
    val_scaled = apply_unit(scaled_f_impl)
    
    # Scaling auxiliary variables' initial values
    scaled_slack_dofs_out = {}
    if not (scaled_slack_dofs is None):
        # We loop over all auxiliary variables' init function, 
        # and substitute in the value of unit with the presently known value.
        for key in scaled_slack_dofs:
            scaled_slack_dofs_new = apply_unit(scaled_slack_dofs[key])
            scaled_slack_dofs_out[key] = scaled_slack_dofs_new
    # Scaling g and h
    if scaled_g_ineq_impl is not None:
        g_ineq_list_scaled = [apply_unit(scaled_g_ineq_impl)]
    else:
        g_ineq_list_scaled = []
    if scaled_h_eq_impl is not None:
        h_eq_list_scaled = [apply_unit(scaled_h_eq_impl)]
    else:
        h_eq_list_scaled = []
    return (
        val_scaled,
        g_ineq_list_scaled,
        h_eq_list_scaled,
        unit_callable,
        scaled_slack_dofs_out
    )


def _parse_objectives(
        objective_name, 
        objective_unit, 
        objective_weight, 
        smoothing,
        smoothing_params,
    ): 
    r'''
    Parses a tuple of ``str`` quantities names (or a single ``str`` for one objective only), 
    an array of weights, and a tuple of units into a ``callable`` 
    that outputs the weighted sum of the corresponding functions in ``quadcoil.objectives``. 

    Parameters
    ----------  
    objective_name : str or tuple of str
        The name of the quantities to combine into an objective funtion. 
        The corresponding functions must all return scalars.
    objective_weight : float or array of float, optional, default=1
        The weight(s) of each objective terms.
    objective_unit : float or tuple of float, optional, default=None
        The normalization factor of each objective term. If set to ``None``
        or a `tuple` with ``None``, then the corresponding objective will 
        be normalized with its value when the current is uniform.
        (or in other words, :math:`\Phi_{sv}=0`).
    smoothing : str
        Smoothing mode.
    smoothing_params
        Smoothing parameters.

    Returns
    -------
    f_tot : callable(QuadcoilParams, ndarray)
        The weighted objective function that maps a QuadcoilParams 
        and an array of :math:`\Phi_{sv}` Fourier coefficients into a scalar.
    g_list : List[Callable or None]
    h_list : List[Callable or None]
    scaled_slack_dofs : dict{str: None, Tuple or Callable}
    '''
    if isinstance(objective_name, str):
        objective_name = (objective_name,)
        objective_unit = (objective_unit,)
        objective_weight = jnp.array([1.,])
    if len(objective_name) != len(objective_weight): # or len(objective_name) != len(objective_unit):
        raise ValueError('objective, objective_weight and objective_unit must have the same length.')
    scaled_slack_dofs = {}
    f_list = []
    g_list = []
    h_list = []
    for i in range(len(objective_name)):
        (
            val_scaled,
            g_ineq_list_scaled,
            h_eq_list_scaled,
            _,
            scaled_slack_dofs_i,   
        ) = _add_quantity(
            name=objective_name[i],
            unit=objective_unit[i],
            use_case='f',
            smoothing=smoothing,
            smoothing_params=smoothing_params,
        )
        scaled_slack_dofs = scaled_slack_dofs | scaled_slack_dofs_i
        f_list.append(val_scaled)
        g_list = g_list + g_ineq_list_scaled
        h_list = h_list + h_eq_list_scaled
    def f_tot(
            qp, dofs, 
            f_list=f_list, 
            objective_weight=objective_weight
        ):
        out = 0
        for i in range(len(f_list)):
            out = out + f_list[i](qp, dofs) * objective_weight[i]
        return out
    return jit(f_tot), g_list, h_list, scaled_slack_dofs

def _parse_constraints(
    constraint_name, 
    constraint_type, 
    constraint_unit, 
    constraint_value, 
    smoothing,
    smoothing_params,
):
    r'''
    Parses a series of tuples and arrays specifying the quantities, 
    types (``'>=', '<=', '=='``)

    Parameters
    ----------  
    constraint_name : tuple of str 
        A tuple of quantity names in ``quadcoil.objective``. The corresponding 
        quantity can be both scalars or a vector fields (``ndarray``).
    constraint_type : tuple of str 
        A tuple of strings. Must consists of ``'>=', '<=', '=='`` only.
    constraint_unit : tuple of float, may contain None
        A tuple of float/ints giving the constraints' order of magnitude.
        If a corresponding element is None, will normalize by the value of the objective 
        when the poloidal/toroidal current is uniform.
    constraint_value : array(float)
        An array of constraint thresholds.
    smoothing : str
        Smoothing mode.
    smoothing_params
        Smoothing parameters.

    Returns
    -------    
    g_list : List[Callable or None]
    h_list : List[Callable or None]
        A list of ``Callable`` for the inequality/equality constraints. Returns will be 
        greater than 0 when the constraints are violated.
    scaled_slack_dofs : dict{str: None, Tuple or Callable}
        A dictionary containing the shapes of the auxiliary variables, or the \
        ``Callables`` required to calculate them
    '''
    # Outputs g_ineq and h_ineq for the augmented lagrangian solver:
    # min f(x)
    # subject to 
    # h(x) = 0, g(x) <= 0
    # First, we parse the constraints from strings into functions.
    n_cons_total = len(constraint_name)
    # Detecting input shape issues
    if (
        n_cons_total != len(constraint_type)
        or n_cons_total != len(constraint_unit)
        or n_cons_total != len(constraint_value)
    ):
        raise ValueError('constraint_name, constraint_type, '\
                         'and constraint_value must have the same length.')
    # Contains a list of callables 
    # that maps (QuadcoilParams, cp_mn)
    # to arrays or scalars
    # that are =0 or <=0 when the constraint is satisfied. 
    g_ineq_list = []
    h_eq_list = []
    scaled_slack_dofs = {}
    for i in range(n_cons_total):
        cons_type_i = constraint_type[i]
        cons_val_i = constraint_value[i]
        (
            cons_func_i_scaled,
            aux_g_ineq_i_scaled,
            aux_h_eq_i_scaled,
            unit_callable_i,
            scaled_slack_dofs_i
        ) = _add_quantity(
            name=constraint_name[i],
            unit=constraint_unit[i],
            use_case=cons_type_i,
            smoothing=smoothing,
            smoothing_params=smoothing_params,
        )
        scaled_slack_dofs = scaled_slack_dofs | scaled_slack_dofs_i
        g_ineq_list = g_ineq_list + aux_g_ineq_i_scaled
        h_eq_list = h_eq_list + aux_h_eq_i_scaled
        # Flipping the sign of >= constraints.
        if cons_type_i == '>=':
            sign_i = -1
        else:
            sign_i = 1
        # This is the proper way to generate a list of 
        # callable without running into the lambda reference 
        # issue.
        def cons_func_centered_i(
                qp, dofs, 
                cons_func_i_scaled=cons_func_i_scaled, 
                cons_val_i=cons_val_i,
                unit_callable_i=unit_callable_i,
                sign=sign_i
            ): 
            # Scaling and centering constraints
            return sign * (cons_func_i_scaled(qp, dofs) - cons_val_i/unit_callable_i(qp))
        # Creating a list of function in h and g.
        if cons_type_i == '==':
            h_eq_list.append(cons_func_centered_i)
        elif cons_type_i in ['<=', '>=']:
            g_ineq_list.append(cons_func_centered_i)
        else:
            raise ValueError('Constraint type can only be \"<=\", \">=\", or \"==\"')
    
    # # Merging the list of function into one 
    # # callable for both g and h.
    # # merge_callables already contains jit.
    # g_ineq = merge_callables(g_ineq_list)
    # h_eq = merge_callables(h_eq_list)
    return g_ineq_list, h_eq_list, scaled_slack_dofs
    
