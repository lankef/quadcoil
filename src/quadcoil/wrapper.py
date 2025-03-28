from . import objective
import jax.numpy as jnp

def get_objective(func_name: str):
    '''
    Takes a string as input and returns the function with the 
    same name in ``quadcoil.objective``.
    throws an error if a function with the same name cannot be found.
    Used to parse ``str`` in ``quadcoil.quadcoil``.

    Parameters
    ----------  
    func_name : str
        Name of the function to find. 
    
    Returns
    -------
    callable
        A callable with the same name in ``quadcoil.objective``.
    '''
    
    if hasattr(objective, func_name):
        func = getattr(objective, func_name)
        if callable(func):
            return func
        else:
            raise ValueError(f"'{func_name}' exists in quadcoil.objective but is not callable.")
    else:
        raise ValueError(f"Function with name '{func_name}' not found in quadcoil.objective.")

def merge_callables(callables):
    '''
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
    def merged_fn(qp, cp_mn):
        outputs = [fn(qp, cp_mn) for fn in callables]
        # Convert scalars to 1D arrays
        outputs = [jnp.atleast_1d(out) for out in outputs]
        # Flatten any array outputs
        outputs = [out.ravel() for out in outputs]
        # Concatenate into a single 1D array
        if len(outputs) == 0:
            return jnp.zeros(1)
        return jnp.concatenate(outputs, axis=0)
    
    return merged_fn
    
def parse_objectives(objective_name, objective_unit=None, objective_weight=1.): 
    '''
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

    Returns
    -------
    f_tot : callable(QuadcoilParams, ndarray)
        The weighted objective function that maps a QuadcoilParams 
        and an array of :math:`\Phi_{sv}` Fourier coefficients into a scalar.
    '''
    if isinstance(objective_name, str):
        def f_tot(a, b, objective_unit=objective_unit):
            if objective_unit is None:
                # If a normalization unit is not provided, automatically 
                # normalize by the value of the objective with 
                # only the constant net poloidal and toroidal currents.
                objective_unit = get_objective(objective_name)(a, jnp.zeros_like(b))
            return get_objective(objective_name)(a, b) * objective_weight / objective_unit
        return(f_tot)
    else:
        if len(objective_name) != len(objective_weight): # or len(objective_name) != len(objective_unit):
            raise ValueError('objective, objective_weight and objective_unit must have the same length.')
        def f_tot(a, b, objective_unit=objective_unit):
            out = 0
            for i in range(len(objective_name)):
                if objective_unit[i] is None:
                    # If a normalization unit is not provided, automatically 
                    # normalize by the value of the objective with 
                    # only the constant net poloidal and toroidal currents.
                    objective_unit_i = get_objective(objective_name[i])(a, jnp.zeros_like(b))
                else:
                    objective_unit_i = objective_unit[i]
                out = out + get_objective(objective_name[i])(a, b) * objective_weight[i] / objective_unit_i
            return out
        return f_tot

def parse_constraints(
    constraint_name, 
    constraint_type, 
    constraint_unit, 
    constraint_value, 
):
    '''
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

    Returns
    -------
    g_ineq : callable(QuadcoilParams, ndarray)
        A ``callable`` for the inequality constraints. Returns will be 
        greater than 0 when the constraints are violated.
    h_eq : callable(QuadcoilParams, ndarray)
        A ``callable`` for the equality constraints. Returns will be 
        non-zero when the constraints are violated.
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
    scaled_g_ineq_terms = []
    scaled_h_eq_terms = []
    for i in range(n_cons_total):
        cons_func_i = get_objective(constraint_name[i])
        cons_type_i = constraint_type[i]
        cons_val_i = constraint_value[i]
        cons_unit_i = constraint_unit[i]
        # Flipping the sign of >= constraints.
        if cons_type_i == '>=':
            sign_i = -1
        else:
            sign_i = 1
        # This is the proper way to generate a list of 
        # callable without running into the objective_weightbda reference 
        # issue.
        def cons_func_centered_i(
            a, b, 
            cons_func_i=cons_func_i, 
            cons_unit_i=cons_unit_i, 
            cons_val_i=cons_val_i,
            sign=sign_i
        ): 
            # When the unit of a quantity is left blank,
            # automatically scale that quantity by its value
            # with only net poloidal/toroidal currents.
            if cons_unit_i is None:
                cons_unit_i = cons_func_i(a, jnp.zeros_like(b))
            # Scaling and centering constraints
            return sign * (cons_func_i(a, b) - cons_val_i) / cons_unit_i
        # Creating a list of function in h and g.
        if cons_type_i == '==':
            scaled_h_eq_terms.append(cons_func_centered_i)
        elif cons_type_i in ['<=', '>=']:
            scaled_g_ineq_terms.append(cons_func_centered_i)
        else:
            raise ValueError('Constraint type can only be \"<=\", \">=\", or \"==\"')
    
    # Merging the list of function into one 
    # callable for both g and h.
    g_ineq = merge_callables(scaled_g_ineq_terms)
    h_eq = merge_callables(scaled_h_eq_terms)
    return g_ineq, h_eq