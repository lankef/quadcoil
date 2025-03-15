from . import objective
import jax.numpy as jnp

def get_objective(func_name: str):
    # Takes a string as input and returns 
    # the function with the same name in quadcoil.objective.
    # throws an error 
    if hasattr(objective, func_name):
        func = getattr(objective, func_name)
        if callable(func):
            return func
        else:
            raise ValueError(f"'{func_name}' exists in quadcoil.objective but is not callable.")
    else:
        raise ValueError(f"Function with name '{func_name}' not found in quadcoil.objective.")

def merge_callables(callables):
    # Merge a list of functions that 
    # takes 2 arguments (all functions in the objective submodule does)
    # and combine/flatten their output into 1 big 1d array. 
    # Used to construct constraints.
    def merged_fn(qp, cp_mn):
        outputs = [fn(qp, cp_mn) for fn in callables]
        # Convert scalars to 1D arrays
        outputs = [jnp.atleast_1d(out) for out in outputs]
        # Flatten any array outputs
        outputs = [out.ravel() for out in outputs]
        # Concatenate into a single 1D array
        if len(outputs) == 0:
            return jnp.array([0])
        return jnp.concatenate(outputs, axis=0)
    
    return merged_fn
    
def parse_objectives(objective_name, objective_weight=None): # , objective_unit=None):
    # Converts a tuple of strings (or a single string for one objective only) 
    # and an array of weights into a callable f_tot(a, b) that outputs the weighted 
    # sum of f(a, b). 
    if isinstance(objective_name, str):
        return get_objective(objective_name)
    else:
        if len(objective_name) != len(objective_weight): # or len(objective_name) != len(objective_unit):
            raise ValueError('objective, objective_weight and objective_unit must have the same length.')
        def f_tot(a, b):
            out = 0
            for i in range(len(objective_name)):
                out = out + get_objective(objective_name[i])(a, b) * objective_weight[i] # / objective_unit[i]
            return out
        return f_tot

def parse_constraints(
    constraint_name, # A tuple of function names in quadcoil.objective.
    constraint_type, # A tuple of strings from ">=", "<=", "==".
    constraint_unit, # A tuple of UNTRACED float/ints giving the constraints' order of magnitude.
    constraint_value, # An array of TRACED floats giving the constraint targets.
):
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