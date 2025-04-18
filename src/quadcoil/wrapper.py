import quadcoil.quantity
from quadcoil.quantity.quantity import _Quantity
import jax.numpy as jnp
from jax import jit

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
    def merged_fn(qp, dofs):
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
            return jnp.zeros(1)
        return jnp.concatenate(outputs, axis=0)
    
    return jit(merged_fn)
    
