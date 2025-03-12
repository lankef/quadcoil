from . import objective

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