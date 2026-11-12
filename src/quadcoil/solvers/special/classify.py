from quadcoil.wrapper import get_quantity


def classify_problem(
    objective_name,
    constraint_name,
    constraint_type,
):
    r'''
    Classify a QUADCOIL problem into a special solvable class, or ``None``.

    Currently detects only ``'qcqp_one'``: a single quadratic objective and
    exactly one scalar quadratic ``'<='`` constraint.

    Does not inspect ``smoothing`` or ``merge_constraints``; special detection
    overrides those settings.
    '''
    if not isinstance(objective_name, str):
        return None
    try:
        obj = get_quantity(objective_name)
    except ValueError:
        return None
    if obj.convexity_type != 'quadratic':
        return None

    if isinstance(constraint_name, str):
        cons_names = (constraint_name,)
    elif isinstance(constraint_name, (tuple, list)):
        cons_names = tuple(constraint_name)
    else:
        return None
    if len(cons_names) != 1:
        return None

    if isinstance(constraint_type, str):
        cons_types = (constraint_type,)
    elif isinstance(constraint_type, (tuple, list)):
        cons_types = tuple(constraint_type)
    else:
        return None
    if len(cons_types) != 1 or cons_types[0] != '<=':
        return None

    try:
        cons = get_quantity(cons_names[0])
    except ValueError:
        return None
    if cons.convexity_type != 'quadratic':
        return None

    return 'qcqp_one'
