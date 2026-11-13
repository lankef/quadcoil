from quadcoil.wrapper import get_quantity


def classify_problem(
    objective_name,
    constraint_name,
    constraint_type,
):
    r'''
    Classify a QUADCOIL problem into a special solvable class, or ``None``.

    Detection order
    ---------------
    1. ``'qcqp_one'`` — single quadratic objective, exactly one scalar
       quadratic ``'<='`` constraint.
    2. ``'qp'`` — every objective term is ``'quadratic'`` or ``'linear'``;
       at least one constraint; every constraint is ``'linear'`` with
       type ``'<='`` or ``'>='``.  Equalities and unconstrained problems
       stay on the generic path.
    3. ``'socp'`` — every objective term is ``'quadratic'`` or ``'linear'``;
       at least one constraint; each constraint is either ``'linear'``
       with ``'<='`` / ``'>='`` or ``'quadratic'`` with ``'<='``; and at
       least one quadratic constraint (else it would be ``'qp'``).
       Quadratic ``'>='`` / ``'=='``, ``*_slack``, and non-convex types
       stay on the generic path.

    Does not inspect ``smoothing`` or ``merge_constraints``; special
    detection overrides those settings.
    '''
    # ── Normalise names / types to tuples ────────────────────────────────
    if isinstance(objective_name, str):
        obj_names = (objective_name,)
    elif isinstance(objective_name, (tuple, list)):
        obj_names = tuple(objective_name)
    else:
        return None

    if isinstance(constraint_name, str):
        cons_names = (constraint_name,)
    elif isinstance(constraint_name, (tuple, list)):
        cons_names = tuple(constraint_name)
    else:
        return None

    if isinstance(constraint_type, str):
        cons_types = (constraint_type,)
    elif isinstance(constraint_type, (tuple, list)):
        cons_types = tuple(constraint_type)
    else:
        return None

    if len(cons_names) != len(cons_types):
        return None

    try:
        objs = [get_quantity(n) for n in obj_names]
        cons = [get_quantity(n) for n in cons_names] if cons_names else []
    except ValueError:
        return None

    # ── qcqp_one ─────────────────────────────────────────────────────────
    if (
        len(obj_names) == 1
        and objs[0].convexity_type == 'quadratic'
        and len(cons_names) == 1
        and cons_types[0] == '<='
        and cons[0].convexity_type == 'quadratic'
    ):
        return 'qcqp_one'

    # ── Shared convexity gate for qp / socp ──────────────────────────────
    if len(cons_names) == 0:
        return None
    if any(o.convexity_type not in ('quadratic', 'linear') for o in objs):
        return None

    # ── qp ───────────────────────────────────────────────────────────────
    if (
        all(c.convexity_type == 'linear' for c in cons)
        and all(t in ('<=', '>=') for t in cons_types)
    ):
        return 'qp'

    # ── socp ─────────────────────────────────────────────────────────────
    # Each constraint is either linear (<= / >=) or quadratic (<=).
    # At least one quadratic; otherwise the qp branch above would have
    # already matched.
    has_quadratic = False
    for c, t in zip(cons, cons_types):
        ct = c.convexity_type
        if ct == 'linear':
            if t not in ('<=', '>='):
                return None
        elif ct == 'quadratic':
            if t != '<=':
                return None
            has_quadratic = True
        else:
            # *_slack, None, or unknown — generic path.
            return None
    if has_quadratic:
        return 'socp'
    return None
