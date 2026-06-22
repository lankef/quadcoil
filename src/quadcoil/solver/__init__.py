from .auglag import (
    solve_constrained_auglag_lbfgs,
    solve_unconstrained_auglag_lbfgs,
    recover_multipliers,
    # Different shapes for constraint function g 
    gplus_hard,
    gplus_elu,
    gplus_softplus,
)
from .ipm import (
    solve_constrained_ipm,
    solve_unconstrained_ipm,
)
from .slsqp import (
    solve_constrained_slsqp,
    solve_unconstrained_slsqp,
)
from .kkt_adjoint import (
    stationarity_kkt,
    adjoint_kkt,
)
