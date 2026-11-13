from .classify import classify_problem
from .qcqp_one import solve_qcqp_one, QCQP_ONE_STATUS
from .qp_form import (
    ConicProblem,
    canonicalize_qp,
    canonicalize_socp,
    orthant_cones,
    CONIC_STATUS,
)
from .ipm_conic import solve_conic_ipm, DEFAULT_CONIC_OPTIONS

# Clarabel CPU backend is optional (needs the FFI extension or clarabel wheel).
try:
    from .qp_cpu import solve_conic_clarabel, clarabel_available
except Exception:  # pragma: no cover - missing optional dependency
    def solve_conic_clarabel(*args, **kwargs):
        raise ImportError(
            'Clarabel backend unavailable. Install the clarabel package '
            'or build the ffi/clarabel extension.'
        )

    def clarabel_available():
        return False


SPECIAL_STATUS = {
    'qcqp_one': QCQP_ONE_STATUS,
    'qp': CONIC_STATUS,
    'socp': CONIC_STATUS,
}


__all__ = [
    'classify_problem',
    'solve_qcqp_one',
    'QCQP_ONE_STATUS',
    'ConicProblem',
    'canonicalize_qp',
    'canonicalize_socp',
    'orthant_cones',
    'CONIC_STATUS',
    'solve_conic_ipm',
    'DEFAULT_CONIC_OPTIONS',
    'solve_conic_clarabel',
    'clarabel_available',
    'SPECIAL_STATUS',
]
