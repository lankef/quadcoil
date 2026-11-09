'''Build the phi-independent field operators once per solve.

``B = A @ phi + B0``. ``A`` and ``B0`` depend on the surfaces and the net
currents, not on the current potential. Building them inside
``y_to_qp`` (outside the solver loop) means each iteration only does the
matrix-vector product, on every device. Quantities fall back to building
the operator themselves when ``qp.cached_ops`` is ``None``.
'''
import jax

from .force import _B_self_eval_op
from .magnetic_field import _winding_surface_B_op
from .self_field import _B_self_ws_op

# name -> callable(qp) -> (A, B0)
OPERATOR_BUILDERS = {
    'winding_surface_B': _winding_surface_B_op,
    'B_self_ws': _B_self_ws_op,
    'B_self_eval': _B_self_eval_op,
}


def build_cached_ops(qp):
    '''Return ``qp`` with every operator in :data:`OPERATOR_BUILDERS` stored.

    An ``optimization_barrier`` is applied to each array so the compiler
    cannot fuse the build back into a loop that only consumes ``A @ phi + B0``.
    Operators the active objective never reads have no consumers and are
    removed by dead-code elimination.
    '''
    cached = {}
    for name, builder in OPERATOR_BUILDERS.items():
        A, B0 = builder(qp)
        cached[name] = (
            jax.lax.optimization_barrier(A),
            jax.lax.optimization_barrier(B0),
        )
    return qp.with_cached_ops(cached)
