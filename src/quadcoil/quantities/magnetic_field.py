import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from .current import _K_op
from .quantity import _Quantity

_levi_civita = np.zeros((3, 3, 3), dtype=np.float64)
for _i, _a, _m in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    _levi_civita[_i, _a, _m] = 1.0
for _i, _a, _m in [(0, 2, 1), (1, 0, 2), (2, 1, 0)]:
    _levi_civita[_i, _a, _m] = -1.0

# ----- Implementations -----
def _winding_surface_B_op(qp):
    r'''
    phi-independent operator for the winding-surface field on the plasma:

    ``B = A @ phi + B0``, with shapes ``(n_phi, n_theta, 3, ndofs)`` and
    ``(n_phi, n_theta, 3)``. The ``mu_0 / 4 pi`` factor and the quadrature
    normalization are included.

    Contracting the sources into ``A, B0`` before multiplying by ``phi``
    keeps nested JVPs at shape ``(3, ndofs)`` instead of the unreduced
    source kernel. :meth:`QuadcoilParams.get_op` returns a copy built once
    per solve when one is cached, so a solver loop does not rebuild it.
    '''
    # A port of simsoptpp/winding_surface.cpp
    # Array WindingSurfaceB(Array& points, Array& ws_points, Array& ws_normal, Array& K)
    gamma = qp.plasma_surface.gamma()  # The shapes will be used later
    points = gamma.reshape((-1, 3))
    ws_gamma = qp.winding_surface.gamma()
    ws_points = ws_gamma.reshape((-1, 3))
    ws_normal = qp.winding_surface.normal().reshape((-1, 3))
    nphi = ws_gamma.shape[0]
    ntheta = ws_gamma.shape[1]
    fak = 1e-7  # mu0 divided by 4 * pi factor
    scale = fak / nphi / ntheta
    b_K, c_K = _K_op(qp, winding_surface_mode=True)
    b_K = b_K.reshape((-1, 3, b_K.shape[-1]))
    c_K = c_K.reshape((-1, 3))
    eps = jnp.asarray(_levi_civita)
    nmag = jnp.sqrt(jnp.sum(ws_normal**2, axis=1))

    def compute_op(point):
        r = point.reshape(1, -1) - ws_points
        w = nmag / jnp.sqrt(jnp.sum(r**2, axis=1))**3
        A = jnp.einsum('s,ijk,sjm,sk->im', w, eps, b_K, r)
        B0 = jnp.einsum('s,ijk,sj,sk->i', w, eps, c_K, r)
        return A, B0

    # lax.map(..., batch_size=None) is a sequential scan, not vmap. Keep the
    # original fully vectorized path unless a finite chunk size is set.
    if qp.bs_chunk_size is None:
        A, B0 = vmap(compute_op)(points)
    else:
        A, B0 = jax.lax.map(compute_op, points, batch_size=qp.bs_chunk_size)
    return (A * scale).reshape(gamma.shape + (A.shape[-1],)), (B0 * scale).reshape(gamma.shape)


@jit
def _winding_surface_B(qp, dofs):
    # NOTE: Bnormal_plasma is not necessarily stellarator
    # symmetric even for stellarator symmetric equilibria.
    A, B0 = qp.get_op('winding_surface_B', _winding_surface_B_op)
    return A @ dofs['phi'] + B0
_winding_surface_B_desc_unit = lambda scales: scales["B"]

@jit 
def _Bnormal(qp, dofs):
    # Calculates the field at the plasma due to both 
    # the winding surface and the plasma
    unitnormal_plasma = qp.plasma_surface.unitnormal()
    Bnormal_winding_surface = jnp.sum(unitnormal_plasma * _winding_surface_B(qp, dofs) , axis=-1)
    return Bnormal_winding_surface + qp.Bnormal_plasma

_Bnormal2 = jit(lambda qp, dofs: _Bnormal(qp, dofs)**2)

@jit 
def _f_B(qp, dofs):
    # The nescoil objective.
    Bnormal_val = _Bnormal(qp, dofs)
    return qp.plasma_surface.integrate(Bnormal_val**2/2) * qp.nfp
_f_B_desc_unit = lambda scales: scales["B"]**2 * scales["R0"] * scales["a"]

@jit
def _f_B_normalized_by_Bnormal_IG(qp, dofs):
    # f_B normalized by f_B produced by B_normal_IG.
    # This normalization method does not change the functional
    # form of f_B's dependence on the current potential phi_mn,
    # and the optimum to this objective is the same as f_B.
    f_B_with_unit = _f_B(qp, dofs)
    f_B_IG = _f_B(qp, {'phi': jnp.zeros(qp.ndofs)})
    return(f_B_with_unit / f_B_IG)

# ----- Wrappers -----
# This is a linear 3d vector field. Setting its components would be trivial
# but it should still support <= and >=
winding_surface_B = _Quantity.generate_c2(
    func=_winding_surface_B, 
    compatibility=['<=', '>='], 
    desc_unit=_winding_surface_B_desc_unit,
    convexity_type='linear',
)

# This is a linear scalar field. Setting its components would be trivial
# but it should still support <= and >=
Bnormal = _Quantity.generate_c2(
    func=_Bnormal, 
    compatibility=['<=', '>='], 
    desc_unit=_winding_surface_B_desc_unit,
    convexity_type='linear',
)

# This is a quadratic scalar field. Setting its components would be trivial
# but it should still support <= constraints.
Bnormal2 = _Quantity.generate_c2(
    func=_Bnormal2, 
    compatibility=['<='], 
    desc_unit=lambda scales: scales["B"]**2,
    convexity_type='quadratic',
)

# This is a positive definite quadratic scalar. 
f_B = _Quantity.generate_c2(
    func=_f_B, 
    compatibility=['f', '<='], 
    desc_unit=_f_B_desc_unit,
    convexity_type='quadratic',
)

# This is a positive definite quadratic scalar. 
f_B_normalized_by_Bnormal_IG = _Quantity.generate_c2(
    func=_f_B_normalized_by_Bnormal_IG, 
    compatibility=['f', '<='], 
    desc_unit=lambda scales: 1.,
    convexity_type='quadratic',
)

# This is an l-infinity norm objective.
f_max_Bnormal = _Quantity.generate_linf_norm(
    func=_Bnormal, 
    aux_argname='scaled max Bnormal', 
    desc_unit=_winding_surface_B_desc_unit,
    convexity_type='linear_slack',
)

# This is an l-infinity norm objective.
f_max_Bnormal2 = _Quantity.generate_linf_norm(
    func=_Bnormal2, 
    aux_argname='scaled max Bnormal2', 
    desc_unit=lambda scales: scales["B"]**2,
    convexity_type='quadratic_slack',
    positive_definite=True,
)
