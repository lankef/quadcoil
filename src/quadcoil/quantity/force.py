import jax.numpy as jnp
import numpy as np
from .current import _K, _K_desc_unit
from .self_field import _B_self_integrands_xyz, _integrate_B_self
from .quantity import _Quantity
from quadcoil import project_arr_cylindrical

_levi_civita = np.zeros((3, 3, 3), dtype=np.float64)
for _i, _a, _m in [(0,1,2),(1,2,0),(2,0,1)]:
    _levi_civita[_i, _a, _m] = 1.0
for _i, _a, _m in [(0,2,1),(1,0,2),(2,1,0)]:
    _levi_civita[_i, _a, _m] = -1.0

def _force_integrands_xyz(qp, dofs, winding_surface_mode=False):
    r'''
    Returns the rank-2 tensor integrand nominators of the Robin-Volpe (2022)
    self-force.  These are the Levi-Civita duals of the vector integrands
    :math:`\mathbf S, \mathbf D` computed by
    :func:`~quadcoil.quantity.self_field._B_self_integrands_xyz`:

    .. math::

        T^S_{ai} = \varepsilon_{iam} S_m,\quad T^D_{ai} = \varepsilon_{iam} D_m,

    so that :math:`K_a T_{ai} = (\mathbf K \times \mathbf S)_i`.

    The live force path (:func:`_force_xyz`, :func:`_force_cyl`) no longer
    calls this function — it integrates :math:`\mathbf S, \mathbf D` directly
    and crosses with :math:`\mathbf K` afterwards.  This wrapper is retained
    for legacy callers (:func:`_force_cyl_legacy`) and tests.

    Parameters
    ----------
    qp : QuadcoilParams
    dofs : dict
    winding_surface_mode : bool or ``'divide'``, optional
        Same convention as :func:`_B_self_integrands_xyz`.

    Returns
    -------
    integrand_single : ndarray, shape ``(n_phi_x, n_theta_x, 3, 3)``
        :math:`T^S_{ai}`, weighted by :math:`\mu_0/(4\pi)`.
    integrand_double : ndarray, shape ``(n_phi_x, n_theta_x, 3, 3)``
        :math:`T^D_{ai}`, weighted by :math:`\mu_0/(4\pi)`.

    Notes
    -----
    Do not use with BIEST: the x, y, z components are not field-period
    periodic.
    '''
    S, D = _B_self_integrands_xyz(qp, dofs, winding_surface_mode=winding_surface_mode)
    # T_{...ai} = ε_{iam} S_{...m}
    eps = jnp.array(_levi_civita)
    integrand_single = jnp.einsum('iam,...m->...ai', eps, S)
    integrand_double = jnp.einsum('iam,...m->...ai', eps, D)
    return integrand_single, integrand_double

def _force_xyz(qp, dofs):
    r'''
    Calculates the self-force's x, y, z components on ``qp.eval_surface``.

    Implements :math:`\mathbf L'(\mathbf r') = \mathbf K(\mathbf r') \times
    \mathbf B_{self}(\mathbf r')` using the regularized integrand vectors
    :math:`\mathbf S, \mathbf D` from :func:`_B_self_integrands_xyz`.

    The integrand is evaluated over the whole winding surface rather than over
    one field period. Folding onto one field period would require expressing the
    integrand in a basis that rotates with the field period; the saving is small
    anyway (O(n_phi n_theta) vs. the O((n_phi n_theta)^2) kernel).
    '''
    S_xyz, D_xyz = _B_self_integrands_xyz(qp, dofs, winding_surface_mode=True)
    single_results, double_results = _integrate_B_self(
        qp.eval_surface.gamma(),           # (n_phiy, n_thetay, 3)
        qp.winding_surface.gamma(),        # (n_phix*nfp, n_thetax, 3)
        qp.winding_surface.unitnormal(),   # (n_phix*nfp, n_thetax, 3)
        qp.winding_surface.da(),           # (n_phix*nfp, n_thetax)  — nfp=1 treats full surface as one period
        S_xyz,                             # (n_phix*nfp, n_thetax, 3)
        D_xyz,                             # (n_phix*nfp, n_thetax, 3)
        1,
    )
    B_xyz = single_results + double_results  # (n_phiy, n_thetay, 3)
    return jnp.cross(_K(qp, dofs, winding_surface_mode=False), B_xyz, axis=-1)


def _force_cyl(qp, dofs):
    r'''
    Calculates the self-force's R, Phi, Z components on ``qp.eval_surface``.

    Computes the xyz self-force via :func:`_force_xyz` and projects it to
    cylindrical at the eval points.  Projecting an xyz vector field into an
    orthonormal cylindrical basis commutes with the cross product, so

    .. math::

        \mathbf L'_{cyl} = \mathbf K_{cyl} \times \mathbf B_{self,cyl}

    holds on the grid with the same quadrature weights as :func:`_force_xyz`.
    This avoids mixing the eval-point and source-point cylindrical bases that
    occurs if the integrand vectors are projected at the source rather than
    at the eval point.
    '''
    gamma_y = qp.eval_surface.gamma()  # (n_phiy, n_thetay, 3)
    force_xyz = _force_xyz(qp, dofs)   # (n_phiy, n_thetay, 3)
    return project_arr_cylindrical(gamma_y, force_xyz)

# N = T * A * m = T * A/m * m^2
_force_desc_unit = lambda scales: scales["B"] * _K_desc_unit(scales) * scales["a"]**2
_forcel1_desc_unit = lambda scales: scales["B"] * _K_desc_unit(scales) * scales["a"]**2 * scales["R0"] * scales["a"]
_force2_desc_unit = lambda scales: (scales["B"] * _K_desc_unit(scales) * scales["a"]**2)**2

# This is an l-inf norm. We have implemented a template
# in _Quantity. It's non-convex but Shor-relaxable into SDP.
f_max_force_cyl = _Quantity.generate_linf_norm(
    func=_force_cyl, 
    aux_argname='max_force_cyl', 
    desc_unit=_force_desc_unit,
    auto_stellsym=True,
)

# This is an l-inf norm. We have implemented a template
# in _Quantity. It's non-convex but Shor-relaxable into SDP.
f_l1_force_cyl = _Quantity.generate_l1_norm(
    func=_force_cyl, 
    aux_argname='l1_force_cyl', 
    desc_unit=_forcel1_desc_unit,
    auto_stellsym=True,
)

# This is an l-inf norm. We have implemented a template
# in _Quantity. It's non-convex but Shor-relaxable into SDP.
f_max_force2_cyl = _Quantity.generate_linf_norm(
    func=_force_cyl, 
    aux_argname='max_force2_cyl', 
    desc_unit=_force2_desc_unit,
    square=True,
    auto_stellsym=True,
)

# Legacy code

def _force_cyl_legacy(qp, dofs):
    '''
    Calculates the self-force's R, Phi, Z components. Evaluates on 1/3 winding_surface rather than 
    eval_surface. This is not the standard choice of quadrature points so it's discontinued.

    This version uses too much memory and is depreciate, but it's more readable.
    '''
    n_phi_1fp = len(qp.winding_surface.quadpoints_phi)//qp.winding_surface.nfp
    (
        single_integrand_xyz,
        double_integrand_xyz
    ) = _force_integrands_xyz(qp, dofs, winding_surface_mode=True)
    gamma_x = qp.winding_surface.gamma()
    gamma_y = gamma_x[:n_phi_1fp, :, :] # qp.eval_surface.gamma()
    K_y = _K(qp, dofs, winding_surface_mode='divide')
    unitnormal_x = qp.winding_surface.unitnormal()
    # See _force_xyz_legacy: the integral is taken in x, y, z over the whole winding
    # surface, and the resulting vector is projected at the evaluation point.
    single_results, double_results = _integrate_force_legacy(
        gamma_y,          # (n_phiy, n_thetay, 3)
        gamma_x,          # (n_phix*nfp, n_thetax, 3)
        unitnormal_x,     # (n_phix*nfp, n_thetax, 3)
        K_y,              # (n_phiy, n_thetay, 3)
        qp.winding_surface.da(),  # (n_phix*nfp, n_thetax)
        single_integrand_xyz,     # (n_phix*nfp, n_thetax, 3, 3)
        double_integrand_xyz,     # (n_phix*nfp, n_thetax, 3, 3)
        1,
    )
    out = (single_results + double_results) # * 4 * jnp.pi 
    return project_arr_cylindrical(gamma_y, out)

def _integrate_force_legacy(
    gamma_y,          # (n_phiy, n_thetay, 3)
    gamma_x,          # (n_phix*nfp, n_thetax, 3)
    unitnormal_x,     # (n_phix*nfp, n_thetax, 3)
    K_y,              # (n_phiy, n_thetay, 3)
    da_x,             # (n_phix, n_thetax)
    single_integrand,  # (n_phix, n_thetax, 3, 3)
    double_integrand,  # (n_phix, n_thetax, 3, 3)
    nfp,
):
    '''
    Performs the singular integration. Readable but uses too much memory.
    '''
    # Shape: n_phiy, n_thetay, n_phix*nfp, n_thetax, 3(xyz)
    diff = gamma_y[:, :, None, None, :] - gamma_x[None, None, :, :, :] 
    dist = jnp.linalg.norm(diff, axis=-1)
    # Shape: n_phiy, n_thetay, n_phix*nfp, n_thetax
    double_layer_denom = jnp.sum(diff * unitnormal_x[None, None, :, :, :], axis=-1)
    # Shape: n_phiy, n_thetay, n_phix*nfp, n_thetax
    # This step also causes autodiff issues!
    single_layer_kernel = jnp.where(dist!=0, 1/dist, 0)
    double_layer_kernel = jnp.where(dist!=0, double_layer_denom/(dist**3), 0)
    # Calculating useful shapes
    shapey = list(single_layer_kernel.shape[:2])
    shapex_1fp = list(single_integrand.shape[:2])
    shape_integral = shapey + [nfp] + shapex_1fp
    # Shape: n_phiy, n_thetay, nfp, n_phix, n_thetax
    single_layer_kernel_reshaped = single_layer_kernel.reshape(shape_integral)
    double_layer_kernel_reshaped = double_layer_kernel.reshape(shape_integral)
    # Shape: n_phiy, n_thetay, 3(xyz)
    single_results = jnp.sum(
        # Argument of the sum is:
        K_y[:, :, None, None, None, :, None]
        # Shape: n_phiy, n_thetay, nfp, n_phix, n_thetax, 3(xyz, operates on K_y), 3(xyz)
        * single_layer_kernel_reshaped[:, :, :, :, :, None, None]
        * da_x[None, None, None, :, :, None, None]
        # Shape: n_phix, n_thetax, 3(xyz), 3(xyz)
        * single_integrand[None, None, None, :, :, :, :],
        axis=(2, 3, 4, 5)
    )
    # Shape: n_phiy, n_thetay, 3(xyz)
    double_results = jnp.sum(
        # Argument of the sum is:
        K_y[:, :, None, None, None, :, None]
        # Shape: n_phiy, n_thetay, nfp, n_phix, n_thetax, 3(xyz), 3(xyz)
        * double_layer_kernel_reshaped[:, :, :, :, :, None, None]
        * da_x[None, None, None, :, :, None, None]
        # Shape: n_phix, n_thetax, 3(xyz), 3(xyz)
        * double_integrand[None, None, None, :, :, :, :],
        axis=(2, 3, 4, 5)
    )
    return single_results, double_results
