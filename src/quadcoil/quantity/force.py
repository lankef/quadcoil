import jax.numpy as jnp
import numpy as np
from jax.lax import scan, dynamic_slice
from .current import _K, _K_desc_unit, K_cyl
from .self_field import _B_self_cyl
from .quantity import _Quantity
from quadcoil import project_arr_cylindrical

def _force_cyl(qp, dofs):
    jnp.cross(K_cyl(qp, dofs), _B_self_cyl(qp, dofs), axis=-1)

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


# ----- Legacy implementation -----
# The singular integration is now moved to self_field.py

# Calculates the integrands in Robin, Volpe from a number of arrays.
# The arrays needs trimming compared to the outputs
# with a standard cp.
# The inputs are array properties of a surface object
# containing only one field period so that the code is easy to port 
# into c++.
def _force_integrands_xyz(qp, dofs, winding_surface_mode=False):
    ''' 
    Calculates the nominators of the sheet current self-force in Robin, Volpe 2022.
    The K_y dependence is lifted outside the integrals. Therefore, the nominator 
    this function calculates are operators that acts on the QUADCOIL vector
    (scaled Phi, 1). The operator produces a 
    (n_phi_x, n_theta_x, 3(xyz, to act on Ky), 3(xyz))
    After the integral, this will become a (n_phi_y, n_theta_y, 3, 3)
    tensor that acts on K(y) to produce a vector with shape (n_phi_y, n_theta_y, 3, n_dof+1)
    Shape: (n_phi_x, n_theta_x, 3(xyz), 3(xyz)).

    Reminder: Do not use this with BIEST, because the x, y, z components of the vector field 
    has only one period, however many field periods that vector field has.
    ''' 
    ''' Surface properties '''
    if winding_surface_mode=='divide':
        n_phi_1fp = len(qp.winding_surface.quadpoints_phi)//qp.winding_surface.nfp
        surface = qp.winding_surface.copy_and_set_quadpoints(
            quadpoints_phi=qp.winding_surface.quadpoints_phi[:n_phi_1fp], 
            quadpoints_theta=qp.winding_surface.quadpoints_theta, 
        )
    elif winding_surface_mode:
        surface = qp.winding_surface
    else:
        surface = qp.eval_surface
    unitnormal_x = surface.unitnormal()
    unitnormaldash1_x = surface.unitnormaldash(1, 0)
    unitnormaldash2_x = surface.unitnormaldash(0, 1)
    grad1_x, grad2_x = surface.grad_helper()

    ''' K-related quantities '''
    phi_mn = dofs['phi']
    (
        Kdash1_sv_op, 
        Kdash2_sv_op, 
        Kdash1_const,
        Kdash2_const
    ) = qp.Kdash_helper(winding_surface_mode=winding_surface_mode)
    K_x = _K(qp, dofs, winding_surface_mode=winding_surface_mode)
    Kdash1_x = Kdash1_sv_op @ phi_mn + Kdash1_const
    Kdash2_x = Kdash2_sv_op @ phi_mn + Kdash2_const

    ''' nabla_x cdot [pi_x K(y)] K(x) '''
    # divergence of the unit normal
    # Shape: (n_phi_x, n_theta_x)
    div_n_x = (
        jnp.sum(grad1_x * unitnormaldash1_x, axis=-1)
        + jnp.sum(grad2_x * unitnormaldash2_x, axis=-1)
    )

    ''' div_x pi_x '''
    # Shape: (n_phi_x, n_theta_x, 3)
    n_x_dot_grad_n_x = (
        jnp.sum(unitnormal_x * grad1_x, axis=-1)[:, :, None] * unitnormaldash1_x
        + jnp.sum(unitnormal_x * grad2_x, axis=-1)[:, :, None] * unitnormaldash2_x
    )
    # Shape: (n_phi_x, n_theta_x, 3)
    div_pi_x = -(
        div_n_x[:, :, None] * unitnormal_x
        + n_x_dot_grad_n_x
    )

    ''' Integrands '''
    integrand_single = 1e-7 * (
        # Term 1
        # n(x) div n K(x) 
        # - (
        #     grad phi partial_phi 
        #     + grad theta partial_theta
        # ) K(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz))
        (
            unitnormal_x[:, :, :, None] * div_n_x[:, :, None, None] * K_x[:, :, None, :]
        ) 
        - (
            grad1_x[:, :, :, None] * Kdash1_x[:, :, None, :]
            + grad2_x[:, :, :, None] * Kdash2_x[:, :, None, :]
        ) 
        # Term 3
        # K(x) div pi_x 
        # + partial_phi K(x) grad phi
        # + partial_theta K(x) grad theta
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz))
        + (K_x[:, :, :, None] * div_pi_x[:, :, None, :]) 
        + (
            Kdash1_x[:, :, :, None] * grad1_x[:, :, None, :]
            + Kdash2_x[:, :, :, None] * grad2_x[:, :, None, :]
        )
    ) 
    integrand_double = 1e-7 * (
        # Term 2
        # n(x) K(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x)) 
        (unitnormal_x[:, :, :, None] * K_x[:, :, None, :]) 
        # Term 4
        # K(x) n(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        - (K_x[:, :, :, None] * unitnormal_x[:, :, None, :])
    )

    # return (K_x, integrand_single, integrand_double)
    return integrand_single, integrand_double

def _force_xyz_legacy(qp, dofs):
    '''
    Calculates the self-force's x, y, z components on ``qp.eval_surface``.

    The integrand is evaluated over the whole winding surface rather than over
    one field period. Folding the integral onto one field period requires
    expressing the integrand in a basis that rotates with the field period,
    which in turn requires rotating K(r') into, and the result out of, the
    basis at each source point r''. Skipping those rotations introduces an
    O(1) error that does not vanish with refinement. The saving was small
    anyway: computing the integrand over nfp field periods is O(n_phi n_theta),
    against the O((n_phi n_theta)^2) kernel that dominates.
    '''
    (
        single_integrand_xyz,
        double_integrand_xyz
    ) = _force_integrands_xyz(qp, dofs, winding_surface_mode=True)
    single_results, double_results = _integrate_force(
        qp.eval_surface.gamma(),               # (n_phiy, n_thetay, 3)
        qp.winding_surface.gamma(),            # (n_phix*nfp, n_thetax, 3)
        qp.winding_surface.unitnormal(),       # (n_phix*nfp, n_thetax, 3)
        _K(qp, dofs, winding_surface_mode=False),  # (n_phiy, n_thetay, 3)
        qp.winding_surface.da(),               # (n_phix*nfp, n_thetax)
        single_integrand_xyz,                  # (n_phix*nfp, n_thetax, 3, 3)
        double_integrand_xyz,                  # (n_phix*nfp, n_thetax, 3, 3)
        1,
    )
    out = (single_results + double_results) # * 4 * jnp.pi 
    return out

def _force_cyl_legacy(qp, dofs):
    '''
    Calculates the self-force's R, Phi, Z components.
    '''
    return project_arr_cylindrical(
        qp.eval_surface.gamma(),
        _force_xyz_legacy(qp, dofs),
    )

def _integrate_force(
    gamma_y,
    gamma_x,
    unitnormal_x,
    K_y,
    da_x,
    single_integrand,
    double_integrand,
    nfp,
):
    '''
    Performs the singular integration with reduced memory usage.
    Self-interaction is removed structurally (index-based).

    ``nfp`` is the number of field periods the integrands are repeated over.
    ``K_y`` and the integrands must be expressed in the same basis, so this is
    called with ``nfp=1`` and x, y, z components over the whole winding
    surface. See ``_force_xyz_legacy``.
    '''

    # Original diff construction (UNCHANGED)
    diff = gamma_y[:, :, None, None, :] - gamma_x[None, None, :, :, :]

    # Useful shapes
    shapey = list(diff.shape[:2])
    shapex_1fp = list(single_integrand.shape[:2])

    # Reshapes the array for simpler integration over all field periods.
    shape_integral = shapey + [nfp] + shapex_1fp

    # Masks
    n_phiy, n_thetay = shapey
    n_phix, n_thetax = shapex_1fp
    fp_idx    = np.arange(nfp)[None, None, :, None, None]
    phi_xidx  = np.arange(n_phix)[None, None, None, :, None]
    th_xidx   = np.arange(n_thetax)[None, None, None, None, :]
    phi_yidx  = np.arange(n_phiy)[:, None, None, None, None]
    th_yidx   = np.arange(n_thetay)[None, :, None, None, None]
    self_mask = (
        (fp_idx == 0)
        & (phi_xidx == phi_yidx)
        & (th_xidx == th_yidx)
    )

    # Autodiff error handling
    # At x=0, l2 norm is non-differentiable. The 1e-10 * mask adds a small non-zero
    # quantities to sqrt(dist**2) at self-intersecting points to placate autodiff.
    # These points will be later removed using the same mask so it does not 
    # affect the results. Otherwise the non-differentiability propagates and causes 
    # the entire autodiff to be zero.
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10 * self_mask.reshape(diff.shape[:-1]))
    double_layer_denom = jnp.sum(
        diff * unitnormal_x[None, None, :, :, :], axis=-1
    )
    
    dist_reshaped = dist.reshape(shape_integral)
    denom_reshaped = double_layer_denom.reshape(shape_integral)

    # Computing the kernels with masks.
    single_kernel_da = jnp.where(
        self_mask,
        0.0,
        da_x[None, None, None, :, :] / dist_reshaped
    )
    double_kernel_da = jnp.where(
        self_mask,
        0.0,
        da_x[None, None, None, :, :] * denom_reshaped / (dist_reshaped**3)
    )
    
    # Original contractions
    single_contracted = jnp.einsum(
        'ijklm,lmno->ijkno',
        single_kernel_da,
        single_integrand,
    )

    double_contracted = jnp.einsum(
        'ijklm,lmno->ijkno',
        double_kernel_da,
        double_integrand,
    )

    single_results = jnp.einsum(
        'ijk,ijlkm->ijm',
        K_y,
        single_contracted,
    )

    double_results = jnp.einsum(
        'ijk,ijlkm->ijm',
        K_y,
        double_contracted,
    )

    return single_results, double_results

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

