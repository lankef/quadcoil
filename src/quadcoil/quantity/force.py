import jax.numpy as jnp
import numpy as np
from jax.lax import scan, dynamic_slice
from .current import _K, _K_desc_unit
from .quantity import _Quantity
from quadcoil import project_arr_cylindrical

# Calculates the integrands in Robin, Volpe from a number of arrays.
# The arrays needs trimming compared to the outputs
# with a standard cp.
# The inputs are array properties of a surface object
# containing only one field period so that the code is easy to port 
# into c++.
def _self_force_integrands_xyz(qp, dofs, winding_surface_mode=False):
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
    unitnormaldash1_x, unitnormaldash2_x = surface.unitnormaldash()
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

def _self_force_cyl(qp, dofs):
    '''
    Calculates the self-force's R, Phi, Z components.

    This version uses too much memory and is depreciate, but it's more readable.
    '''
    n_phi_1fp = len(qp.winding_surface.quadpoints_phi)//qp.winding_surface.nfp
    (
        single_integrand_xyz,
        double_integrand_xyz
    ) = _self_force_integrands_xyz(qp, dofs, winding_surface_mode='divide')
    gamma_x = qp.winding_surface.gamma()
    gamma_y = gamma_x[:n_phi_1fp, :, :] # qp.eval_surface.gamma()
    da_1fp = qp.winding_surface.da()[:n_phi_1fp, :]
    K_y = _K(qp, dofs, winding_surface_mode='divide')
    K_cylindrical = project_arr_cylindrical(
        gamma_y, 
        K_y
    )
    # Integrand over a single FP
    single_integrand_cylindrical = project_arr_cylindrical(
        gamma_y, 
        single_integrand_xyz
    )
    double_integrand_cylindrical = project_arr_cylindrical(
        gamma_y, 
        double_integrand_xyz
    )
    # The projection function assumes that the first 3 components of the array represents the 
    # phi, theta grid and resulting components of the array. Hence the swapaxes.
    # Shape: n_phix, n_thetax, 3(xyz), 3(xyz)
    single_integrand_cylindrical = project_arr_cylindrical(
        gamma_y, 
        single_integrand_cylindrical.swapaxes(2, 3) 
    ).swapaxes(2,3)
    double_integrand_cylindrical = project_arr_cylindrical(
        gamma_y, 
        double_integrand_cylindrical.swapaxes(2, 3) 
    ).swapaxes(2,3)
    unitnormal_x = qp.winding_surface.unitnormal()
    single_results, double_results = _integrate_force(
        gamma_y,          # (n_phiy, n_thetay, 3)
        gamma_x,          # (n_phix*nfp, n_thetax, 3)
        unitnormal_x,     # (n_phix*nfp, n_thetax, 3)
        K_cylindrical,    # (n_phiy, n_thetay, 3)
        da_1fp,           # (n_phix, n_thetax)
        single_integrand_cylindrical,  # (n_phix, n_thetax, 3, 3)
        double_integrand_cylindrical,  # (n_phix, n_thetax, 3, 3)
        qp.nfp,
    )
    out = (single_results + double_results) # * 4 * jnp.pi 
    return out

def _integrate_force_legacy(
    gamma_y,          # (n_phiy, n_thetay, 3)
    gamma_x,          # (n_phix*nfp, n_thetax, 3)
    unitnormal_x,     # (n_phix*nfp, n_thetax, 3)
    K_cylindrical,    # (n_phiy, n_thetay, 3)
    da_1fp,           # (n_phix, n_thetax)
    single_integrand_cylindrical,  # (n_phix, n_thetax, 3, 3)
    double_integrand_cylindrical,  # (n_phix, n_thetax, 3, 3)
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
    shapex_1fp = list(single_integrand_cylindrical.shape[:2])
    shape_integral = shapey + [nfp] + shapex_1fp
    # Shape: n_phiy, n_thetay, nfp, n_phix, n_thetax
    single_layer_kernel_reshaped = single_layer_kernel.reshape(shape_integral)
    double_layer_kernel_reshaped = double_layer_kernel.reshape(shape_integral)
    # Shape: n_phiy, n_thetay, 3(xyz)
    single_results = jnp.sum(
        # Argument of the sum is:
        K_cylindrical[:, :, None, None, None, :, None]
        # Shape: n_phiy, n_thetay, nfp, n_phix, n_thetax, 3(xyz, operates on K_y), 3(xyz)
        * single_layer_kernel_reshaped[:, :, :, :, :, None, None]
        * da_1fp[None, None, None, :, :, None, None]
        # Shape: n_phix, n_thetax, 3(xyz), 3(xyz)
        * single_integrand_cylindrical[None, None, None, :, :, :, :],
        axis=(2, 3, 4, 5)
    )
    # Shape: n_phiy, n_thetay, 3(xyz)
    double_results = jnp.sum(
        # Argument of the sum is:
        K_cylindrical[:, :, None, None, None, :, None]
        # Shape: n_phiy, n_thetay, nfp, n_phix, n_thetax, 3(xyz), 3(xyz)
        * double_layer_kernel_reshaped[:, :, :, :, :, None, None]
        * da_1fp[None, None, None, :, :, None, None]
        # Shape: n_phix, n_thetax, 3(xyz), 3(xyz)
        * double_integrand_cylindrical[None, None, None, :, :, :, :],
        axis=(2, 3, 4, 5)
    )
    return single_results, double_results

def _integrate_force_legacy2(
    gamma_y,          # (n_phiy, n_thetay, 3)
    gamma_x,          # (n_phix*nfp, n_thetax, 3)
    unitnormal_x,     # (n_phix*nfp, n_thetax, 3)
    K_cylindrical,    # (n_phiy, n_thetay, 3)
    da_1fp,           # (n_phix, n_thetax)
    single_integrand_cylindrical,  # (n_phix, n_thetax, 3, 3)
    double_integrand_cylindrical,  # (n_phix, n_thetax, 3, 3)
    nfp,
):
    '''
    Performs the singular integration with reduced memory usage.
    '''
    # Shape: n_phiy, n_thetay, n_phix*nfp, n_thetax, 3(xyz)
    diff = gamma_y[:, :, None, None, :] - gamma_x[None, None, :, :, :] 
    dist = jnp.linalg.norm(diff, axis=-1)
    
    # Shape: n_phiy, n_thetay, n_phix*nfp, n_thetax
    double_layer_denom = jnp.sum(diff * unitnormal_x[None, None, :, :, :], axis=-1)

    # Reshape kernels: n_phiy, n_thetay, nfp, n_phix, n_thetax
    shapey = list(dist.shape[:2])
    shapex_1fp = list(single_integrand_cylindrical.shape[:2])
    shape_integral = shapey + [nfp] + shapex_1fp
    dist_reshaped = dist.reshape(shape_integral)
    
    # Singularity removal based on value causes autodiff errors (it'll trace into both)
    # branches and get infs that messes with the where statement. Since gammax and gammay 
    dist_safe = jnp.where(dist_reshaped == 0, 1.0, dist_reshaped)
    dist_cubed_safe = jnp.where(dist_reshaped**3 == 0, 1.0, dist_reshaped**3)
    single_kernel_da = jnp.where(
        dist_reshaped != 0,
        da_1fp[None, None, None, :, :] / dist_safe,
        0
    )
    double_kernel_da = jnp.where(
        dist_reshaped**3 != 0,
        da_1fp[None, None, None, :, :] * double_layer_denom.reshape(shape_integral) / dist_cubed_safe,
        0
    )
    
    # Contract integrand with kernel*da first to get: n_phiy, n_thetay, nfp, 3, 3
    # einsum: (y1,y2,nfp,x1,x2) * (x1,x2,3,3) -> (y1,y2,nfp,3,3)
    single_contracted = jnp.einsum('ijklm,lmno->ijkno', 
                                     single_kernel_da, 
                                     single_integrand_cylindrical)
    double_contracted = jnp.einsum('ijklm,lmno->ijkno', 
                                     double_kernel_da, 
                                     double_integrand_cylindrical)
    
    # Now contract with K_cylindrical: (y1,y2,3) * (y1,y2,nfp,3,3) -> (y1,y2,3)
    # einsum: (y1,y2,3_k) * (y1,y2,nfp,3_k,3_out) -> (y1,y2,3_out)
    single_results = jnp.einsum('ijk,ijlkm->ijm', K_cylindrical, single_contracted)
    double_results = jnp.einsum('ijk,ijlkm->ijm', K_cylindrical, double_contracted)
    
    return single_results, double_results
def _integrate_force(
    gamma_y,
    gamma_x,
    unitnormal_x,
    K_cylindrical,
    da_1fp,
    single_integrand_cylindrical,
    double_integrand_cylindrical,
    nfp,
):
    '''
    Performs the singular integration with reduced memory usage.
    Self-interaction is removed structurally (index-based).
    '''

    # Original diff construction (UNCHANGED)
    diff = gamma_y[:, :, None, None, :] - gamma_x[None, None, :, :, :]

    # Useful shapes
    shapey = list(diff.shape[:2])
    shapex_1fp = list(single_integrand_cylindrical.shape[:2])

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
        da_1fp[None, None, None, :, :] / dist_reshaped
    )
    double_kernel_da = jnp.where(
        self_mask,
        0.0,
        da_1fp[None, None, None, :, :] * denom_reshaped / (dist_reshaped**3)
    )
    
    # Original contractions
    single_contracted = jnp.einsum(
        'ijklm,lmno->ijkno',
        single_kernel_da,
        single_integrand_cylindrical,
    )

    double_contracted = jnp.einsum(
        'ijklm,lmno->ijkno',
        double_kernel_da,
        double_integrand_cylindrical,
    )

    single_results = jnp.einsum(
        'ijk,ijlkm->ijm',
        K_cylindrical,
        single_contracted,
    )

    double_results = jnp.einsum(
        'ijk,ijlkm->ijm',
        K_cylindrical,
        double_contracted,
    )

    return single_results, double_results
# def _integrate_force_original(
#     gamma_y,
#     gamma_x,
#     unitnormal_x,
#     K_cylindrical,
#     da_1fp,
#     single_integrand_cylindrical,
#     double_integrand_cylindrical,
#     nfp,
# ):
#     '''
#     Performs the singular integration with reduced memory usage.
#     Self-interaction is removed structurally (index-based).
#     '''

#     # Original diff construction (UNCHANGED)
#     diff = gamma_y[:, :, None, None, :] - gamma_x[None, None, :, :, :]
#     dist = jnp.linalg.norm(diff, axis=-1)

#     double_layer_denom = jnp.sum(
#         diff * unitnormal_x[None, None, :, :, :], axis=-1
#     )

#     # Original reshape logic (UNCHANGED)
#     shapey = list(dist.shape[:2])
#     shapex_1fp = list(single_integrand_cylindrical.shape[:2])
#     shape_integral = shapey + [nfp] + shapex_1fp

#     dist_reshaped = dist.reshape(shape_integral)
#     denom_reshaped = double_layer_denom.reshape(shape_integral)

#     # --- NEW: structural self-mask in reshaped layout ---
#     n_phiy, n_thetay = shapey
#     n_phix, n_thetax = shapex_1fp

#     fp_idx    = jnp.arange(nfp)[None, None, :, None, None]
#     phi_xidx  = jnp.arange(n_phix)[None, None, None, :, None]
#     th_xidx   = jnp.arange(n_thetax)[None, None, None, None, :]

#     phi_yidx  = jnp.arange(n_phiy)[:, None, None, None, None]
#     th_yidx   = jnp.arange(n_thetay)[None, :, None, None, None]

#     self_mask = (
#         (fp_idx == 0)
#         & (phi_xidx == phi_yidx)
#         & (th_xidx == th_yidx)
#     )

#     # --- Kernels (same algebra, different mask) ---
#     single_kernel_da = jnp.where(
#         self_mask,
#         0.0,
#         da_1fp[None, None, None, :, :] / dist_reshaped
#     )

#     double_kernel_da = jnp.where(
#         self_mask,
#         0.0,
#         da_1fp[None, None, None, :, :] * denom_reshaped / (dist_reshaped**3)
#     )

#     # Original contractions (UNCHANGED)
#     single_contracted = jnp.einsum(
#         'ijklm,lmno->ijkno',
#         single_kernel_da,
#         single_integrand_cylindrical,
#     )

#     double_contracted = jnp.einsum(
#         'ijklm,lmno->ijkno',
#         double_kernel_da,
#         double_integrand_cylindrical,
#     )

#     single_results = jnp.einsum(
#         'ijk,ijlkm->ijm',
#         K_cylindrical,
#         single_contracted,
#     )

#     double_results = jnp.einsum(
#         'ijk,ijlkm->ijm',
#         K_cylindrical,
#         double_contracted,
#     )

#     return single_results, double_results


# N = T * A * m = T * A/m * m^2
_force_desc_unit = lambda scales: scales["B"] * _K_desc_unit(scales) * scales["a"]**2
_force2_desc_unit = lambda scales: (scales["B"] * _K_desc_unit(scales) * scales["a"]**2)**2

# This is an l-inf norm. We have implemented a template
# in _Quantity. It's non-convex but Shor-relaxable into SDP.
f_max_force_cyl = _Quantity.generate_linf_norm(
    func=_self_force_cyl, 
    aux_argname='max_force_cyl', 
    desc_unit=_force_desc_unit,
    auto_stellsym=True,
)

# This is an l-inf norm. We have implemented a template
# in _Quantity. It's non-convex but Shor-relaxable into SDP.
f_max_force2_cyl = _Quantity.generate_linf_norm(
    func=_self_force_cyl, 
    aux_argname='max_force2_cyl', 
    desc_unit=_force2_desc_unit,
    square=True,
    auto_stellsym=True,
)
