import jax.numpy as jnp
from .current import _K, _K_desc_unit
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
    if winding_surface_mode:
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

    return (integrand_single, integrand_double)