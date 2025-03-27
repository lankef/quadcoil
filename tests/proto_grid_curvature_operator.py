import sys
from proto_utils import sin_or_cos, project_arr_cylindrical

# Import packages.
import jax.numpy as jnp
from jax import jit
from functools import partial
# from simsopt.field import CurrentPotentialFourier #, CurrentPotentialSolve
from proto_operator_helper import norm_helper, Kdash_helper, diff_helper, A_b_c_to_block_operator

@partial(jit, static_argnames=[
    'nfp', 
    'stellsym',
    'L2_unit',
])
def grid_curvature(
        # cp:CurrentPotentialFourier, 
        normal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
        L2_unit=False
):
    ''' 
    Generates a (n_phi, n_theta, 3(xyz), dof, dof) bilinear 
    operator that calculates K cdot grad K on grid points
    specified by 
    cp.winding_surface.quadpoints_phi
    cp.winding_surface.quadpoints_theta
    from 2 sin/sin-cos Fourier current potentials 
    sin or cos(m theta - n phi) with m, n given by
    cp_m
    cp_n.

    Parameters: -----
    - `cp:CurrentPotential` - CurrentPotential object (Fourier representation)
    to optimize

    - ` free_GI:bool` - When `True`, treat I,G as free quantities as well. 
    the operator acts on the vector (Phi, I, G) rather than (Phi)

    Returns: -----
    A (n_phi, n_theta, 3(xyz), n_dof_phi, n_dof_phi) operator 
    that evaluates K dot grad K on grid points. When free_GI is True,
    the operator has shape (n_phi, n_theta, 3(xyz), n_dof_phi+2, n_dof_phi+2) 
    '''

    # winding_surface = cp.winding_surface
    (
        Kdash1_sv_op, 
        Kdash2_sv_op, 
        Kdash1_const,
        Kdash2_const
    ) = Kdash_helper(
        normal=normal,
        gammadash1=gammadash1,
        gammadash2=gammadash2,
        gammadash1dash1=gammadash1dash1,
        gammadash1dash2=gammadash1dash2,
        gammadash2dash2=gammadash2dash2,
        nfp=nfp, 
        cp_m=cp_m, 
        cp_n=cp_n,
        net_poloidal_current_amperes=net_poloidal_current_amperes,
        net_toroidal_current_amperes=net_toroidal_current_amperes,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        stellsym=stellsym
    )
    (
        _, # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = diff_helper(
        nfp=nfp, cp_m=cp_m, cp_n=cp_n,  
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        stellsym=stellsym
    )
    
    ''' Pointwise product with partial r/partial phi or theta'''

    normN_prime_2d, inv_normN_prime_2d = norm_helper(normal)
    
    term_a_op = (trig_diff_m_i_n_i@partial_phi)[:, :, None, :, None]\
        * Kdash2_sv_op[:, :, :, None, :]
    term_b_op = (trig_diff_m_i_n_i@partial_theta)[:, :, None, :, None]\
        * Kdash1_sv_op[:, :, :, None, :]    
    # Includes only the single-valued contributions
    K_dot_grad_K_operator_sv = (term_a_op-term_b_op) * inv_normN_prime_2d[:, :, None, None, None]
    
    # NOTE: direction phi or 1 is poloidal in simsopt.
    G = net_poloidal_current_amperes
    I = net_toroidal_current_amperes
        
    # Constant component of K dot grad K
    # Shape: (n_phi, n_theta, 3(xyz))
    K_dot_grad_K_const = inv_normN_prime_2d[:, :, None]*(G*Kdash2_const - I*Kdash1_const)
    
    # Component of K dot grad K linear in Phi
    # Shape: (n_phi, n_theta, 3(xyz), n_dof)
    K_dot_grad_K_linear = inv_normN_prime_2d[:, :, None, None]*(
        (trig_diff_m_i_n_i@partial_phi)[:, :, None, :]*Kdash2_const[:, :, :, None]
        +G*Kdash2_sv_op
        -(trig_diff_m_i_n_i@partial_theta)[:, :, None, :]*Kdash1_const[:, :, :, None]
        -I*Kdash1_sv_op
    )
    
    # K cdot grad K is (term_a_op-term_b_op)/|N|. 
    # Shape: (n_phi, n_theta, 3(xyz), dof, dof)
    # Multiply by sqrt grid spacing, so that it's L2 norm^2 is K's surface integral.
    if L2_unit:
        # Here, 1/sqrt(|N|) cancels with the Jacobian |N|
        # whenever the square of K dot grad K is integrated.
        dtheta_coil = (quadpoints_theta[1] - quadpoints_theta[0])
        dphi_coil = (quadpoints_phi[1] - quadpoints_phi[0])
        L2_scale = jnp.sqrt(
            dphi_coil * dtheta_coil * normN_prime_2d[:, :, None, None, None]
        )
        K_dot_grad_K_operator_sv *= L2_scale
        K_dot_grad_K_linear *= L2_scale
        K_dot_grad_K_const *= L2_scale

    return(
        K_dot_grad_K_operator_sv,
        K_dot_grad_K_linear,
        K_dot_grad_K_const,
    )

@partial(jit, static_argnames=[
    'nfp', 
    'stellsym',
])
def grid_curvature_cylindrical(
        normal,
        gamma,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    (
        A_KK,
        b_KK,
        c_KK,
    ) = grid_curvature(
        normal=normal,
        gammadash1=gammadash1,
        gammadash2=gammadash2,
        gammadash1dash1=gammadash1dash1,
        gammadash1dash2=gammadash1dash2,
        gammadash2dash2=gammadash2dash2,
        net_poloidal_current_amperes=net_poloidal_current_amperes,
        net_toroidal_current_amperes=net_toroidal_current_amperes,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        nfp=nfp, 
        cp_m=cp_m, 
        cp_n=cp_n,
        stellsym=stellsym,
    )
    A_KK_cyl = project_arr_cylindrical(gamma=gamma, operator=A_KK)
    b_KK_cyl = project_arr_cylindrical(gamma=gamma, operator=b_KK)
    c_KK_cyl = project_arr_cylindrical(gamma=gamma, operator=c_KK)
    return(
        A_KK_cyl,
        b_KK_cyl,
        c_KK_cyl,
    )