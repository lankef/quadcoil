# Import packages.
import jax.numpy as jnp
from jax import jit
from functools import partial
import sys
from proto_utils import sin_or_cos
# from simsopt.field import CurrentPotentialSolve, CurrentPotentialFourier
# import simsoptpp as sopp
from proto_operator_helper import diff_helper

@partial(jit, static_argnames=[
    'nfp',
])
def f_B(
        gj, b_e, # Quantities in CurrentPotential
        plasma_normal,
        nfp,
):
    '''
    Produces a dimensionless f_B and K operator that act on X by 
    tr(AX). Also produces a scaling factor current_scale that is an 
    estimate for the order of magnitude of Phi.

    -- Input: 
    A CurrentPotentialSolve.
    -- Output:
    1. An f_B operator with shape (nfod+1, ndof+1). Produces a 
    scalar when acted on X by tr(AX).
    2. f_B operator in REGCOIL
    3. A scalar scaling factor that estimates the order of magnitude
    of Phi. Used when constructing other operators, 
    such as grid_curvature_operator_cylindrical
    and K_operator_cylindrical.
    '''
    # Equivalent to A in regcoil.
    # normN = np.linalg.norm(cpst.plasma_surface.normal().reshape(-1, 3), axis=-1)
    normN = jnp.linalg.norm(plasma_normal.reshape(-1, 3), axis=-1)
    # The matrices may not have been precomputed
    B_normal = gj/jnp.sqrt(normN[:, None])
    ''' f_B operator '''
    ATA = B_normal.T@B_normal
    ATb = B_normal.T@b_e
    bTb = jnp.dot(b_e,b_e)
    A_f_B = ATA/2*nfp
    b_f_B = -ATb*nfp # the factor of 2 cancelled
    c_f_B = bTb/2*nfp
    return(A_f_B, b_f_B, c_f_B, B_normal)

@partial(jit, static_argnames=[
    'nfp', 
    'stellsym',
])
def K_helper(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    '''
    We take advantage of the fj matrix already 
    implemented in CurrentPotentialSolve to calculate K.
    This is a helper method that applies the necessary units 
    and scaling factors. 
    
    When L2_unit=True, the resulting matrices 
    contains the surface element and jacobian for integrating K^2
    over the winding surface.

    When L2_unit=False, the resulting matrices calculates
    the actual components of K.
    '''
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
        stellsym=stellsym,
    )
    inv_normN_prime_2d = 1/jnp.linalg.norm(normal, axis=-1)
    dg1 = gammadash1
    dg2 = gammadash2
    G = net_poloidal_current_amperes
    I = net_toroidal_current_amperes
    b_K = inv_normN_prime_2d[:, :, None, None] * (
        dg2[:, :, :, None] * (trig_diff_m_i_n_i @ partial_phi)[:, :, None, :]
        - dg1[:, :, :, None] * (trig_diff_m_i_n_i @ partial_theta)[:, :, None, :]
    )
    c_K = inv_normN_prime_2d[:, :, None] * (
        dg2 * G
        - dg1 * I
    )
    return(b_K, c_K)
    
@partial(jit, static_argnames=[
    'nfp', 
    'stellsym',
])
def K(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    '''
    Produces the A, b, c for K(phi). Because K is linear in phi, A is blank.

    It cannot directly produce a scalar objective, but can be used 
    for constructing norms (L1, L2). 
    '''
    b_K, c_K = K_helper(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
    )
    # To fill the part of ther operator representing
    # 2nd order coefficients
    A_K = jnp.zeros(
        list(b_K.shape)+[b_K.shape[-1]]
    )
    return(A_K, b_K, c_K)

@partial(jit, static_argnames=[
    'nfp', 
    'stellsym',
])
def K2(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    '''
    An operator that calculates the L2 norm of K.
    Shape: (n_phi (1 field period) x n_theta, n_dof+1, n_dof+1)
    '''
    AK, bK = K_helper(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
    )
    # AK = AK[
    #     :AK.shape[0]//nfp,
    #     :,
    #     :,
    #     :,
    # ]
    # # Take only one field period
    # bK = bK[
    #     :bK.shape[0]//nfp,
    #     :,
    #     :,
    # ]
    # To fill the part of ther operator representing
    # 2nd order coefficients
    A_K2 = jnp.matmul(jnp.swapaxes(AK, -1, -2), AK)
    b_K2 = 2*jnp.sum(AK * bK[:, :, :, None], axis=-2)
    c_K2 = jnp.sum(bK * bK, axis=-1)
    return(A_K2, b_K2, c_K2)

@partial(jit, static_argnames=[
    'nfp', 
    'stellsym',
])
def K_theta(
        net_poloidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    ''' 
    K in the theta direction. Used to eliminate windowpane coils.  
    by K_theta >= -G. (Prevents current from flowing against G).
    '''
    n_harmonic = len(cp_m)
    iden = jnp.identity(n_harmonic)
    # When stellsym is enabled, Phi is a sin fourier series.
    # After a derivative, it becomes a cos fourier series.
    if stellsym:
        trig_choice = 1
    # Otherwise, it's a sin-cos series. After a derivative,
    # it becomes a cos-sin series.
    else:
        trig_choice = jnp.repeat([1,-1], n_harmonic//2)
    partial_phi = -cp_n*trig_choice*iden*nfp*2*jnp.pi
    phi_grid = jnp.pi*2*quadpoints_phi[:, None]
    theta_grid = jnp.pi*2*quadpoints_theta[None, :]
    trig_diff_m_i_n_i = sin_or_cos(
        (cp_m)[None, None, :]*theta_grid[:, :, None]
        -(cp_n*nfp)[None, None, :]*phi_grid[:, :, None],
        -trig_choice
    )
    K_theta_shaped = (trig_diff_m_i_n_i@partial_phi)
    # Take 1 field period
    # K_theta_shaped = K_theta_shaped[:K_theta_shaped.shape[0]//nfp, :]
    # K_theta = K_theta_shaped.reshape((-1, K_theta_shaped.shape[-1]))
    K_theta = K_theta_shaped
    A_K_theta = jnp.zeros((K_theta.shape[0], K_theta.shape[1],  K_theta.shape[2],  K_theta.shape[2]))
    b_K_theta = K_theta
    c_K_theta = net_poloidal_current_amperes*jnp.ones((K_theta.shape[0], K_theta.shape[1]))
    return(A_K_theta, b_K_theta, c_K_theta)
