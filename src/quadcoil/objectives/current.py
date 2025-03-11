import jax.numpy as jnp
from quadcoil import sin_or_cos
from jax import jit

@jit
def K(quadcoil_params, current_potential_mn):
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
    cp_m, cp_n = quadcoil_params.make_mn()
    (
        _, # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = diff_helper(
        nfp=quadcoil_params.nfp, 
        cp_m=cp_m, cp_n=cp_n,
        quadpoints_phi=quadcoil_params.eval_surface.quadpoints_phi,
        quadpoints_theta=quadcoil_params.eval_surface.quadpoints_theta,
        stellsym=quadcoil_params.stellsym,
    )
    inv_normN_prime_2d = 1/jnp.linalg.norm(quadcoil_params.eval_surface.normal(), axis=-1)
    dg1 = quadcoil_params.eval_surface.gammadash1()
    dg2 = quadcoil_params.eval_surface.gammadash2()
    G = quadcoil_params.net_poloidal_current
    I = quadcoil_params.net_toroidal_current
    A_K = inv_normN_prime_2d[:, :, None, None] * (
        dg2[:, :, :, None] * (trig_diff_m_i_n_i @ partial_phi)[:, :, None, :]
        - dg1[:, :, :, None] * (trig_diff_m_i_n_i @ partial_theta)[:, :, None, :]
    )
    b_K = inv_normN_prime_2d[:, :, None] * (
        dg2 * G
        - dg1 * I
    )
    return(A_K @ current_potential_mn + b_K)

def K2(quadcoil_params, current_potential_mn):
    return(K(quadcoil_params, current_potential_mn)**2)

@jit
def K_theta(quadcoil_params, current_potential_mn):
    ''' 
    K in the theta direction. Used to eliminate windowpane coils.  
    by K_theta >= -G. (Prevents current from flowing against G).
    '''
    cp_m, cp_n = quadcoil_params.make_mn()
    stellsym = quadcoil_params.stellsym
    nfp = quadcoil_params.nfp
    quadpoints_phi = quadcoil_params.eval_surface.quadpoints_phi
    quadpoints_theta = quadcoil_params.eval_surface.quadpoints_theta
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
    A_K_theta = K_theta
    b_K_theta = net_poloidal_current*jnp.ones((K_theta.shape[0], K_theta.shape[1]))
    return(A_K_theta @ current_potential_mn + b_K_theta)
