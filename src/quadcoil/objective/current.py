import jax.numpy as jnp
from quadcoil import sin_or_cos
from jax import jit
from functools import partial

@partial(jit, static_argnames=[
    'winding_surface_mode',
])
def K(qp, cp_mn, winding_surface_mode=False):
    # When winding_surface_mode is set to true, 
    # The evaluation will be done over the full winding surface 
    # instead. This is used when calculating B.
    if winding_surface_mode:
        normal = qp.winding_surface.normal()
        dg1 = qp.winding_surface.gammadash1()
        dg2 = qp.winding_surface.gammadash2()
    else:
        normal = qp.eval_surface.normal()
        dg1 = qp.eval_surface.gammadash1()
        dg2 = qp.eval_surface.gammadash2()
    net_poloidal_current_amperes = qp.net_poloidal_current_amperes
    net_toroidal_current_amperes = qp.net_toroidal_current_amperes
    nfp = qp.nfp
    cp_m, cp_n = qp.make_mn()
    stellsym = qp.stellsym
    (
        _, # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = qp.diff_helper(winding_surface_mode=winding_surface_mode)
    inv_normN_prime_2d = 1/jnp.linalg.norm(normal, axis=-1)
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
    return b_K@cp_mn + c_K

@jit
def K2(qp, cp_mn):
    return(jnp.sum(K(qp, cp_mn)**2, axis=-1))

@jit
def K_theta(qp, cp_mn):
    ''' 
    K in the theta direction. Used to eliminate windowpane coils.  
    by K_theta >= -G. (Prevents current from flowing against G).
    '''
    cp_m, cp_n = qp.make_mn()
    stellsym = qp.stellsym
    nfp = qp.nfp
    quadpoints_phi = qp.eval_surface.quadpoints_phi
    quadpoints_theta = qp.eval_surface.quadpoints_theta
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
    stellsym = qp.stellsym
    (
        _, # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        _, # partial_phi,
        _, # partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = qp.diff_helper(winding_surface_mode=winding_surface_mode)
    K_theta_shaped = (trig_diff_m_i_n_i@partial_phi)
    K_theta = K_theta_shaped
    A_K_theta = K_theta
    b_K_theta = net_poloidal_current*jnp.ones((K_theta.shape[0], K_theta.shape[1]))
    return(A_K_theta @ cp_mn + b_K_theta)


@jit 
def f_K(qp, cp_mn):
    K2_val = K2(qp, cp_mn)
    return qp.eval_surface.integrate(K2_val/2)*qp.nfp
