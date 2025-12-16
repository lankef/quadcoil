import jax.numpy as jnp
from jax import jit
from functools import partial
from scipy.constants import mu_0
from .quantity import _Quantity

# ----- Implementations -----
@partial(jit, static_argnames=('winding_surface_mode'))
def _K(qp, dofs, winding_surface_mode=False):
    # winding_surface_mode is for using 
    # one or more field periods.
    phi_mn = dofs['phi']
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
    inv_normN_prime_2d = 1/jnp.linalg.norm(normal, axis=-1)
    G = net_poloidal_current_amperes
    I = net_toroidal_current_amperes
    # This part of the implementation is specific to 
    # Fourier parameterization. May be modified later
    # to accommodate other bases.
    (
        _, # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = qp.diff_helper(winding_surface_mode=winding_surface_mode)
    b_K = inv_normN_prime_2d[:, :, None, None] * (
        dg2[:, :, :, None] * (trig_diff_m_i_n_i @ partial_phi)[:, :, None, :]
        - dg1[:, :, :, None] * (trig_diff_m_i_n_i @ partial_theta)[:, :, None, :]
    )
    c_K = inv_normN_prime_2d[:, :, None] * (
        dg2 * G
        - dg1 * I
    )
    return b_K@phi_mn + c_K
_K_desc_unit = lambda scales: scales["B"] / mu_0 # based on infinite solenoid: B = mu_0 K_pol.

# @jit # Not needed because _K is jitted and this can make compile time excessive
def _K2(qp, dofs):
    return(jnp.sum(_K(qp, dofs)**2, axis=-1))
_K2_desc_unit = lambda scales: _K_desc_unit(scales)**2

@jit
def _K_theta(qp, dofs):
    ''' 
    K in the theta direction. Used to eliminate windowpane coils.  
    by K_theta >= -G. (Prevents current from flowing against G).
    '''
    phi_mn = dofs['phi']
    cp_m, cp_n = qp.make_mn()
    stellsym = qp.stellsym
    nfp = qp.nfp
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
    stellsym = qp.stellsym
    (
        _, # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        _, # partial_phi,
        _, # partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = qp.diff_helper()
    K_theta_shaped = (trig_diff_m_i_n_i@partial_phi)
    K_theta = K_theta_shaped
    A_K_theta = K_theta
    b_K_theta = qp.net_poloidal_current_amperes*jnp.ones((K_theta.shape[0], K_theta.shape[1]))
    return A_K_theta @ phi_mn + b_K_theta
# Shares unit with K

@jit 
def _f_K(qp, dofs):
    K2_val = _K2(qp, dofs)
    return qp.eval_surface.integrate(K2_val/2)*qp.nfp
_f_K_desc_unit = lambda scales: _K_desc_unit(scales)**2 * scales["R0"] * scales["a"]

# ----- Wrappers -----
# This is the xyz component of the 
# current. It's an linear function of the 
# current potential Phi. Although in theory it should be 
# compatible with all types of constraints, setting 
# components of K are arguably a class of trivial 
# constraints, therefore we prohibit it from being
# used as constraints or objectives.
# When compatibility is empty an _Quantity is still a wrapper for 
# a private function.
K = _Quantity.generate_c2(
    func=_K, 
    compatibility=['<=', '>='], 
    desc_unit=_K_desc_unit,
)

# This is a positive definite quadratic vector field.
# Therefore, it cannot be used as an objectivem, but 
# can be used in '==' and '<=' constraints. However, 
# '==' constraints on K^2 is trivial. We therefore prohibit
# it here too.
K2 = _Quantity.generate_c2(
    func=_K2, 
    compatibility=['<='], 
    desc_unit=_K2_desc_unit,
)

f_max_K2 = _Quantity.generate_linf_norm(
    func=_K2, 
    aux_argname='scaled_max_K2_f_max_K2', 
    desc_unit=_K2_desc_unit,
    square=False,
    auto_stellsym=True,
)

# This is a linear scalar field. It's compatible with 
# '<=', '>=', and '==' constraints. '==' seems trivial 
# so we exclude it here to prevent user typo.
K_theta = _Quantity.generate_c2(
    func=_K_theta, 
    compatibility=['<=', '>='], 
    desc_unit=_K_desc_unit,
)

# This is a positive definite quadratic scalar. 
f_K = _Quantity.generate_c2(
    func=_f_K, 
    compatibility=['f', '<='], 
    desc_unit=_f_K_desc_unit,
)
