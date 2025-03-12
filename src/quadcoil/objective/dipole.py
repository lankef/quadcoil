import jax.numpy as jnp
from quadcoil import sin_or_cos
from jax import jit

@jit
def Phi(qp, cp_mn):
    # Calculates the current potential Phi
    # (dipole density) on a grid on the surface. 
    (
        trig_m_i_n_i,
        _, # trig_diff_m_i_n_i,
        _, # partial_phi,
        _, # partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = qp.diff_helper()
    return trig_m_i_n_i@cp_mn

@jit
def f_max_Phi(qp, cp_mn):
    return jnp.max(jnp.abs(Phi(qp, cp_mn)))
    
@jit
def f_max_Phi2(qp, cp_mn):
    return jnp.max(Phi(qp, cp_mn)**2)