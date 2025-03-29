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

def Phi_with_net_current(qp, cp_mn):
    theta2d, phi2d = qp.eval_surface.theta_mesh, qp.eval_surface.phi_mesh
    Phi_val = Phi(qp, cp_mn) \
        + phi2d * qp.net_poloidal_current_amperes \
        + theta2d * qp.net_toroidal_current_amperes
    return(Phi_val)

@jit
def Phi2(qp, cp_mn):
    return Phi(qp, cp_mn)**2

@jit
def Phi_abs(qp, cp_mn):
    return jnp.abs(Phi(qp, cp_mn))

@jit
def f_max_Phi(qp, cp_mn):
    return jnp.max(jnp.abs(Phi(qp, cp_mn)))

@jit
def f_l1_Phi(qp, cp_mn):
    return jnp.sum(jnp.abs(Phi(qp, cp_mn)))
    
@jit
def f_max_Phi2(qp, cp_mn):
    return jnp.max(Phi(qp, cp_mn)**2)