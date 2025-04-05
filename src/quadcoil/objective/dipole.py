import jax.numpy as jnp
from quadcoil import sin_or_cos
from jax import jit
# For calculating normalization constant
from quadcoil.objective import K

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
Phi_desc_unit = lambda scales: K_desc_unit(scales) * scales["a"]

def Phi_with_net_current(qp, cp_mn):
    theta2d, phi2d = qp.eval_surface.theta_mesh, qp.eval_surface.phi_mesh
    Phi_val = Phi(qp, cp_mn) \
        + phi2d * qp.net_poloidal_current_amperes \
        + theta2d * qp.net_toroidal_current_amperes
    return(Phi_val)
Phi_with_net_current_desc_unit = lambda scales: Phi_desc_unit(scales)

@jit
def Phi2(qp, cp_mn):
    return Phi(qp, cp_mn)**2
Phi2_desc_unit = lambda scales: Phi_desc_unit(scales)**2

@jit
def Phi_abs(qp, cp_mn):
    return jnp.abs(Phi(qp, cp_mn))
Phi_abs_desc_unit = lambda scales: Phi_desc_unit(scales)

@jit
def f_max_Phi(qp, cp_mn):
    return jnp.max(jnp.abs(Phi(qp, cp_mn)))
f_max_Phi_desc_unit = lambda scales: Phi_desc_unit(scales)

@jit
def f_l1_Phi(qp, cp_mn):
    return qp.eval_surface.integrate(jnp.abs(Phi(qp, cp_mn)))*qp.nfp
f_l1_Phi_desc_unit = lambda scales: Phi_desc_unit(scales) * scales["R0"] * scales["a"]
    
@jit
def f_max_Phi2(qp, cp_mn):
    return jnp.max(Phi(qp, cp_mn)**2)
f_max_Phi2_desc_unit = lambda scales: Phi_desc_unit(scales)**2