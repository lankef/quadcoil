import jax.numpy as jnp
from jax import jit
# For calculating normalization constant
from .current import _K
from .quantity import _Quantity
from .current import _K_desc_unit

# ----- Implementations -----
@jit
def _Phi(qp, dofs):
    # Calculates the current potential Phi
    # (dipole density) on a grid on the surface. 
    phi_mn = dofs['phi']
    (
        trig_m_i_n_i,
        _, # trig_diff_m_i_n_i,
        _, # partial_phi,
        _, # partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = qp.diff_helper()
    return trig_m_i_n_i@phi_mn
_Phi_desc_unit = lambda scales: _K_desc_unit(scales) * scales["a"]

# @jit
def _Phi_with_net_current(qp, dofs):
    theta2d, phi2d = qp.eval_surface.theta_mesh, qp.eval_surface.phi_mesh
    Phi_val = _Phi(qp, dofs) \
        + phi2d * qp.net_poloidal_current_amperes \
        + theta2d * qp.net_toroidal_current_amperes
    return(Phi_val)

# @jit
def _Phi2(qp, dofs):
    return _Phi(qp, dofs)**2

# @jit 
def _f_Phi(qp, dofs):
    _Phi2_val = _Phi2(qp, dofs)
    return qp.eval_surface.integrate(_Phi2_val/2)*qp.nfp
_f_Phi_desc_unit = lambda scales: _Phi_desc_unit(scales)**2 * scales["R0"] * scales["a"]
    
_Phi2_desc_unit = lambda scales: _Phi_desc_unit(scales)**2
_Phi4_desc_unit = lambda scales: _Phi_desc_unit(scales)**4

_f_l1_Phi_desc_unit = lambda scales: _Phi_desc_unit(scales) * scales["R0"] * scales["a"]

# ----- Wrappers -----
# This is a linear scalar field. Again, 
# == is trivial so we prohibit it.
Phi = _Quantity.generate_c2(
    func=_Phi, 
    compatibility=['<=', '>='], 
    desc_unit=_Phi_desc_unit,
)

# This is a linear scalar field. Again, 
# == is trivial so we prohibit it.
Phi_with_net_current = _Quantity.generate_c2(
    func=_Phi_with_net_current, 
    compatibility=['<=', '>='], 
    desc_unit=_Phi_desc_unit,
)

# This is a convex quadratic scalar field. It supports only <=.  
Phi2 = _Quantity.generate_c2(
    func=_Phi2, 
    compatibility=['<='], 
    desc_unit=_Phi2_desc_unit,
)

f_max_Phi = _Quantity.generate_linf_norm(
    func=_Phi, 
    aux_argname='scaled_max_Phi', 
    desc_unit=_Phi_desc_unit,
    auto_stellsym=True,
)

f_max_Phi2 = _Quantity.generate_linf_norm(
    func=_Phi, 
    aux_argname='scaled_max_Phi_f_max_Phi2', 
    desc_unit=_Phi2_desc_unit,
    square=True,
    auto_stellsym=True,
)

f_max_Phi4 = _Quantity.generate_linf_norm_4(
    func=_Phi, 
    aux_argname='scaled_max_Phi2_f_max_Phi4', 
    desc_unit=_Phi4_desc_unit,
)

# f_max_Phi2 = _Quantity.generate_linf_norm(
#     func=_Phi2, 
#     aux_argname='max_Phi2', 
#     desc_unit=_Phi2_desc_unit,
#     positive_definite=True
# )

f_l1_Phi = _Quantity.generate_l1_norm(
    func=_Phi, 
    aux_argname='scaled_abs_Phi', 
    desc_unit=_f_l1_Phi_desc_unit,
    auto_stellsym=True,
)

# This is a positive definite quadratic scalar. 
f_Phi = _Quantity.generate_c2(
    func=_f_Phi, 
    compatibility=['f', '<='], 
    desc_unit=_f_Phi_desc_unit,
)
    