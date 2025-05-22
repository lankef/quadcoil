import jax.numpy as jnp
from jax import jit
# For calculating normalization constant
from .quantity import _Quantity
from .current import _K_desc_unit

# ----- Implementations -----
@jit
def _Phi_coeff(qp, dofs):
    # Calculates the current potential Phi
    # (dipole density) on a grid on the surface. 
    phi_mn = dofs['phi']
    return phi_mn
_Phi_coeff_unit = lambda scales: _K_desc_unit(scales) * scales["a"]

@jit
def _f_Phi2(qp, dofs):
    # Calculates the current potential Phi
    # (dipole density) on a grid on the surface. 
    phi_mn = dofs['phi']
    return jnp.sum(phi_mn**2)
_f_Phi2_coeff_unit = lambda scales: (_K_desc_unit(scales) * scales["a"])**2

# ----- Wrappers -----
# For preventing the dofs from blowing up
f_max_phi_dof = _Quantity.generate_linf_norm(
    func=_Phi_coeff, 
    aux_argname='max_Phi_coeff', 
    desc_unit=_Phi_coeff_unit
)

# This is a positive definite quadratic scalar. 
f_Phi2 = _Quantity.generate_c2(
    func=_f_Phi2, 
    compatibility=['f', '<='], 
    desc_unit=_f_Phi2_coeff_unit,
)