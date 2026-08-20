import jax.numpy as jnp
from jax import jit
# For calculating normalization constant
from .quantity import _Quantity

# ----- Implementations -----
@jit
def _Phi_coeff(qp, dofs):
    # Calculates the current potential Phi
    # (dipole density) on a grid on the surface. 
    phi_mn = dofs['phi']
    return phi_mn
_Phi_coeff_unit = lambda scales: _K_desc_unit(scales) * scales["a"]

# ----- Wrappers -----
# For preventing the dofs from blowing up
f_max_phi_dof = _Quantity.generate_linf_norm(
    func=_Phi_coeff, 
    aux_argname='max_Phi_coeff', 
    desc_unit=_Phi_coeff_unit
)