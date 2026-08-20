from .quantity import _Quantity
from .dipole import _Phi_desc_unit

# The current potential Fourier coefficients themselves, exposed as a
# quantity so that the metric machinery can produce d(phi)/dy.
_phi_dofs = lambda qp, dofs: dofs['phi']

phi_dofs = _Quantity.generate_c2(
    func=_phi_dofs,
    compatibility=[],
    desc_unit=_Phi_desc_unit,
)
