Mathematic backgrounds
======================

QUADCOIL uses the Simsopt angle convention, where the toroidal angle
is :math:`\phi` and the poloidal angle is :math:`\theta`. The angles go from 0 to 1
over all field perods. (0 to :math:`1/n_{FP}` for one field period.)

QUADCOIL represents a sheet current approximating a coil set, :math:`\mathbf{K}`, 
with a "current potential" :math:`\Phi`:

.. math::
   \mathbf{K} = \hat{\mathbf{n}} \cdot \nabla \Phi.

:math:`\Phi` can be split into 3 parts:

.. math::
   \Phi = \Phi_{sv} + \frac{G\phi'}{2\pi} + \frac{I\theta'}{2\pi}

The first term represents contributions from a "single-valued" component, :math:`\Phi_{sv}`. 
:math:`\Phi_{sv}` is the degree of freedom QUADCOIL solves for. It can also be thought of as a 
dipole density distribution on the winding surface.

The second and third terms represent contributions from the net poloidal current :math:`G` and 
the net toroidal current :math:`I`. :math:`G` is determined by the equilibrium, and :math:`I` is a free variable.



