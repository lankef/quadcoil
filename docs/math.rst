Mathematic backgrounds
======================

QUADCOIL represents sheet current \( \mathbf{K} \) with "current potential" \( \Phi \):

.. math::
   \mathbf{K} = \hat{\mathbf{n}} \cdot \nabla \Phi.

\( \Phi \) can be split into 3 parts:

.. math::
   \Phi = \Phi_{sv} + \frac{G\phi'}{2\pi} + \frac{I\theta'}{2\pi}

The first term represents contributions from a "single-valued" component, \( \Phi_{sv} \). 
\( \Phi_{sv} \) is the degree of freedom QUADCOIL solves for.

The second and third terms represent contributions from the net poloidal current \( G \) and 
the net toroidal current \( I \). \( G \) is determined by the equilibrium, and \( I \) is a free variable.

