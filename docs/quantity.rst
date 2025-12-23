.. _available_quantities:

Available Quantities
====================

Below is a list of all quantities QUADCOIL supports. Scalar 
quantities can be used as both objectives and constraints. 
Fields (with array output) can only be used as constraints. 
To use these quantities as objectives, constraints or metrics,
simply pass their names into ``quadcoil.quadcoil``.
``quadcoil.quadcoil``, like:

.. code-block:: python

    phi_mn, out_dict, qp, status = quadcoil(
        ...
        objective_name=('f_B',),
        constraint_name=('K_theta',),
        constraint_type=('>=',),
        constraint_value=(0.,),
        constraint_unit=(None,),
        metric_name=('f_B',),
        ...
    )

If needed, these quantities can also be
directly imported as functions from ``quadcoil.quantity``:

.. code-block:: python

    from quadcoil.quantity import K_theta
    print(K_theta(qp, phi_mn))

All members of ``quadcoil.quantity`` require the same inputs:

- ``qp : QuadcoilParams`` - Stores the plasma and winding surface information.
- ``phi_mn : ndarray`` - The Fourier Coefficients of :math:`\Phi_{sv}` produced by ``quadcoil.quadcoil``.

Notation
---------

- :math:`\mathbf{B}`: The magnetic field on the plasma boundary.
- :math:`\mathbf{K}`: The sheet current representing the coil set.
- :math:`\hat{\mathbf{n}}`: The unit normal of the plasma boundary.
- :math:`n_{FP}`: The number of field periods.
- :math:`n_\phi^P`: The plasma toroidal resolution, ``len(plasma_quadpoints_phi)``.
- :math:`n_\theta^P`: The plasma poloidal resolution, ``len(plasma_quadpoints_theta)``.
- :math:`n_\phi^E`: The toroidal evaluation resolution on the winding surface, ``len(quadpoints_phi)``.
- :math:`n_\theta^E`: The poloidal evaluation resolution on the winding surface, ``len(quadpoints_theta)``.

Magnetic Field
--------------

These objectives are related to the magnetic field on the plasma surface:

.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Description
   * - ``'winding_surface_B'``
     - :math:`\mathbf{B}`
     - :math:`(n_\phi^P, n_\theta^P, 3)`
     - The :math:`(x, y, z)` components of the magnetic field on the plasma surface.
   * - ``'Bnormal'``
     - :math:`\mathbf{B}\cdot\hat{\mathbf{n}}`
     - :math:`(n_\phi^P, n_\theta^P)`
     - The normal magnetic field on the plasma surface.
   * - ``'f_B'``
     - :math:`f_B\equiv\frac{n_{FP}}{2}\oint_\text{plasma} da \|\mathbf{B}\cdot\hat{\mathbf{n}}\|^2`
     - Scalar
     - The integrated normal field error. Also the NESCOIL objective.
   * - ``'f_B_normalized_by_Bnormal_IG'``
     - :math:`\frac{f_B}{f_B(\Phi_{sv}=0)}`
     - Scalar
     - :math:`f_B`, normalized by its value with only the net toroidal and poloidal currents.
   * - ``'f_max_Bnormal_abs'``
     - :math:`\max_\text{plasma surface} \|\mathbf{B}\cdot\hat{\mathbf{n}}\|`
     - Scalar
     - The maximum normal magnetic field strength.
   * - ``'f_max_Bnormal2'``
     - :math:`\max_\text{plasma surface} \|\mathbf{B}\cdot\hat{\mathbf{n}}\|^2`
     - Scalar
     - The maximum normal magnetic field strength squared. A convex quadratic constraint may behave better than a linear constraint.

Current Magnitude and Sign
--------------------------

These objectives are related to the magnitude and sign of the sheet current :math:`\mathbf{K}` that represents a coil set:

.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Description
   * - ``'K'``
     - :math:`\mathbf{K}`
     - :math:`(n_\phi^E, n_\theta^E, 3)`
     - The :math:`(x, y, z)` components of the current on the winding surface.
   * - ``'K2'``
     - :math:`\|\mathbf{K}\|^2`
     - :math:`(n_\phi^E, n_\theta^E)`
     - The current strength on the winding surface.
   * - ``'K_theta'``
     - :math:`K_\theta`
     - :math:`(n_\phi^E, n_\theta^E)`
     - The poloidal current distribution on the winding surface.
   * - ``'f_K'``
     - :math:`\frac{n_{FP}}{2}\oint_\text{WS} da \|\mathbf{K}\|^2`
     - Scalar
     - The integrated magnetic field strength on the winding surface. Also the REGCOIL regularization factor.
   * - ``'f_max_K2'``
     - :math:`\max_\text{WS}\|K\|_2^2`
     - Scalar
     - The integrated magnetic field strength on the winding surface. Also the REGCOIL regularization factor.

Current Curvature
-----------------

These objectives are related to the curvature of the sheet current:

.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Description
   * - ``'K_dot_grad_K'``
     - :math:`\mathbf{K}\cdot\nabla\mathbf{K}`
     - :math:`(n_\phi^E, n_\theta^E, 3)`
     - The :math:`(x, y, z)` components of :math:`\mathbf{K}\cdot\nabla\mathbf{K}` on the winding surface.
   * - ``'K_dot_grad_K_cyl'``
     - :math:`(\mathbf{K}\cdot\nabla\mathbf{K})_{(R, \Phi, Z)}`
     - :math:`(n_\phi^E, n_\theta^E, 3)`
     - The :math:`(R, \Phi, Z)` components of :math:`\mathbf{K}\cdot\nabla\mathbf{K}` on the winding surface.
   * - ``'f_max_K_dot_grad_K_cyl'``
     - :math:`\max_\text{WS}\|(\mathbf{K}\cdot\nabla\mathbf{K})_{(R, \Phi, Z)}\|_\infty`
     - Scalar
     - Maximum :math:`(R, \Phi, Z)` component of :math:`\mathbf{K}\cdot\nabla\mathbf{K}` over the winding surface.

Dipole
------

These objectives are related to dipole optimization:

.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Description
   * - ``'Phi'``
     - :math:`\Phi_{sv}`
     - :math:`(n_\phi^E, n_\theta^E)`
     - The dipole density distribution on the winding surface. Also referred to as the single valued component of the current potential.
   * - ``'Phi_abs'``
     - :math:`\|\Phi_{sv}\|`
     - :math:`(n_\phi^E, n_\theta^E)`
     - The absolute value of the dipole density distribution on the winding surface.
   * - ``'Phi2'``
     - :math:`\|\Phi_{sv}\|^2`
     - :math:`(n_\phi^E, n_\theta^E)`
     - The squared dipole density distribution on the winding surface.
   * - ``'Phi_with_net_current'``
     - :math:`\Phi = \Phi_{sv} + \frac{G\phi'}{2\pi} + \frac{I\theta'}{2\pi}`
     - :math:`(n_\phi^E, n_\theta^E)`
     - The full current potential on the winding surface, with the secular components representing the net poloidal and toroidal currents :math:`G` and :math:`I`.
   * - ``'f_max_Phi'``
     - :math:`\max_\text{WS}\|\Phi_{sv}\|`
     - Scalar
     - The maximum dipole density on the winding surface.
   * - ``'f_l1_Phi'``
     - :math:`\int_\text{WS}dA\|\Phi_{sv}\|`
     - Scalar
     - The sum of the absolute values of dipole density over the winding surface for L1 sparsity regularization.
   * - ``'f_max_Phi2'``
     - :math:`\max_\text{WS}\|\Phi_{sv}\|^2`
     - Scalar
     - The maximum dipole density squared on the winding surface. A convex quadratic constraint may behave better than a linear constraint.

Lorentz Force
-------------

Lorentz force is not yet fully implemented.
