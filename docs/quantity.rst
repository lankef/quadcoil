.. _available_quantities:

Available Quantities
====================

Below is a list of all quantities QUADCOIL supports. To use these quantities 
as objectives, constraints or metrics, simply pass their names into 
``quadcoil.quadcoil``, like:

.. code-block:: python

    out_dict, qp, dofs_opt, solve_results = quadcoil(
        ...
        objective_name=('f_B',),
        constraint_name=('K_theta',),
        constraint_type=('>=',),
        constraint_value=(0.,),
        constraint_unit=(None,),
        metric_name=('f_B',),
        ...
    )

Each quantity declares its own allowed uses, listed in the "Allowed uses" column below:

- ``objective``: can be an objective term. Only scalar quantities can be objectives.
- ``<=``, ``>=``, ``==``: can appear in constraints of that type. For a field 
  (array output), the constraint threshold is a scalar applied to every element.
- ``metric only``: can only be evaluated as a metric.

Each quantity also declares a convexity type, listed in the "Convexity type" column:

- ``'linear'``: affine in the current potential.
- ``'quadratic'``: convex quadratic in the current potential.
- ``'linear_slack'``: affine after introducing slack variables.
- ``'quadratic_slack'``: convex quadratic after introducing slack variables.
- ``None``: non-convex, or convex but not QP/QCQP-representable; solved by AugLag only.

Every quantity can be used as a metric in ``metric_name``. Combinations that are 
trivial (for example, ``==`` on a linear field) or non-convex in an unhelpful way
(for example, ``>=`` on a norm) are disabled to catch typos.

If needed, these quantities can also be
directly imported as functions from ``quadcoil.quantities``
(``quadcoil.quantity`` remains a compatibility alias):

.. code-block:: python

    from quadcoil.quantities import K_theta
    print(K_theta(qp, dofs_opt))

All members of ``quadcoil.quantities`` require the same inputs:

- ``qp : QuadcoilParams`` - Stores the plasma and winding surface information.
- ``dofs_opt : dict`` - The optimized degrees of freedom produced by ``quadcoil.quadcoil``.

Notation
---------

- :math:`\mathbf{B}`: The magnetic field on the plasma boundary.
- :math:`\mathbf{K}`: The sheet current representing the coil set.
- :math:`\hat{\mathbf{n}}`: The unit normal of the plasma boundary or the winding surface.
- :math:`n_{FP}`: The number of field periods.
- :math:`n_\phi^P`: The plasma toroidal resolution, ``len(plasma_quadpoints_phi)``.
- :math:`n_\theta^P`: The plasma poloidal resolution, ``len(plasma_quadpoints_theta)``.
- :math:`n_\phi^E`: The toroidal evaluation resolution on the winding surface, ``len(quadpoints_phi)``. Covers one field period.
- :math:`n_\theta^E`: The poloidal evaluation resolution on the winding surface, ``len(quadpoints_theta)``.
- :math:`n_\phi^W`: The toroidal source resolution on the winding surface, ``len(winding_quadpoints_phi)``. Covers all field periods.
- :math:`n_\theta^W`: The poloidal source resolution on the winding surface, ``len(winding_quadpoints_theta)``.
- :math:`\oint_S da\ f`: The integral of :math:`f` over the full surface :math:`S`
  (``plasma`` or ``WS``, the winding surface). It is computed by integrating over one 
  field period and multiplying by :math:`n_{FP}`.
- :math:`\max_S`: The maximum over the quadrature points of one field period of :math:`S`.

Smoothing
---------

Maximum (``f_max_*``) and L-1 (``f_l1_*``) quantities are not :math:`C^1`. How they are 
handled in a solve depends on the ``smoothing`` argument of ``quadcoil.quadcoil``:

- ``smoothing='slack'``: maxima and L-1 norms are converted exactly into slack 
  variables and inequality constraints.
- ``smoothing='approx'``: maxima are replaced with LogSumExp functions, and absolute
  values in L-1 norms are replaced with smooth approximations. The sharpness is
  set by ``smoothing_params['lse_epsilon']``.

``f_huber_K`` is always smoothed, regardless of ``smoothing``. When evaluated
as a metric or called directly, all quantities return their exact, unsmoothed values.

Basic
-----
.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Allowed uses
     - Convexity type
     - Description
   * - ``'phi_dofs'``
     - :math:`\Phi_{mn}`
     - :math:`(n_\text{dofs})`
     - metric only
     - ``'linear'``
     - The current potential degrees of freedom. Use as a metric to obtain their Jacobians.
   * - ``'f_obj'``
     - :math:`\Sigma_i \frac{w_i}{a_i} f_i`
     - Scalar
     - metric only
     - n/a
     - A special name only available in ``metric_name``. The total weighted objective of the current problem, as seen by the optimizer: normalized by ``objective_unit`` and smoothed according to ``smoothing``.

Magnetic Field
--------------

These objectives are related to the magnetic field on the plasma surface:

.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Allowed uses
     - Convexity type
     - Description
   * - ``'winding_surface_B'``
     - :math:`\mathbf{B}`
     - :math:`(n_\phi^P, n_\theta^P, 3)`
     - ``<=``, ``>=``
     - ``'linear'``
     - The :math:`(x, y, z)` components of the magnetic field from the sheet current on the plasma surface.
   * - ``'Bnormal'``
     - :math:`\mathbf{B}\cdot\hat{\mathbf{n}}`
     - :math:`(n_\phi^P, n_\theta^P)`
     - ``<=``, ``>=``
     - ``'linear'``
     - The normal magnetic field on the plasma surface, including ``Bnormal_plasma``.
   * - ``'Bnormal2'``
     - :math:`(\mathbf{B}\cdot\hat{\mathbf{n}})^2`
     - :math:`(n_\phi^P, n_\theta^P)`
     - ``<=``
     - ``'quadratic'``
     - The squared normal field error on the plasma surface.
   * - ``'f_B'``
     - :math:`f_B\equiv\frac{1}{2}\oint_\text{plasma} da\ (\mathbf{B}\cdot\hat{\mathbf{n}})^2`
     - Scalar
     - objective, ``<=``
     - ``'quadratic'``
     - The integrated normal field error. Also the NESCOIL objective.
   * - ``'f_B_normalized_by_Bnormal_IG'``
     - :math:`\frac{f_B}{f_B(\Phi_{sv}=0)}`
     - Scalar
     - objective, ``<=``
     - ``'quadratic'``
     - :math:`f_B`, normalized by its value with only the net toroidal and poloidal currents.
   * - ``'f_max_Bnormal'``
     - :math:`\max_\text{plasma} |\mathbf{B}\cdot\hat{\mathbf{n}}|`
     - Scalar
     - objective, ``<=``
     - ``'linear_slack'``
     - The maximum normal field error.
   * - ``'f_max_Bnormal2'``
     - :math:`\max_\text{plasma} (\mathbf{B}\cdot\hat{\mathbf{n}})^2`
     - Scalar
     - objective, ``<=``
     - ``'quadratic_slack'``
     - The maximum normal field error squared. A convex quadratic constraint may behave better than a linear constraint.

Current Magnitude and Sign
--------------------------

These objectives are related to the magnitude and sign of the sheet current :math:`\mathbf{K}` that represents a coil set:

.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Allowed uses
     - Convexity type
     - Description
   * - ``'K'``
     - :math:`\mathbf{K}`
     - :math:`(n_\phi^E, n_\theta^E, 3)`
     - ``<=``, ``>=``
     - ``'linear'``
     - The :math:`(x, y, z)` components of the current on the winding surface.
   * - ``'K_cyl'``
     - :math:`\mathbf{K}_{(R, \phi, Z)}`
     - :math:`(n_\phi^E, n_\theta^E, 3)`
     - ``<=``, ``>=``
     - ``'linear'``
     - The :math:`(R, \phi, Z)` components of the current on the winding surface.
   * - ``'K2'``
     - :math:`\|\mathbf{K}\|^2`
     - :math:`(n_\phi^E, n_\theta^E)`
     - ``<=``
     - ``'quadratic'``
     - The current density squared on the winding surface.
   * - ``'K_theta'``
     - :math:`K_\theta`
     - :math:`(n_\phi^E, n_\theta^E)`
     - ``<=``, ``>=``
     - ``'linear'``
     - The poloidal current distribution on the winding surface.
   * - ``'f_K'``
     - :math:`\frac{1}{2}\oint_\text{WS} da\ \|\mathbf{K}\|^2`
     - Scalar
     - objective, ``<=``
     - ``'quadratic'``
     - The integrated current density squared on the winding surface. Also the REGCOIL regularization factor.
   * - ``'f_max_K2'``
     - :math:`\max_\text{WS}\|\mathbf{K}\|^2`
     - Scalar
     - objective, ``<=``
     - ``'quadratic_slack'``
     - The maximum current density squared on the winding surface.
   * - ``'f_huber_K'``
     - :math:`\oint_\text{WS} da\ \|\mathbf{K}\|`
     - Scalar
     - objective, ``<=``
     - ``None``
     - The surface L-1 norm of :math:`\|\mathbf{K}\|`. Promotes sparsity in :math:`\mathbf{K}`. Always smoothed into a pseudo-Huber penalty, :math:`\sqrt{\|\mathbf{K}\|^2/a^2+\epsilon^2}-\epsilon`, during solves.

Current Curvature
-----------------

These objectives are related to the curvature of the sheet current:

.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Allowed uses
     - Convexity type
     - Description
   * - ``'K_dot_grad_K'``
     - :math:`\mathbf{K}\cdot\nabla\mathbf{K}`
     - :math:`(n_\phi^E, n_\theta^E, 3)`
     - ``<=``, ``>=``, ``==``
     - ``None``
     - The :math:`(x, y, z)` components of :math:`\mathbf{K}\cdot\nabla\mathbf{K}` on the winding surface.
   * - ``'K_dot_grad_K_cyl'``
     - :math:`(\mathbf{K}\cdot\nabla\mathbf{K})_{(R, \phi, Z)}`
     - :math:`(n_\phi^E, n_\theta^E, 3)`
     - ``<=``, ``>=``
     - ``None``
     - The :math:`(R, \phi, Z)` components of :math:`\mathbf{K}\cdot\nabla\mathbf{K}` on the winding surface.
   * - ``'f_max_K_dot_grad_K_cyl'``
     - :math:`\max_\text{WS}\|(\mathbf{K}\cdot\nabla\mathbf{K})_{(R, \phi, Z)}\|_\infty`
     - Scalar
     - objective, ``<=``
     - ``None``
     - Maximum :math:`(R, \phi, Z)` component of :math:`\mathbf{K}\cdot\nabla\mathbf{K}` over the winding surface.

Dipole
------

These objectives are related to dipole optimization:

.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Allowed uses
     - Convexity type
     - Description
   * - ``'Phi'``
     - :math:`\Phi_{sv}`
     - :math:`(n_\phi^E, n_\theta^E)`
     - ``<=``, ``>=``
     - ``'linear'``
     - The dipole density distribution on the winding surface. Also referred to as the single valued component of the current potential.
   * - ``'Phi2'``
     - :math:`\Phi_{sv}^2`
     - :math:`(n_\phi^E, n_\theta^E)`
     - ``<=``
     - ``'quadratic'``
     - The squared dipole density distribution on the winding surface.
   * - ``'f_Phi'``
     - :math:`\frac{1}{2}\oint_\text{WS} da\ \Phi_{sv}^2`
     - Scalar
     - objective, ``<=``
     - ``'quadratic'``
     - The integrated squared dipole density on the winding surface.
   * - ``'Phi_with_net_current'``
     - :math:`\Phi = \Phi_{sv} + \frac{G\phi'}{2\pi} + \frac{I\theta'}{2\pi}`
     - :math:`(n_\phi^E, n_\theta^E)`
     - ``<=``, ``>=``
     - ``'linear'``
     - The full current potential on the winding surface, with the secular components representing the net poloidal and toroidal currents :math:`G` and :math:`I`.
   * - ``'f_max_Phi'``
     - :math:`\max_\text{WS}|\Phi_{sv}|`
     - Scalar
     - objective, ``<=``
     - ``'linear_slack'``
     - The maximum dipole density on the winding surface.
   * - ``'f_l1_Phi'``
     - :math:`\oint_\text{WS}da\ |\Phi_{sv}|`
     - Scalar
     - objective, ``<=``
     - ``'linear_slack'``
     - The surface L-1 norm of the dipole density, for sparsity regularization.
   * - ``'f_max_Phi2'``
     - :math:`\max_\text{WS}\Phi_{sv}^2`
     - Scalar
     - objective, ``<=``
     - ``'linear_slack'``
     - The maximum dipole density squared on the winding surface. A convex quadratic constraint may behave better than a linear constraint.
   * - ``'f_max_Phi4'``
     - :math:`\max_\text{WS}\Phi_{sv}^4`
     - Scalar
     - objective, ``<=``
     - ``'quadratic_slack'``
     - The maximum fourth power of the dipole density on the winding surface. Experimental. Added to test the convergence behavior of high-order convex terms.

Self Field
----------

These quantities are related to the self-field of the sheet current, 
:math:`\mathbf{B}_{self}`. This is the regularized field that the sheet current 
produces on itself (the average of the fields on the two sides of the winding surface), 
following Robin & Volpe (2022). It generates the self-force per unit area through
:math:`\mathbf{L}' = \mathbf{K}\times\mathbf{B}_{self}` (see Lorentz Force below). 
:math:`\mathbf{B}_{self}` is affine in :math:`\Phi_{sv}`.

Unlike the other winding-surface quantities, self-field quantities are evaluated on the 
first field period of the winding surface's *source* grid (``winding_quadpoints_phi``, 
``winding_quadpoints_theta``), not on the evaluation grid (``quadpoints_phi``, 
``quadpoints_theta``).

.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Allowed uses
     - Convexity type
     - Description
   * - ``'B_self'``
     - :math:`\mathbf{B}_{self}`
     - :math:`(n_\phi^W/n_{FP}, n_\theta^W, 3)`
     - ``<=``, ``>=``
     - ``'linear'``
     - The :math:`(x, y, z)` components of the self-field on the winding surface.
   * - ``'B_self_cyl'``
     - :math:`\mathbf{B}_{self, (R, \phi, Z)}`
     - :math:`(n_\phi^W/n_{FP}, n_\theta^W, 3)`
     - ``<=``, ``>=``
     - ``'linear'``
     - The :math:`(R, \phi, Z)` components of the self-field on the winding surface.
   * - ``'B2_self'``
     - :math:`\|\mathbf{B}_{self}\|^2`
     - :math:`(n_\phi^W/n_{FP}, n_\theta^W)`
     - ``<=``
     - ``'quadratic'``
     - The squared self-field strength on the winding surface.
   * - ``'B_self_norm'``
     - :math:`\mathbf{B}_{self}\cdot\hat{\mathbf{n}}`
     - :math:`(n_\phi^W/n_{FP}, n_\theta^W)`
     - ``<=``, ``>=``
     - ``'linear'``
     - The normal component of the self-field on the winding surface.
   * - ``'f_B_self'``
     - :math:`\frac{1}{2}\oint_\text{WS} da\ \|\mathbf{B}_{self}\|^2`
     - Scalar
     - objective, ``<=``
     - ``'quadratic'``
     - The integrated squared self-field on the winding surface.
   * - ``'f_max_B2_self'``
     - :math:`\max_\text{WS}\|\mathbf{B}_{self}\|^2`
     - Scalar
     - objective, ``<=``
     - ``'quadratic_slack'``
     - The maximum squared self-field strength on the winding surface.
   * - ``'f_l1_B_self_norm'``
     - :math:`\oint_\text{WS} da\ |\mathbf{B}_{self}\cdot\hat{\mathbf{n}}|`
     - Scalar
     - objective, ``<=``
     - ``'linear_slack'``
     - The surface L-1 norm of the normal self-field on the winding surface.

Lorentz Force
-------------

These objectives are related to the self-force per unit area of the sheet current, 
:math:`\mathbf{F}\equiv\mathbf{L}' = \mathbf{K}\times\mathbf{B}_{self}`, evaluated 
on the evaluation grid of the winding surface.

.. list-table::
   :header-rows: 1

   * - Name
     - Formula
     - Output Shape
     - Allowed uses
     - Convexity type
     - Description
   * - ``'force_xyz'``
     - :math:`\mathbf{F}`
     - :math:`(n_\phi^E, n_\theta^E, 3)`
     - ``<=``, ``>=``
     - ``None``
     - The :math:`(x, y, z)` components of the sheet-current self-force on the winding surface.
   * - ``'force_cyl'``
     - :math:`\mathbf{F}_{(R, \phi, Z)}`
     - :math:`(n_\phi^E, n_\theta^E, 3)`
     - ``<=``, ``>=``
     - ``None``
     - The :math:`(R, \phi, Z)` components of the sheet-current self-force on the winding surface.
   * - ``'f_max_force_cyl'``
     - :math:`\max_\text{WS}\|\mathbf{F}_{(R, \phi, Z)}\|_\infty`
     - Scalar
     - objective, ``<=``
     - ``None``
     - Maximum :math:`(R, \phi, Z)` component of the sheet-current self-force over the winding surface.
   * - ``'f_l1_force_cyl'``
     - :math:`\oint_\text{WS} da\ \|\mathbf{F}_{(R, \phi, Z)}\|_1`
     - Scalar
     - objective, ``<=``
     - ``None``
     - L-1 penalty on the :math:`(R, \phi, Z)` components of the sheet-current self-force.
   * - ``'f_max_force2_cyl'``
     - :math:`\max_\text{WS}\|\mathbf{F}_{(R, \phi, Z)}\|_\infty^2`
     - Scalar
     - objective, ``<=``
     - ``None``
     - Maximum squared :math:`(R, \phi, Z)` component of the sheet-current self-force over the winding surface. Experimental. Added to test the convergence behavior of high-order non-convex terms.
