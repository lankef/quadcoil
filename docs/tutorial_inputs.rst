Tutorial I: running QUADCOIL
================================

``quadcoil.quadcoil()`` is a wrapper that performs all necessary steps needed to generate a sheet current coil set from a plasma boundary, given coil-plasma distance and other engineering requirements. These includes:

1. Generating the winding surface (if not provided).
2. Setting up and solving the QUADCOIL problem.
3. Evaluating the coil metrics and their derivative.

QUADCOIL can be run by simply importing and calling ``quadcoil.quadcoil()``. 
A minimal example can be found in ``examples/simple_example.ipynb``:

.. code-block:: python
  
    from quadcoil import quadcoil
    from simsopt import load

    # Loading an equilibrium's boundary using simsopt
    equil_qs = Vmec('wout_LandremanPaul2021_QA_lowres.nc', keep_all_files=True)
    plasma_surface = equil_qs.boundary
    net_poloidal_current_amperes = equil_qs.external_current()

    nescoil_out_dict, nescoil_qp, nescoil_phi_mn, _ = quadcoil(
        nfp=plasma_surface.nfp,
        stellsym=plasma_surface.stellsym,
        mpol=4, # 4 poloidal harmonics for the current potential
        ntor=4, # 4 toroidal harmonics for the current potential
        plasma_dofs=plasma_surface.get_dofs(),
        plasma_mpol=plasma_surface.mpol,
        plasma_ntor=plasma_surface.ntor,
        net_poloidal_current_amperes=net_poloidal_current_amperes,
        net_toroidal_current_amperes=0.,
        plasma_coil_distance=plasma_surface.minor_radius(),
        # Set the objective to 
        # f_B
        objective_name='f_B',
        objective_weight=None,
        objective_unit=None,
        # Set the output metrics to f_B and f_K
        metric_name=('f_B', 'f_K')
    )

    # Plotting the solution
    from quadcoil.objective import Phi_with_net_current
    import matplotlib.pyplot as plt
    
    plt.contour(
        nescoil_qp.quadpoints_phi, 
        nescoil_qp.quadpoints_theta,
        Phi_with_net_current(nescoil_qp, nescoil_phi_mn), 
        levels=40
    )

Here, we solved the NESCOIL problem (minimizing field error with no additional constraints) on the Landreman-Paul QS configuration. This tutorial will explain how to set up a more compelex coil optimizer/proxy with QUADCOIL using ``quadcoil.quadcoil()``, by going over all input parameters and their physical meaning. These parameters fall in 7 categories:

1. Plasma boundary
2. Sheet current properties (net current, resolution, ...)
3. Coil-plasma distance or winding surface
4. Objective functions for coil optimization. Encodes engineering requirements.
5. Constraints for coil optimization. Encodes engineering requirements.
6. Metrics for evaluating the coil set satisfying these requirements.
7. (Optional) Augmented Lagrangial options.

For readability, we label:

- ❗: Necessary inputs.
- ⭐: Inputs required by optional features.
- The rest are resolution and numerical settings that can be left to the defaults.

All parameters to ``quadcoil.quadcoil()`` are ``ndarrays``, ``str``, or other built-in types. No additional imports are required. Objective functions are chosen by choosing names from the list of available quantities: :ref:`available_quantities`.

1. Defining the plasma boundary
----------------------------------------

We first look at parameters defining the plasma boundary. QUADCOIL currently only supports :math:`(R, Z)` Fourier surfaces. The plasma boundary parameters uses the conventions in ``simsopt.geo.surfaceRZFourier``. More surface implementations will be added.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ❗ ``nfp``
     - ``int``, static
     - N/A
     - Number of field periods. Equivalent to ``SurfaceRZFourier.nfp``.
   * - ❗ ``stellsym``
     - ``bool``, static
     - N/A
     - Whether the coils have stellarator symmetry. Equivalent to ``SurfaceRZFourier.stellsym``.
   * - ❗ ``plasma_mpol``
     - ``int``, static
     - N/A
     - Number of poloidal harmonics. Equivalent to ``SurfaceRZFourier.mpol``.
   * - ❗ ``plasma_ntor``
     - ``int``, static
     - N/A
     - Number of toroidal harmonics. Equivalent to ``SurfaceRZFourier.ntor``.
   * - ❗ ``plasma_dofs``
     - ``ndarray``, traced
     - N/A
     - Plasma dofs. Obtainable from ``SurfaceRZFourier.get_dofs()``.
   * - ``plasma_quadpoints_phi``
     - ``ndarray``, traced
     - ``jnp.linspace(0, 1/nfp, 32, endpoint=False)``
     - Plasma toroidal quadrature points. Must be an 1D array that goes from 0 to ``1/nfp``, without the endpoint. Equivalent to ``SurfaceRZFourier.quadpoints_phi``.
   * - ``plasma_quadpoints_theta``
     - ``ndarray``, traced
     - ``jnp.linspace(0, 1, 34, endpoint=False)``
     - Plasma poloidal quadrature points. Must be an 1D array that goes from 0 to 1, without the endpoint. Equivalent to ``SurfaceRZFourier.quadpoints_theta``.
   * - ⭐ ``Bnormal_plasma``
     - ``ndarray``, traced
     - ``0``
     - Normal magnetic field on the plasma boundary, :math:`B_\text{normal}^\text{plasma}`. Zero by default. Must be ``len(plasma_quadpoints_phi)`` x ``len(plasma_quadpoints_theta)``.
   * - ⭐ ``plasma_stellsym``
     - ``bool``, static
     - ``True``
     - Whether the plasma have stellarator symmetry. Equivalent to ``SurfaceRZFourier.stellsym``

Here, ``plasma.dofs`` can be obtained from Simsopt using ``simsopt.geo.SurfaceRZFourier.get_dofs()``.

2. Setting net currents and resolutions
------------------------------------------

These parameters defines basic properties of the sheet current solutions.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ❗ ``net_poloidal_current_amperes``
     - ``float``, traced
     - N/A
     - The net poloidal current :math:`G` in Amperes. Determined by the equilibrium.
   * - ⭐ ``net_toroidal_current_amperes``
     - ``float``, traced
     - 0
     - The net toroidal current :math:`I` in Amperes. A free variable.
   * - ``mpol``
     - ``int``, static
     - 6
     - The number of poloidal harmonics in :math:`\Phi_{sv}`
   * - ``ntor``
     - ``int``, static
     - 4
     - The number of toroidal harmonics in :math:`\Phi_{sv}`
   * - ``quadpoints_phi``
     - ``ndarray``, traced
     - The first field period from the winding surface
     - Toroidal quadrature points on the winding surface for evaluating coil quantities. Must be an 1D array that goes from 0 to ``1/nfp``, without the endpoint. Equivalent to ``SurfaceRZFourier.quadpoints_phi``
   * - ``quadpoints_theta``
     - ``ndarray``, traced
     - The winding surface quadpoints
     - Poloidal quadrature points on the winding surface for evaluating coil quantities.
   * - ``phi_init``
     - ``ndarray``, traced
     - All zeros
     - Initial state of x. All zeros by default. 
   * - ``phi_unit``
     - ``float``, traced
     - :math:`\frac{d_{cs}B_\text{normal}^\text{plasma}}{\mu_0}`.
     - A normalization constant :math:`a_\Phi`, so that :math:`\Phi_{sv}`'s Fourier coefficients satisfy :math:`\Phi_{sv, M, N}/a_\Phi\approx O(1)`. Automatically calculated by default.

3. Choosing the winding surface
--------------------------------------------

The winding surface can either be generated automatically or specified.

Auto-generate
~~~~~~~~~~~~~

QUADCOIL can automatically generate winding surfaces when used as an equilibrium-stage coil complexity proxy. To auto generate the winding surface, set:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ❗ ``plasma_coil_distance``
     - ``float``, traced
     - ``None``, but **must be specified** to auto-generate winding surface.
     - The coil-plasma distance :math:`d_{cs}`.
   * - ``winding_mpol``
     - ``int``, static
     - 6
     - The number of poloidal harmonics in the winding surface.
   * - ``winding_ntor``
     - ``int``, static
     - 5
     - The number of toroidal harmonics in the winding surface.
   * - ``winding_surface_generator``
     - ``callable``, static. Must have the correct signatures
     - ``gen_winding_surface_arc``
     - The winding surface generator.
   * - ⭐ ``winding_stellsym``
     - ``bool``, static
     - ``True``
     - Whether the winding surface have stellarator symmetry. Equivalent to ``SurfaceRZFourier.stellsym``.

Known winding surface
~~~~~~~~~~~~~~~~~~~~~

QUADCOIL can also run on a known winding surface for tasks such as blanket optimization. To specify a winding surface, set:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ❗ ``winding_dofs``
     - ``ndarray``, traced
     - ``None``, but **must be specified** to auto-generate winding surface.
     - The winding surface degrees of freedom.
   * - ❗ ``winding_mpol``
     - ``int``, static
     - ``6``, but **must change match** ``winding_dofs``.
     - The winding surface poloidal harmonic numbers.
   * - ❗ ``winding_ntor``
     - ``int``, static
     - ``5``, but **must change match** ``winding_dofs``.
     - The winding surface toroidal harmonic numbers.
   * - ``winding_quadpoints_phi``
     - ``ndarray``, traced
     - ``jnp.linspace(0, 1, 32*nfp, endpoint=False)``
     - Toroidal quadrature points on the winding surface for evaluating surface integrals. Must be an 1D array that goes from 0 to 1, without the endpoint. Equivalent to SurfaceRZFourier.quadpoints_phi
   * - ``winding_quadpoints_theta``
     - ``ndarray``, traced
     - ``jnp.linspace(0, 1, 34, endpoint=False)``
     - Poloidal quadrature points on the winding surface for evaluating integrals.
   * - ⭐ ``winding_stellsym``
     - ``bool``, static
     - ``True``
     - Whether the winding surface have stellarator symmetry. Equivalent to ``SurfaceRZFourier.stellsym``.

4. Choosing the objective function(s)
----------------------------------------

QUADCOIL can perform single or multi-objective optimization. Objectives and constraints in QUADCOIL must be selected from :ref:`available_quantities` by entering their names as ``str``\s. The quantity selected as objective(s) must have scalar output. 

**CAUTION!**

As we will see below, every objective and constraint **must be accompanied** by a normalization constant, referred to as ``<something>_unit``, that scales the objective/constraint to :math:`O(1)`. Without this constant, the optimizer will not behave well. QUADCOIL can automatically calculate these constants from :math:`f(\Phi_{sv}=0)`, but this can be inaccurate. We **strongly** advise providing a value. For the objective, the constant can come from an optimum :math:`\Phi_{sv}^*` that uses automatically calculated normalizing constants. For the constraint, the constant can be the constraint threshold. 


Single-objective
~~~~~~~~~~~~~~~~

In this mode, QUADDCOIL will minimize one quantity selected from the list. To select single-objective mode, pass a single ``str`` as the ``objective_name``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ⭐ ``objective_name``
     - ``str``, static
     - ``'f_B'``
     - The objective function :math:`f`. By default the NESCOIL objective.
   * - ⭐ ``objective_unit``
     - ``float``, traced
     - :math:`f(\Phi_{sv}=0)`
     - A normalization constant :math:`a`, so that :math:`f/c\approx O(1)`. Will be automatically calculated from :math:`f`'s with only current from :math:`I, G`.

Multi-objective
~~~~~~~~~~~~~~~

While performing multi-objective optimization, QUADCOIL will minimize a weighted sum of multiple quantities:

.. math::

    f(\Phi_{sv}) = \Sigma_i \frac{w_i}{a_i} f_i(\Phi_{sv}).

Here, :math:`w_i` are the weights/regularization strength of each objective term, and :math:`a_i` are normalization constants so that :math:`f_i/a_i\approx O(1)`, and the optimizer is well-behaved. In gradient calculations, :math:`\nabla_{w_i}` will be available, but **not** :math:`\nabla_{a_i}`. Note that multi-objective problems can have constraints too.

To select multi-objective mode, pass a ``tuple`` as ``objective_name``. ``objective_name``, ``objective_weight``, ``objective_unit`` Must have the same length.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ⭐ ``objective_name``
     - ``tuple`` of ``str``, static
     - ``'f_B'``
     - A tuple of objective terms :math:`f_i`.
   * - ⭐ ``objective_weight``
     - ``ndarray``, traced
     - ``None``
     - An array of weights :math:`w_i`.
   * - ⭐ ``objective_unit``
     - ``tuple`` of ``float``, traced
     - ``None``
     - A tuple of normalization constants :math:`a_i`. If an element is ``None``, :math:`a_i` will be set to :math:`f_i(\Phi_{sv}=0)`.

5. Setting constraints
--------------------------

QUADCOIL supports both equality and inequality constraints, on scalar quantities or fields:

.. math::

    \frac{g_j(\Phi_{sv})}{b_j}\leq \text{ or } \geq\text{ or } = \frac{p_j}{b_j} \\
    ...

Like in multi-objective optimization, QUADCOIL will calculate :math:`\nabla_{p_j}`, but not :math:`\nabla_{b_j}`.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ⭐ ``constraint_name``
     - ``tuple`` of ``str``, static
     - ``()``
     - A tuple of constraint names. No constraints by default.
   * - ⭐ ``constraint_type``
     - ``tuple`` of ``str``, static
     - ``()``
     - A tuple of constraint types. Choose from ``>=``, ``<=`` and ``==``.
   * - ⭐ ``constraint_unit``
     - ``tuple`` of ``float``, traced
     - ``()``
     - A tuple of normalization constants, :math:`b_j`, so that :math:`g_j/b_j` and :math:`p_j/b_j\approx O(1)`. If an element is ``None``, :math:`a_i` will be set to :math:`f_i(\Phi_{sv}=0)`.
   * - ⭐ ``constraint_value``
     - ``ndarray``, traced
     - ``()``
     - An array of constraint thresholds, :math:`p_j`.

6. Important numerical settings.
--------------------------------

The following are important numerical settings.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ⭐ ``smoothing``
     - ``str``, static
     - ``'slack'``
     - Smoothing method for non-smooth problems.
   * - ``smoothing_params``
     - ``dict``, traced
     - ``{'lse_epsilon': 1e-3}``
     - Smoothing parameters. Only used when needed.
   * - ⭐ ``value_only``
     - ``bool``, static
     - ``False``
     - When ``True``, skips adjoint gradient calculation and greatly increases speed. 
   * - ``convex``
     - ``bool``, static
     - ``False``
     - When ``True``, adds positive-semidefinite flags to certain matrix solves to improve speed and accuracy.
   * - ``verbose``
     - ``int``, static
     - ``0``
     - Takes int from 0 to 3. 1 prints important info only. 2 prints outer Auglag loops. 3 prints inner Auglag loops.

QUADCOIL performs optimization on non-smooth objectives. For the optimizer to converge well, it has to
convert the non-smooth problem to a smooth problem. The currently supported values for ``'smoothing'`` are:

.. list-table::
   :header-rows: 1
   * - Value for ``smoothing``
     - Type
     - Advantages
     - Disadvantages
   * - ``'slack'``
     - Exact conversion using slasck variables.
     - More accurate optimimum.
     - Inaccurate adjoint differentiation. Slower, higher memory usage, and high constraint count.
   * - ``'approx'``
     - Approximate conversion by replacing maximum with LogSumExp functions.
     - Accurate adjoint gradients. Faster, lower memory usage, and low constraint count.
     - Less accurate optimum.

We advise using ``'slack'`` for generating coil solutions and ``'approx'`` for quasi-single-stage optimization.

7. Setting coil metrics
---------------------------

We are almost there. After an optimum coil set :math:`\Phi^*_{sv}` is found, QUADCOIL will evaluate a list of coil quality metrics :math:`M_l(\Phi^*_{sv})`. Derivatives w.r.t. the following quantities will also be available:

- ``plasma_dofs``
- ``net_poloidal_current_amperes``
- ``net_toroidal_current_amperes``
- ``plasma_coil_distance`` or ``winding_dofs``
- ``objective_weight`` (if enabled)
- ``constraint_value`` (if enabled)

We still choose these metrics by giving a ``tuple`` containing their names:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ⭐ ``metric_name``
     - ``tuple`` of ``str``, static
     - ``('f_B', 'f_K')``
     - A tuple of metric names.

8. (Optional) Tweaking the augmented Lagrangian solver
-------------------------------------------------------------------------

The augmented Lagrangian solver can be fine-tuned for a specific problem if the default parameters do not yield sufficiently accurate results.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ``c_init``
     - ``float``, traced
     - ``1.``
     - The *c* factor. Please see *Constrained Optimization and Lagrange* *Multiplier Methods*, Chapter 3.
   * - ``c_growth_rate``
     - ``float``, traced
     - ``1.2``
     - The growth rate of the *c* factor.
   * - ``fstop_outer``
     - ``float``, traced
     - ``1e-6``
     - :math:`f_{obj}(\Phi_{sv})` stopping criterion of the outer augmented Lagrangian loop. Terminates the convergence rate falls below this number.
   * - ``xstop_outer``
     - ``float``, traced
     - ``1e-6``
     - :math:`\Phi_{sv}` stopping criterion of the outer augmented Lagrangian loop. Terminates the convergence rate falls below this number.
   * - ``ctol_outer``
     - ``float``, traced
     - ``1e-6``
     - Constraint tolerance of the outer augmented Lagrangian loop. Terminates when both tolerances are satisfied.
   * - ``fstop_inner, fstop_inner_last``
     - ``float``, traced
     - ``1e-6, 0``
     - :math:`f_{obj}(\Phi_{sv})` stopping criterion of the inner LBFGS iteration. Terminates the convergence rate falls below this number.
   * - ``xstop_inner, xstop_inner_last``
     - ``float``, traced
     - ``1e-6, 1e-10``
     - :math:`\Phi_{sv}` stopping criterion of the inner LBFGS iteration. Terminates the convergence rate falls below this number.
   * - ``gtol_inner, gtol_inner_last``
     - ``float``, traced
     - ``1e-6, 1e-10``
     - Gradient tolerance of the inner LBFGS iteration.
   * - ``maxiter_tot``
     - ``int``, static
     - ``10000``
     - The maximum number of total LBFGS iterations permitted, summed across all outer iterations.
   * - ``maxiter_inner``
     - ``int``, static
     - ``1000``
     - The maximum number of inner iterations permitted.

Thus far, we have successfully run an instance of QUADCOIL. The next section will explain how to interpret the outputs.
