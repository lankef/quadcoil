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
    from simsopt.mhd import Vmec

    # Loading an equilibrium's boundary using simsopt
    equil_qs = Vmec('wout_LandremanPaul2021_QA_lowres.nc', keep_all_files=True)
    plasma_surface = equil_qs.boundary
    net_poloidal_current_amperes = equil_qs.external_current()

    nescoil_out_dict, nescoil_qp, nescoil_dofs, _ = quadcoil(
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
        objective_weight=1.,
        objective_unit=None,
        # Set the output metrics to f_B and f_K
        metric_name=('f_B', 'f_K'),
    )

    # Plotting the solution
    from quadcoil.quantity import Phi_with_net_current
    import matplotlib.pyplot as plt
    
    plt.contour(
        nescoil_qp.quadpoints_phi, 
        nescoil_qp.quadpoints_theta,
        Phi_with_net_current(nescoil_qp, nescoil_dofs), 
        levels=40
    )

Here, we solved the NESCOIL problem (minimizing field error with no additional constraints) on the Landreman-Paul QS configuration. This tutorial will explain how to set up a more complex coil optimizer/proxy with QUADCOIL using ``quadcoil.quadcoil()``, by going over all input parameters and their physical meaning. These parameters fall in 8 categories:

1. Plasma boundary
2. Sheet current properties (net current, resolution, ...)
3. Coil-plasma distance or winding surface
4. Objective functions for coil optimization. Encodes engineering requirements.
5. Constraints for coil optimization. Encodes engineering requirements.
6. Important numerical settings.
7. Metrics for evaluating the coil set satisfying these requirements.
8. (Optional) Solver options.

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

QUADCOIL can automatically generate winding surfaces when used as an equilibrium-stage coil complexity proxy. To auto-generate the winding surface, provide ``plasma_coil_distance`` (and leave ``winding_dofs`` as ``None``). 

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ❗ ``plasma_coil_distance``
     - ``float``, traced
     - ``None``, but **must be specified** to auto-generate the winding surface.
     - The coil-plasma distance :math:`d_{cs}`.
   * - ``surface_type``
     - ``str``, static
     - ``'SurfaceRZFourier'``
     - Surface parametrization. One of ``'SurfaceRZFourier'``, ``'SurfaceXYZTensorFourier'``, ``'SurfaceXYZFourier'``. 
   * - ``winding_mpol``
     - ``int``, static
     - 6
     - The number of poloidal harmonics in the winding surface.
   * - ``winding_ntor``
     - ``int``, static
     - 5
     - The number of toroidal harmonics in the winding surface.
   * - ``winding_surface_mode``
     - ``str``, static
     - ``'self-intersection'``
     - Winding-surface generation mode. ``'self-intersection'`` removes toroidal self-intersections; ``'hull'`` uses a convex-hull rule; ``'uniform'`` keeps a pure normal offset with no post-processing.
   * - ``winding_theta_mode``
     - ``str``, static
     - ``'arclen'``
     - Poloidal reparameterization for the fit. ``'arclen'`` uses arc length on poloidal cross-sections (robust for concave surfaces); ``'arctan'`` uses angle about the cross-section center of mass (smoother, but can misbehave when the surface is concave).
   * - ``winding_phi_interp``
     - ``int``, static
     - ``2``
     - Toroidal oversampling factor during the Fourier fit of the offset surface.
   * - ``winding_theta_interp``
     - ``int``, static
     - ``2``
     - Poloidal oversampling factor during the Fourier fit of the offset surface.
   * - ``winding_theta_rule_subsample``
     - ``int``, static
     - ``2``
     - Poloidal subsampling stride for the self-intersection check (keeps that step from becoming :math:`O(n_\theta^2)` expensive).
   * - ``winding_lam_tikhonov``
     - ``float``, traced
     - ``1e-5``
     - Tikhonov regularization weight for the least-squares surface fit.
   * - ⭐ ``winding_stellsym``
     - ``bool``, static
     - ``True``
     - Whether the winding surface has stellarator symmetry. Equivalent to ``SurfaceRZFourier.stellsym``.

Known winding surface
~~~~~~~~~~~~~~~~~~~~~

QUADCOIL can also run on a known winding surface for tasks such as blanket optimization. To specify a winding surface, set ``winding_dofs`` and omit ``plasma_coil_distance``:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ❗ ``winding_dofs``
     - ``ndarray``, traced
     - ``None``, but **must be specified** when not auto-generating the winding surface.
     - The winding surface degrees of freedom (``simsopt.geo.Surface.get_dofs()`` convention).
   * - ❗ ``winding_mpol``
     - ``int``, static
     - ``6``, but **must match** ``winding_dofs``.
     - The winding surface poloidal harmonic numbers.
   * - ❗ ``winding_ntor``
     - ``int``, static
     - ``5``, but **must match** ``winding_dofs``.
     - The winding surface toroidal harmonic numbers.
   * - ``winding_quadpoints_phi``
     - ``ndarray``, traced
     - ``jnp.linspace(0, 1, 32*nfp, endpoint=False)``
     - Toroidal quadrature points on the winding surface for evaluating surface integrals. Must be an 1D array that goes from 0 to 1, without the endpoint. Equivalent to ``simsopt.geo.Surface.quadpoints_phi``.
   * - ``winding_quadpoints_theta``
     - ``ndarray``, traced
     - ``jnp.linspace(0, 1, 34, endpoint=False)``
     - Poloidal quadrature points on the winding surface for evaluating integrals.
   * - ⭐ ``winding_stellsym``
     - ``bool``, static
     - ``True``
     - Whether the winding surface has stellarator symmetry. Equivalent to ``simsopt.geo.Surface.stellsym``.

4. Choosing the objective function(s)
----------------------------------------

QUADCOIL can perform single or multi-objective optimization. Objectives and constraints in QUADCOIL must be selected from :ref:`available_quantities` by entering their names as ``str``\s. The quantity selected as objective(s) must have scalar output. 

**CAUTION!**

As we will see below, every objective and constraint **must be accompanied** by a normalization constant, referred to as ``<something>_unit``, that scales the objective/constraint to :math:`O(1)`. Without this constant, the optimizer will not behave well. QUADCOIL can automatically calculate these constants from :math:`f(\Phi_{sv}=0)`, but this can be inaccurate. We **strongly** advise providing a value. For the objective, the constant can come from an optimum :math:`\Phi_{sv}^*` that uses automatically calculated normalizing constants. For the constraint, the constant can be the constraint threshold. 


Single-objective
~~~~~~~~~~~~~~~~

In this mode, QUADCOIL will minimize one quantity selected from the list. To select single-objective mode, pass a single ``str`` as the ``objective_name``.

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
     - ``'approx'``
     - Smoothing method for non-smooth problems.
   * - ``smoothing_params``
     - ``dict``, traced
     - ``{'lse_epsilon': 1e-3}``
     - Smoothing parameters. Only used when needed.
   * - ⭐ ``value_only``
     - ``bool``, static
     - ``False``
     - When ``True``, skips adjoint gradient calculation and greatly increases speed.
   * - ``solver``
     - ``str``, static
     - ``'auglag-lbfgs'``
     - Optimizer backend. One of ``'auglag-lbfgs'``, ``'ipm'``, ``'slsqp'``. See Section 8 for solver-specific options.
   * - ``maxiter``
     - ``int``, static
     - ``None`` (``10000`` / ``100`` / ``500`` by solver)
     - Maximum solver iterations. Defaults to ``10000`` for ``'auglag-lbfgs'`` (outer loop), ``100`` for ``'ipm'``, and ``500`` for ``'slsqp'``.
   * - ``maxiter_inner``
     - ``int``, static
     - ``None`` (``500``)
     - Maximum inner L-BFGS iterations per outer step. Only used by ``'auglag-lbfgs'``.
   * - ``phi_init_with_nescoil``
     - ``bool``, static
     - ``True``
     - When ``True``, initialize :math:`\Phi_{sv}` from a NESCOIL solve (overrides ``phi_init`` / ``phi_unit``).
   * - ``precond``
     - ``str`` or ``None``, static
     - ``'svd'``
     - Current-potential preconditioner. One of ``'svd'``, ``'ess'``, ``'svd_K'``, or ``None``.
   * - ``precond_dims``
     - ``tuple`` or ``None``, static
     - ``None``
     - Optional dimensions used by some preconditioners.
   * - ``precond_options``
     - ``dict``, traced
     - ``{'svd_safe_thres': 0., 'ess_alpha': 1., 'ess_p': 2.}``
     - Preconditioner hyperparameters.
   * - ``convex``
     - ``bool``, static
     - ``False``
     - When ``True``, tells supported solvers (``'ipm'``, ``'slsqp'``) to treat the problem as convex in their linear algebra. The KKT adjoint path always uses dense least-squares and does not use this flag.
   * - ``verbose``
     - ``int``, static
     - ``0``
     - ``0`` is silent. ``1`` prints important info. ``2`` also prints outer-loop progress. Higher values may print more solver detail.

QUADCOIL performs optimization on non-smooth objectives. For the optimizer to converge well, it has to
convert the non-smooth problem to a smooth problem. The currently supported values for ``'smoothing'`` are:

.. list-table::
   :header-rows: 1

   * - Value for ``smoothing``
     - Type
     - Advantages
     - Disadvantages
   * - ``'slack'``
     - Exact conversion using slack variables.
     - More accurate optimum.
     - Inaccurate adjoint differentiation. Slower, higher memory usage, and high constraint count.
   * - ``'approx'``
     - Approximate conversion by replacing maximum with LogSumExp functions.
     - Accurate adjoint gradients. Faster, lower memory usage, and low constraint count.
     - Less accurate optimum.

We advise using ``'approx'`` for most uses. ``'slack'`` may lead to better coil 
solutions but can significantly increase adjoint cost. ``'slack'`` is recommended
only with ``solver='auglag-lbfgs'`` and ``value_only=True``.

7. Setting coil metrics
---------------------------

We are almost there. After an optimum coil set :math:`\Phi^*_{sv}` is found, QUADCOIL will evaluate a list of coil quality metrics :math:`M_l(\Phi^*_{sv})`. Derivatives w.r.t. the following quantities will also be available:

- ``plasma_dofs``
- ``net_poloidal_current_amperes``
- ``net_toroidal_current_amperes``
- ``plasma_coil_distance`` or ``winding_dofs``
- ``objective_weight`` (if enabled)
- ``constraint_value`` (if enabled)

Metrics are selected with ``metric_name``:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ⭐ ``metric_name``
     - ``None``, ``str``, ``list``, or ``tuple`` of ``str``, static
     - ``('f_B', 'f_K')``
     - Metric names from :ref:`available_quantities`. A single ``str`` is treated as a one-element tuple. Lists are converted to tuples before JIT. ``None`` or ``()`` skips metric evaluation and KKT adjoint work.

**Adjoint differentiation:** When ``value_only=False`` (default), QUADCOIL performs
 KKT adjoint differentiation for all listed metrics with respect to plasma DOFs,
winding-surface parameters, constraints, and weights. Array-valued metrics are supported:
``'value'`` keeps the metric shape, and each gradient leaf has shape
``(*metric_shape, *param_shape)``. Large fields (for example ``'Bnormal'``) can be
expensive.

**Skipping adjoint:** When ``value_only=True``, QUADCOIL skips adjoint differentiation
for all metrics. This makes QUADCOIL much faster. Use this if you do not need gradients.

**Jacobian of** :math:`\Phi_{sv}`: To obtain the solution DOFs and their derivatives
w.r.t. plasma DOFs, constraint thresholds, etc., include ``'phi_dofs'`` in
``metric_name`` with ``value_only=False`` (default). Then ``out_dict['phi_dofs']``
contains ``'value'`` (a copy of ``dofs_opt['phi']``) and ``'grad'``.

8. (Optional) Tweaking the solver
---------------------------------

Choose the optimizer, iteration limits and solver options.

Example usage::

    out_dict, qp, dofs_opt, solve_results = quadcoil(
        ...,
        solver='auglag-lbfgs',
        maxiter=10000,
        maxiter_inner=1500,
        solver_options={
            'atol_inner': 1e-6,
            'rtol_inner': 1e-6,
        },
    )

Shared / top-level options:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ``solver``
     - ``str``, static
     - ``'auglag-lbfgs'``
     - Optimizer backend. One of ``'auglag-lbfgs'``, ``'ipm'``, ``'slsqp'``.
   * - ``maxiter``
     - ``int``, static
     - ``None`` (solver-dependent)
     - Maximum iterations (outer loop for AugLag; total for IPM/SLSQP).
   * - ``maxiter_inner``
     - ``int``, static
     - ``None`` (``500``)
     - Maximum inner L-BFGS iterations per outer step (``'auglag-lbfgs'`` only).
   * - ``lbfgs_memory``
     - ``int``, static
     - ``10``
     - L-BFGS history length for solvers that use L-BFGS.
   * - ``merge_constraints``
     - ``bool``, static
     - ``False``
     - When ``True``, combines compatible constraint evaluations before solving.
   * - ``solver_options``
     - ``dict`` or ``None``, traced
     - ``None`` (uses ``SOLVER_OPTIONS_DEFAULT_DICT[solver]``)
     - Solver-specific options (see tables below).

``solver='auglag-lbfgs'`` options:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ``solver_options['c_init']``
     - ``float``
     - ``1.``
     - The initial penalty *c* factor. See *Constrained Optimization and Lagrange Multiplier Methods*, Chapter 3.
   * - ``solver_options['c_growth_rate']``
     - ``float``
     - ``2.``
     - Multiplicative growth of *c* each outer step.
   * - ``solver_options['xstop_outer']``
     - ``float``
     - ``1e-6``
     - Outer-loop :math:`x` convergence-rate tolerance.
   * - ``solver_options['ctol_outer']``
     - ``float``
     - ``1e-6``
     - Outer-loop constraint-violation tolerance.
   * - ``solver_options['atol_inner']``, ``solver_options['atol_inner_last']``
     - ``float``
     - ``1e-6``, ``1e-10``
     - Absolute gradient tolerance for ordinary / final inner L-BFGS solves.
   * - ``solver_options['rtol_inner']``, ``solver_options['rtol_inner_last']``
     - ``float``
     - ``1e-6``, ``1e-10``
     - Relative gradient tolerance for ordinary / final inner L-BFGS solves.
   * - ``solver_options['svtol']``
     - ``float``
     - ``1e-6``
     - Singular-value cut-off used by some preconditioning paths.

``solver='ipm'`` options:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ``solver_options['tol_kkt']``
     - ``float``
     - ``1e-6``
     - KKT residual tolerance.
   * - ``solver_options['tau']``
     - ``float``
     - ``0.995``
     - Fraction-to-boundary step-size parameter.
   * - ``solver_options['delta_init']``, ``['delta_min']``, ``['delta_max']``
     - ``float``
     - ``1e-6``, ``1e-10``, ``1e-2``
     - Initial / min / max barrier or regularization parameters.

``solver='slsqp'`` options:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ``solver_options['atol']``
     - ``float``
     - ``1e-7``
     - Absolute convergence tolerance.
   * - ``solver_options['rtol']``
     - ``float``
     - ``1e-7``
     - Relative convergence tolerance.

9. Chunking the adjoint derivative
----------------------------------

When ``metric_name`` contains **array-valued** quantities (see Section 7), we
**strongly recommend** setting ``jac_chunk_size`` to avoid memory overflow.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Definition
   * - ⭐ ``jac_chunk_size``
     - ``int`` or ``None``, static
     - ``None``
     - Number of KKT adjoint metric rows to differentiate at once. ``None`` keeps the fully vectorized path.

**Why this matters.** Adjoint differentiation creates one adjoint row per
scalar metric component. All rows are pushed through a single VJP of the
KKT residual with ``vmap``, so each lane holds a private copy of every
taped intermediate and peak memory grows linearly with the total number of
scalar components. Scalar metrics (for example ``'f_B'``, ``'f_K'``) are
one or two lanes and cost essentially nothing. A length-``ndofs`` vector
metric (for example ``'phi_dofs'``) is hundreds of lanes, and the taped
winding-surface operators of shape
``(n_winding_phi, n_winding_theta, 3, ndofs)`` are replicated once per
lane.

**Worked example.** At ``mpol=ntor=10`` (``ndofs=220``) with the default
winding grid (``32*nfp`` by ``34`` = 2176 points for ``nfp=2``), one
length-220 vector metric replicates ~11 MiB intermediates 220 times and
can push peak device memory into the tens of GiB. Setting
``jac_chunk_size=16`` cuts that peak by roughly ``n_metrics_flat / 16``
at the cost of that many sequential passes over the VJP.

Example usage::

    out_dict, qp, dofs_opt, solve_results = quadcoil(
        ...,
        metric_name=('phi_dofs',),
        jac_chunk_size=16,
    )

Thus far, we have successfully run an instance of QUADCOIL. The next section will explain how to interpret the outputs.
