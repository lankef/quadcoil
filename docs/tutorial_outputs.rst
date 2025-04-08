Tutorial II: interpreting outputs
===================================

This tutorial will explain how to interpret the output from QUADCOIL. 
We illustrate this with the example in ``example/topology.ipynb``:

.. code-block:: python

    # First, test with the NESCOIL problem, auto-generating WS.
    print('Running quadcoil, with auto-generated '\
          'winding surface and K_theta constraint.')
    out_dict, qp, phi_mn, status = quadcoil(
        nfp=cp.nfp,
        stellsym=cp.stellsym,
        mpol=cp.mpol,
        ntor=cp.ntor,
        plasma_dofs=plasma_surface.get_dofs(),
        plasma_mpol=plasma_surface.mpol,
        plasma_ntor=plasma_surface.ntor,
        net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
        plasma_coil_distance=plasma_surface.minor_radius(),
        # Setting the objective to f_B. Use auto-scaling.
        objective_name='f_B',
        # Set the constraint to K_theta. Use auto-scaling.
        constraint_name=('K_theta',),
        constraint_type=('>=',),
        constraint_value=(0.,),
        constraint_unit=(None,),
        # Set the output metrics to f_B and f_K
        metric_name=('f_B', 'f_K'),
        maxiter_inner=1500,
        maxiter_outer=10,
        ftol_inner=0,
        xtol_inner=0,
    )

This produces a low-field-error coil set consisting of purely poloidal coils by
preventing the current from changing sign in the poloidal (:math:`\theta`) direction:

.. math::

   \Phi^*_{sv} = &\text{argmin}_{\Phi_{sv}} f_B(\Phi_{sv}),\\
   &\text{subject to } K_\theta\geq0.

The coil quality matric we chose are the integrated field error :math:`f_B`
and the integrated current density :math:`f_K`. 

Now, we will go over the 4 outputs from ``quadcoil.quadcoil`` one by one.

``out_dict`` - Coil metrics and gradients
------------------------------------------------------------------
``out_dict`` is the most important output. It is a nested dict containing the 
value and gradients of the coil quality metrics selected by ``metric_name``. 
Its structure is:

.. code-block:: python

    'f_B': {
        'grad': {
            'df_dconstraint_value': array(dtype=float64, shape=(1,)),
            'df_dnet_poloidal_current_amperes': 1.0665563051980406e-08,
            'df_dnet_toroidal_current_amperes': 3.7240900815694553e-09,
            'df_dplasma_coil_distance': 0.9833843550159864,
            'df_dplasma_dofs': array(dtype=float64, shape=(187,)),
        },
        'value': 0.06338841775107304,
    },
    'f_K': {
        'grad': {
            'df_dconstraint_value': array(dtype=float64, shape=(1,)),
            'df_dnet_poloidal_current_amperes': 12628782.472101763,
            'df_dnet_toroidal_current_amperes': 660739.232726985,
            'df_dplasma_coil_distance': 144504124060114.53,
            'df_dplasma_dofs': array(dtype=float64, shape=(187,)),
        },
        'value': 75040039727553.6,
    },
    
Here, we can see that the first level of the dictionary is organized by the 
name of the metric, ``'f_B'`` and ``'f_K'``. The second level is organized into 
two categories, ``'value'`` and ``'grad'``. The ``'grad'`` entry contains a third level. 
It is organized by the name of the independent variable of each gradient.
The keys in the third level will change based on whether 
multi-objective optimization is enabled, constraints are present, 
and whether the winding surface is provided.

Setting ``value_only=True`` when running ``quadcoil.quadcoil`` will skip gradient calculations.
In this case, ``out_dict`` will not have the ``'grad'`` layer.

``qp`` - Problem configurations
---------------------------------------------------------------------
``qp : QuadcoilParams`` is an objects that contains information on the plasma boundary, 
winding surface, net currents and resolutions. Together, ``qp`` and ``phi_mn`` contains 
all informations required to evaluate any physical quantities available in ``quadcoil.objective``.
For how to do this, see :ref:`available_quantities`.
It **does not** contain the objective and constraint choices. 

``qp`` can be used to reconstruct the configuration in Simsopt. 
``qp.winding_surface.to_simsopt()`` and ``qp.winding_surface.to_simsopt()`` 
exports both surfaces as ``simsopt.geo.SurfaceRZFourier``.

``phi_mn`` - :math:`\Phi_{sv}` in Fourier representation 
---------------------------------------------------------------------
``phi_mn`` is an ``ndarray`` storing the Fourier coefficients of :math:`\Phi_{sv}`.
It uses the same convention as ``simsopt.field.CurrentPotentialFourier`` in the ``regcoil``
branch of simsopt. 

Together, ``qp`` and ``phi_mn`` contains all informations required to evaluate any 
physical quantities available in ``quadcoil.objective``. 

``status`` - Optimizer end state
---------------------------------------------------------------------

