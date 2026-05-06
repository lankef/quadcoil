Tutorial III: DESC Integrations
=======================================

QUADCOIL uses the Simsopt convention for Fourier RZ surfaces internally.
The surface object, ``SurfaceRZFourierJAX``, can nevertheless be converted
to and from DESC's ``FourierRZToroidalSurface``.

Surface conversion
------------------

To import a DESC surface into QUADCOIL, provide the DESC surface and the
quadrature points to use in QUADCOIL:

.. code-block:: python

    import jax.numpy as jnp
    from quadcoil import SurfaceRZFourierJAX

    quadcoil_surface = SurfaceRZFourierJAX.from_desc(
        desc_eq.surface,
        quadpoints_phi=jnp.linspace(0, 1, 32 * desc_eq.NFP, endpoint=False),
        quadpoints_theta=jnp.linspace(0, 1, 34, endpoint=False),
    )

To export a QUADCOIL surface to DESC:

.. code-block:: python

    desc_surface = quadcoil_surface.to_desc()

The returned object is a DESC ``FourierRZToroidalSurface``. This is a surface
conversion only; it does not create a DESC equilibrium.

Running QUADCOIL from a DESC equilibrium
----------------------------------------

The helper ``quadcoil.io.desc.quadcoil_desc`` wraps ``quadcoil.quadcoil`` and
loads the equilibrium-dependent inputs from a DESC equilibrium:

.. code-block:: python

    from quadcoil.io.desc import quadcoil_desc

    out_dict, qp, dofs_opt, solve_results = quadcoil_desc(
        desc_eq=eq,
        vacuum=False,
        plasma_M_theta=32,
        plasma_N_phi=32,
        plasma_coil_distance=0.5,
        objective_name='f_B',
        metric_name=('f_B', 'f_K'),
    )

The DESC equilibrium ``eq`` supplies the plasma boundary, number of field
periods, stellarator symmetry, net poloidal current :math:`G`, and, when
``vacuum=False``, the plasma contribution to the normal field error
``Bnormal_plasma``. With ``desc_scaling=True``, the wrapper also uses DESC
normalization factors to set objective and constraint units when available.

The remaining keyword arguments are passed to ``quadcoil.quadcoil``. See
:doc:`tutorial_inputs` for the detailed meaning of objectives, constraints,
winding surface choices, metrics, and solver options.
