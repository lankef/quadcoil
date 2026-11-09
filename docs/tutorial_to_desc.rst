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
