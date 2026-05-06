Tutorial V: Simsopt integrations
=======================================

QUADCOIL surface objects follow Simsopt's ``SurfaceRZFourier`` conventions.
They can be constructed from Simsopt surfaces, exported back to Simsopt, or
converted to DESC surfaces as described in :doc:`tutorial_to_desc`.

Surface conversion
------------------

To import a Simsopt ``SurfaceRZFourier``:

.. code-block:: python

    from quadcoil import SurfaceRZFourierJAX

    quadcoil_surface = SurfaceRZFourierJAX.from_simsopt(simsopt_surface)

To export a QUADCOIL surface:

.. code-block:: python

    simsopt_surface = quadcoil_surface.to_simsopt()

This is useful after a QUADCOIL run because ``qp.plasma_surface`` and
``qp.winding_surface`` are both ``SurfaceRZFourierJAX`` objects:

.. code-block:: python

    plasma_surface = qp.plasma_surface.to_simsopt()
    winding_surface = qp.winding_surface.to_simsopt()

Exporting the current potential
-------------------------------

``quadcoil.io.simsopt.quadcoil_to_simsopt_cp`` exports the continuous
QUADCOIL sheet-current solution to Simsopt's REGCOIL object, ``CurrentPotentialFourier``:

.. code-block:: python

    from quadcoil.io.simsopt import quadcoil_to_simsopt_cp

    cp = quadcoil_to_simsopt_cp(qp, dofs_opt)

``CurrentPotentialFourier`` is part of Simsopt's regcoil branch. It represents
the current potential on a winding surface, including the net poloidal and
toroidal current terms. In QUADCOIL, ``dofs_opt`` is the optimized dictionary
returned by ``quadcoil.quadcoil``.

Cutting a current potential into Simsopt coils
----------------------------------------------

``quadcoil.io.coil_cutting.simsopt_coil_from_qp`` cuts a QUADCOIL surface
current solution into a Simsopt filamentary coil set:

.. code-block:: python

    from quadcoil.io.coil_cutting import simsopt_coil_from_qp

    coils = simsopt_coil_from_qp(
        qp,
        dofs_opt,
        coils_per_half_period=5,
        order=10,
        ppp=40,
    )

The helper contours the full current potential, fits the contours with
``simsopt.geo.CurveXYZFourier`` curves, assigns ``simsopt.field.Current``
objects, and applies Simsopt's field-period and stellarator symmetries.
In Simsopt, a filamentary coil is a pairing of a ``Curve`` with a ``Current``;
the resulting coil list can be passed to ``simsopt.field.BiotSavart`` for
magnetic field evaluation and downstream coil optimization.

Set ``base_mode=True`` to return only the base-period curves and currents
before symmetry expansion:

.. code-block:: python

    curves, currents = simsopt_coil_from_qp(
        qp,
        dofs_opt,
        coils_per_half_period=5,
        base_mode=True,
    )
