Tutorial IV: Miscellaneous output utilities
===========================================

QUADCOIL includes a few small output utilities for passing surfaces and coil
solutions to other stellarator tools. This section covers utilities that do
not fit into the DESC or Simsopt-specific tutorials.

FOCUS/FAMUS plasma boundary files
---------------------------------

``quadcoil.io.focus.save_focus`` writes a Fourier RZ surface to a ``.plasma``
file for FOCUS or FAMUS runs with ``CASE_SURFACE=0``:

.. code-block:: python

    from quadcoil.io.focus import save_focus

    save_focus(qp.plasma_surface, "plasma_boundary.plasma")

The input surface may be any of the following:

- a QUADCOIL ``SurfaceRZFourierJAX``;
- a Simsopt ``SurfaceRZFourier``;
- a DESC ``FourierRZToroidalSurface``.

If the filename does not end in ``.plasma``, the extension is added
automatically. The exported file contains the Fourier coefficients of the
surface; it does not include a normal-field target.
