QUADCOIL documentation
======================

Welcome to the documentation for `QUADCOIL <https://github.com/lankef/quadcoil>`_, 
the stellarator coil complexity proxy/global coil optimizer!

.. image:: ./assets/title.png
   :alt: An example coil set for NCSX
   :align: center

QUADCOIL is a global coil optimization code that approximates coils with a smooth sheet current. 
In other words, it's a "winding surface" code. However, unlike other winding surface codes, QUADCOIL:

- Supports constrained optimization.
- Supports non-convex quadratic penalties/constraints, such as curvature 
  :math:`\mathbf{K} \cdot \nabla \mathbf{K}`.
- Includes robust winding surface generators that do not produce self-intersections.
- Is fully differentiable with respect to plasma shape, winding surface shape 
  (if auto-generation is disabled), objective weights, and constraint thresholds.

Installation
------------

### Option 1: GitHub Download

Clone the QUADCOIL source files from its `GitHub repository <https://github.com/lankef/quadcoil>`_, 
and then install by:

.. code-block:: bash

   pip install .

Contact
-------

Please contact `Lanke Fu <mailto:ffu@pppl.gov>`_ at PPPL for questions and bug reports.

Publications
------------

1. `Global stellarator coil optimization with quadratic constraints and objectives <https://doi.org/10.1088/1741-4326/ada810>`_


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   math
   tutorial_inputs
   tutorial_outputs
   objective    
   quadcoil
