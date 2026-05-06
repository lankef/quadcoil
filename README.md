# QUADCOIL

Welcome to the github page of QUADCOIL, the stellarator coil complexity proxy/global coil optimizer!

Developer contact: Lanke Fu, LF2869@nyu.edu

QUADCOIL is a global coil optimization code that approximates coils with a smooth sheet current. 
In other words, it's a "winding surface" code. However, unlike other winding surface codes, QUADCOIL:

- Supports constrained optimization.
- Supports non-convex quadratic penalties/constraints, such as curvatures and Lorentz force.
- Includes robust winding surface generators that do not produce self-intersections.
- Calculates derivatives with respect to plasma shape, winding surface shape, objective weights, and constraint thresholds.

Read QUADCOIL documentations [here](https://quadcoil.readthedocs.io/en/latest/index.html). 

## Changelog

### In progress

This is the first changelog of QUADCOIL. 

Based on feedbacks from ISHW QUADCOIL is currently under a major revision to add:
- Multiple surface types (SurfaceXYZTensor)
- PDIPM solver and adjoint differentiation based on KKT conditions
- Improved winding surface generator to detect poloidal self-intersections. Will be 
based on the Jacobian of the uniform-offset surfaces set.