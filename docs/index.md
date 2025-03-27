# QUADCOIL

Welcome to the documentation for [QUADCOIL](https://github.com/lankef/quadcoil), the stellarator coil complexity proxy/global coil optimizer!

![An example coil set for NCSX](./assets/title.png "An example coil set for NCSX")

QUADCOIL is a global coil optimization code that approximates coils with a smooth sheet current. In other 
words, it's a "winding surface" code. However, unlike other winding surface codes, QUADCOIL:
- Supports constrained optimization.
- Supports non-convex quadratic penalties/constraints,
such as curvature $\textbf{K}\cdot\nabla\textbf{K}$.
- Includes robust winding surface generators without self intersections.
- Is fully differentiable w.r.t. plasma shape, winding surface shape
(if auto-generation is disabled), objective weights and constraint thresholds.

## Installation
### Option 1: Github download
Clone the QUADCOIL source files from its [Github repositoiry](https://github.com/lankef/quadcoil), and then install by:
```
pip install .
```

## Contact
Please contact [Lanke Fu](mailto:ffu@pppl.gov) at PPPL for questions and bug reports.

## Publications
1. [Global stellarator coil optimization with quadratic constraints and objectives](doi.org/10.1088/1741-4326/ada810)