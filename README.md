# QUADCOIL

Welcome to the github page of QUADCOIL, the stellarator coil complexity proxy/global coil optimizer!

Developer contact: Lanke Fu, LF2869@nyu.edu

QUADCOIL is a global coil optimization code that approximates coils with a smooth sheet current. 
In other words, it's a "winding surface" code. However, unlike other winding surface codes, QUADCOIL:

- Supports constrained optimization.
- Supports non-convex quadratic penalties/constraints, such as curvature 
  $\mathbf{K} \cdot \nabla \mathbf{K}$.
- Includes robust winding surface generators that do not produce self-intersections.
- Calculates derivatives with respect to plasma shape, winding surface shape, objective weights, and constraint thresholds.

Read QUADCOIL documentations [here](https://quadcoil.readthedocs.io/en/latest/index.html). 

## Release / PyPI

Before publishing, update `version` in `pyproject.toml` and record the release hash with:

```bash
git rev-parse main
```

Build and validate:

```bash
python -m build
twine check dist/*
```

Upload (recommend TestPyPI first, then PyPI):

```bash
twine upload dist/*
```
