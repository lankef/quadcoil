# Running QUADCOIL

Running QUADCOIL is simple! QUADCOIL requires the 

## Minimum example (NESCOIL)

To use QUADCOIL, import and run the main function:

'''python
from quadcoil import quadcoil
nescoil_phi_mn, nescoil_out_dict, nescoil_qp, _ = quadcoil(
    # Setting the plasma surface.
    nfp=cp.nfp,
    stellsym=cp.stellsym,
    mpol=cp.mpol,
    ntor=cp.ntor,
    plasma_dofs=plasma_surface.get_dofs(),
    plasma_mpol=plasma_surface.mpol,
    plasma_ntor=plasma_surface.ntor,
    # Setting net currents and winding surface distance
    net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
    net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
    plasma_coil_distance=plasma_surface.minor_radius(),
    # Set the objective to normalized f_B, only
    objective_name=('f_B_normalized_by_Bnormal_IG',),
    objective_weight=(1,),
    objective_unit=(1,),
    # Set the output metrics to f_B and f_K
    metric_name=('f_B', 'f_K')
)
'''

This runs the NESCOIL problem, and minimizes: 
$$
f_B = \oint_\text{plasma surface} dA |\mathbf{B}\cdot\hat\mathbf{n}|^2
$$
without any constraints. 

## Unconstrained, multi-objective optimization (REGCOIL)

# First, test with the REGCOIL problem, auto-generating WS.

```
from quadcoil import quadcoil
regcoil1_phi_mn, regcoil1_out_dict, regcoil1_qp, _ = quadcoil(
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
    # Set the objective to 
    # f_B + 0.01 f_K
    objective_name=('f_B', 'f_K'),
    objective_weight=(1., 0.01),
    # Usually we recommend normalizing f_B and f_K to 
    # ~1, but in this case, for testing prupose, not normalizing 
    # is also okay
    objective_unit=(1., 1.), 
    # Set the output metrics to f_B and f_K
    metric_name=('f_B', 'f_K')
)
```