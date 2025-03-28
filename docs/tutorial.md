# Running QUADCOIL

Running QUADCOIL is easy! The main function `quadcoil.quadcoil()` includes all necessary steps needed to generate a sheet current coil set for a plasma boundary. These includes:
1. Generating the winding surface (if not provided).
2. Setting up and solving the 

```python
from quadcoil import quadcoil
phi_mn, out_dict, quadcoil_params, status = quadcoil(
    # - Necessary plasma parameters
    nfp,
    stellsym,
    mpol,
    ntor,
    plasma_mpol,
    plasma_ntor,
    plasma_dofs,
    net_poloidal_current_amperes,
    net_toroidal_current_amperes,
    # - Parameters with default values
    # - Quadcoil parameters
    ...=...
    # - Plasma parameters
    ...=...
    # - Winding parameters
    ...=...
    # - Objectives
    ...=...
    # - Constraints
    ...=...
    # - Metrics to study
    ...=...
    # - Solver options
    ...=...
)
```

This tutorial will offer a detailed guide for choosing the inputs and interpreting the results. 

## Setting up the problem
QUADCOIL is a global optimization code. To run it, we must provide the plasma shape, and then set up the optimization problem that defines the coilset. We start by inputting the plasma information. 

### Plasma info
The following plasma parameters are necessary inputs to QUADCOIL. These parameters do not have default values. 

QUADCOIL only currently supports surfaces given by $(R, Z)$ Fourier representations. 

- `nfp`(int, static) - Thenumber of field periods.
- `stellsym`(bool, static) - Thenumber of field periods.
- `mpol`(int, static) - The number of poloidal Fourier harmonics in $\Phi$.
- `ntor`(int, static) - The number of toroidal Fourier harmonics in $\Phi$.
- `plasma_mpol`(int, static) - The number of poloidal Fourier harmonics in the plasma surface.
- `plasma_ntor`(int, static) - The number of toroidal Fourier harmonics in the plasma surface.
- `plasma_dofs`(float[], traced)
- `net_poloidal_current_amperes: `
- `net_toroidal_current_amperes: `
    
    # - Quadcoil parameters
    quadpoints_phi=None, 
    quadpoints_theta=None,
    cp_mn_unit=None,
    
    # - Plasma parameters
    plasma_quadpoints_phi=None, 
    plasma_quadpoints_theta=None,
    Bnormal_plasma=None,

    # - Winding parameters (offset)
    plasma_coil_distance=None,
    winding_surface_generator=gen_winding_surface_atan,
    winding_surface_generator_args={'pol_interp': 1, 'lam_tikhonov': 0.05},

    # - Winding parameters (known surface)
    winding_dofs=None,
    winding_mpol=5, winding_ntor=5,
    winding_quadpoints_phi=None,
    winding_quadpoints_theta=None,

    # - Objectives
    objective_name='f_B_normalized_by_Bnormal_IG',
    objective_weight=None,
    objective_unit=None,
    # - Constraints
    constraint_name=(),
    constraint_type=(),
    constraint_unit=(),
    constraint_value=(),
    # - Metrics to study
    metric_name=('f_B', 'f_K'),

    # - Solver options
    c_init=1.,
    c_growth_rate=1.1,
    ftol_outer=1e-5, # constraint tolerance
    ctol_outer=1e-5, # constraint tolerance
    xtol_outer=1e-5, # convergence rate tolerance
    gtol_outer=1e-5, # gradient tolerance
    ftol_inner=1e-5,
    xtol_inner=1e-5,
    gtol_inner=1e-5,
    maxiter_inner=1500,
    maxiter_outer=50,
