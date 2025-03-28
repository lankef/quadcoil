
# Available quantites
Below is a list of all quantities QUADCOIL supports. Scalar quantities can be used as both objectives and constraints. Fields (with array output) can only be used as constraints.

Notation:
- $\mathbf{B}$: The magnetifc field on the plasma boundary.
- $\mathbf{K}$: The sheet current representing the coil set.
- $\hat{\mathbf{n}}$: The unit normal of the plasma boundary.
- $n_{FP}$: The number of field periods.
- $n_\phi^P$: The plasma toroidal resolution, `len(plasma_quadpoints_phi)`
- $n_\theta^P$: The plasma poloidal resolution, `len(plasma_quadpoints_theta)`
- $n_\phi^E$: The toroidal evaluation resolution on the winding surface, `len(quadpoints_phi)`
- $n_\theta^E$: The poloidal evaluation resolution on the winding surface, `len(quadpoints_theta)`

## Magnetic field
These objectives are related to the magneitc field on the plasma surface:
| Name |  Formula | Output shape | Description | 
| --- | --- | --- | --- |
|`'winding_surface_B'`| $\mathbf{B}$ |$(n_\phi^P, n_\theta^P, 3)$| The $(x, y, z)$ components of the magnetic field on the plasma surface. |
|`'Bnormal'`| $\mathbf{B}\cdot\hat{\mathbf{n}}$ |$(n_\phi^P, n_\theta^P)$| The normal magnetic field on the plasma surface. |
|`'f_B'`| $f_B\equiv\frac{n_{FP}}{2}\oint_\text{plasma}da\|\mathbf{B}\cdot\hat{\mathbf{n}}\|^2$ | Scalar| The integrated normal field error. Also the NESCOIL objective. |
|`'f_B_normalized_by_Bnormal_IG'`| $\frac{f_B}{f_B\|_\text{net currents only}}$ | Scalar | $f_B$, normalized by its value with only the net toroidal and poloidal currents. |
|`'f_max_Bnormal_abs'`| $\max_\text{plasma surface}\|\mathbf{B}\cdot\hat{\mathbf{n}}\|$ | Scalar | The maximum normal magnetic field strength. |
|`'f_max_Bnormal2'`| $\max_\text{plasma surface}\|\mathbf{B}\cdot\hat{\mathbf{n}}\|^2$ | Scalar | The maximum normal magnetic field strength squared. A convex quadratic constraint may behave better than a linear constraint.|


## Current magnitude and sign
These objectives are related to the magnitude and sign of the sheet current $\mathbf{K}$ that represents a coil set:
| Name |  Formula | Output shape | Description | 
| --- | --- | --- | --- |
|`'K'`| $\mathbf{K}$ |$(n_\phi^E, n_\theta^E, 3)$| The $(x, y, z)$ components of the current on the winding surface. |
|`'K2'`| $\|\mathbf{K}\|^2$ |$(n_\phi^E, n_\theta^E)$| The current strength on the winding surface. |
|`'K_theta'`| $K_\theta$ |$(n_\phi^E, n_\theta^E)$| The poloidal current distribution on the winding surface. |
|`'f_K'`| $f_K\equiv\frac{n_{FP}}{2}\oint_\text{WS}da\|K\|^2$ |Scalar| The integrated magnetic field strength  on the winding surface. Also the REGCOIL regularization factor. |

## Current curvature
These objectives are related to the curvature of the sheet current:
| Name |  Formula | Output shape | Description | 
| --- | --- | --- | --- |
|`'K_dot_grad_K'`| $\mathbf{K}\cdot\nabla\mathbf{K}$ |$(n_\phi^E, n_\theta^E, 3)$| The $(x, y, z)$ components of $\mathbf{K}\cdot\nabla\mathbf{K}$ on the winding surface. |
|`'K_dot_grad_K_cyl'`| $(\mathbf{K}\cdot\nabla\mathbf{K})_{(R, \Phi, Z)}$ |$(n_\phi^E, n_\theta^E, 3)$| The $(R, \Phi, Z)$ components of $\mathbf{K}\cdot\nabla\mathbf{K}$ on the winding surface. |
|`'f_max_K_dot_grad_K_cyl'`| $\max_\text{WS}\|(\mathbf{K}\cdot\nabla\mathbf{K})_{(R, \Phi, Z)}\|_\infty$ |Scalar| Maximum $(R, \Phi, Z)$ component of $\mathbf{K}\cdot\nabla\mathbf{K}$ over the winding surface. |

## Dipole
These objectives are related to dipole optimization.
| Name |  Formula | Output shape | Description | 
| --- | --- | --- | --- |
|`'Phi'`| $\Phi_{sv}$ |$(n_\phi^E, n_\theta^E)$| The dipole density distribution on the winding surface. Also referred to as the single valued component of the current potential. |
|`'Phi_abs'`| $\|\Phi_{sv}\|$ |$(n_\phi^E, n_\theta^E)$| The absolyute value of the dipole density distribution on the winding surface.|
|`'Phi2'`| $\|\Phi_{sv}\|^2$ |$(n_\phi^E, n_\theta^E)$| The squared dipole density distribution on the winding surface.|
|`'Phi_with_net_current'`| $\Phi = \Phi_{sv} + \frac{G\phi'}{2\pi} + \frac{I\theta'}{2\pi}$|$(n_\phi^E, n_\theta^E)$|  The full current potential on the winding surface, with the secular components representing the net poloidal and toroidal currents $G$ and $I$. |
|`'f_max_Phi'`| $max_\text{WS}\|\Phi_{sv}\|$|Scalar| The maximum dipole density on the winding surface. |
|`'f_max_Phi2'`| $max_\text{WS}\|\Phi_{sv}\|^2$|Scalar| The maximum dipole density squared on the winding surface. A convex quadratic constraint may behave better than a linear constraint.|

# Lorentz force
Lorentz force is not yet fully implemented.