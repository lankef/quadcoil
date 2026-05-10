import jax.numpy as jnp
import numpy as np # Don't panic, it's for type checking
from jax import jit, custom_vjp
from jax.tree_util import tree_reduce
import jax.nn as jnn
import lineax as lx

# Physical constants (from scipy.constants)
# Vacuum magnetic permeability [H/m] = [N/A^2]
# Value: 4π × 10^-7 H/m exactly (by definition in SI units)
mu_0 = 1.25663706127e-06

def tree_len(pytree):
    return tree_reduce(
        lambda acc, leaf: acc + jnp.atleast_1d(leaf).size, pytree, initializer=0
    )

def is_ndarray(arr, n=1):
    return isinstance(arr, (np.ndarray, jnp.ndarray)) and arr.ndim == 1

def sin_or_cos(x, mode):
    r'''
    Scans a pair of arrays, ``x`` and ``mode``. Where ``mode==1``, return ``jnp.sin(x)``. 
    Otherwise return ``jnp.cos(x)``. Used in inverse Fourier Transforms.

    Parameters
    ----------  
    x : ndarray
        The data.
    mode : ndarray
        The choice of trigonometry functions.

    Returns
    -------
    ndarray
    '''
    return jnp.where(mode==1, jnp.sin(x), jnp.cos(x))

@jit
def norm_helper(vec):
    r'''
    Calculates :math:`|v|` and :math:`1/|v|` for a vector field
    on a 2d surface. 

    Parameters
    ----------  
    vec : ndarray, shape (Nx, Ny, ..., 3)
        The vector field

    Returns
    -------
    normN_prime_2d : ndarray, shape (Nx, Ny, ...)
        The vector field's length, :math:`|v|`
    inv_normN_prime_2d: ndarray, shape (Nx, Ny, ...)
        1/the vector field's length, :math:`1/|v|`
    '''
    # Length of the non-unit WS normal vector |N|,
    # its inverse (1/|N|) and its inverse's derivatives
    # w.r.t. phi(phi) and theta
    # Not to be confused with the normN (plasma surface Jacobian)
    # in Regcoil.
    norm = jnp.linalg.norm(vec, axis=-1) # |N|
    inv_norm = 1/norm # 1/|N|
    return norm, inv_norm

@jit
def project_arr_coord(
    operator, 
    unit1, unit2, unit3):
    r'''
    Project an array of vector fields on a 2d surface
    in a given basis, ``unit1, unit2, unit3``.

    Parameters
    ----------  
    operator : ndarray, shape (n_phi, n_theta, 3, ...)
        An array of (n_phi, n_theta, 3) vector fields. 
        ``operator.shape[:3]`` must be ``(n_phi, n_theta, 3)``.
        Otehrwise the shape is flexible.
    unit1 : ndarray, shape (n_phi, n_theta, 3)
        Basis vector 1 where the vector field is sampled.
    unit2 : ndarray, shape (n_phi, n_theta, 3)
        Basis vector 2 where the vector field is sampled.
    unit3 : ndarray, shape (n_phi, n_theta, 3)
        Basis vector 3 where the vector field is sampled.
    
    Returns
    -------
    Outputs: ndarray, shape (n_phi, n_theta, 3, ...)
    '''
    # Memorizing shape of the last dimensions of the array
    len_phi = operator.shape[0]
    len_theta = operator.shape[1]
    operator_shape_rest = list(operator.shape[3:])
    operator_reshaped = operator.reshape((len_phi, len_theta, 3, -1))
    # Calculating components
    # shape of operator is 
    # (n_grid_phi, n_grid_theta, 3, n_dof, n_dof)
    # We take the dot product between K and unit vectors.
    operator_1 = jnp.sum(unit1[:,:,:,None]*operator_reshaped, axis=2)
    operator_2 = jnp.sum(unit2[:,:,:,None]*operator_reshaped, axis=2)
    operator_3 = jnp.sum(unit3[:,:,:,None]*operator_reshaped, axis=2)

    operator_1_nfp_recovered = operator_1.reshape([len_phi, len_theta] + operator_shape_rest)
    operator_2_nfp_recovered = operator_2.reshape([len_phi, len_theta] + operator_shape_rest)
    operator_3_nfp_recovered = operator_3.reshape([len_phi, len_theta] + operator_shape_rest)
    operator_comp_arr = jnp.stack([
        operator_1_nfp_recovered,
        operator_2_nfp_recovered,
        operator_3_nfp_recovered
    ], axis=2)
    return(operator_comp_arr)

@jit
def project_arr_cylindrical(
        gamma, 
        operator,
    ):
    r'''
    Project a stack of vector fields onto a cylindrical 
    coordinate for a given set of coordinate points.

    Parameters
    ----------  
    gamma : ndarray, shape (n_phi, n_theta, 3)
        The location of the coordinate points 
        where the field is sampled in x, y, z.
    operator : ndarray, shape (n_phi, n_theta, 3, ...)
        A stack of (n_phi, n_theta, 3) vector fields.
        ``operator.shape[:3]`` must be ``(n_phi, n_theta, 3)``.
        Otherwise the shape is flexible.
    
    Returns
    -------
    Outputs: ndarray, shape (n_phi, n_theta, 3, ...)
    '''
    # Keeping only the x, y components
    r_unit = jnp.zeros_like(gamma)
    r_unit = r_unit.at[:, :, -1].set(0)
    # Calculating the norm and dividing the x, y components by it
    r_unit = r_unit.at[:, :, :-1].set(gamma[:, :, :-1] / jnp.linalg.norm(gamma, axis=2)[:, :, None])

    # Setting Z unit to 1
    z_unit = jnp.zeros_like(gamma)
    z_unit = z_unit.at[:,:,-1].set(1)

    phi_unit = jnp.cross(z_unit, r_unit)
    return(
        project_arr_coord(
            operator,
            unit1=r_unit, 
            unit2=phi_unit, 
            unit3=z_unit,
        )
    )


def max_lse(x, epsilon, **kwargs):
    approx = epsilon * jnn.logsumexp(a=x/epsilon, **kwargs)
    return approx

def abs_lse(x, epsilon, **kwargs):
    x_stacked = jnp.stack((x, -x), x.ndim)
    return max_lse(
        x_stacked, epsilon, axis=-1, **kwargs
    )

def linf_lse(x, epsilon, **kwargs):
    abs = abs_lse(x, epsilon, **kwargs)
    return max_lse(abs, epsilon, **kwargs)

# Custom lineax routine that removes nans. 
# Used to make autodiff more robust to floating point error.
@custom_vjp
def safe_linear_solve(A, b):
    operator = lx.MatrixLinearOperator(A)
    solver = lx.AutoLinearSolver(well_posed=False)
    solution = lx.linear_solve(operator, b, solver)
    return solution.value

def safe_linear_solve_fwd(A, b):
    x = safe_linear_solve(A, b)
    return x, (A, x)

def safe_linear_solve_bwd(res, g):
    A, x = res
    # Clean the gradient before using it
    g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Solve A^T v = g for the VJP
    operator = lx.MatrixLinearOperator(A.T)
    solver = lx.AutoLinearSolver(well_posed=False)
    v = lx.linear_solve(operator, g, solver).value
    v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    
    # dL/dA = -v @ x^T, dL/db = v
    dA = -jnp.outer(v, x)
    db = v
    return (dA, db)

safe_linear_solve.defvjp(safe_linear_solve_fwd, safe_linear_solve_bwd)


# ======================================================================
# Plane fitting utilities
# ======================================================================

@jit
def project_points_to_plane(gamma_pol):
    """
    Project 3D points onto least-squares fit plane and return 2D coordinates.
    
    This function fits a plane to 3D points using SVD (least-squares), projects
    the points onto that plane, and returns their coordinates in the plane's
    local 2D coordinate system.
    
    The plane is fit to minimize the sum of squared perpendicular distances from
    the points to the plane. The plane passes through the centroid of the points,
    and the normal vector is the singular vector corresponding to the smallest
    singular value.
    
    Parameters
    ----------
    gamma_pol : ndarray, shape (n, 3)
        3D points to project onto the fitted plane.
    
    Returns
    -------
    x_pol : ndarray, shape (n,)
        X coordinates of points in the plane's local coordinate system.
    y_pol : ndarray, shape (n,)
        Y coordinates of points in the plane's local coordinate system.
    plane_data : dict
        Dictionary containing plane parameters:
        - 'origin': ndarray, shape (3,) - Point on plane (centroid)
        - 'normal': ndarray, shape (3,) - Unit normal vector
        - 'u_axis': ndarray, shape (3,) - First basis vector in plane
        - 'v_axis': ndarray, shape (3,) - Second basis vector in plane
        - 'fitting_error': float - RMS distance of points from fitted plane
    
    Notes
    -----
    - The function is JIT-compiled and vmap-compatible
    - All linear algebra operations use JAX for automatic differentiation
    - The orthonormal basis (u_axis, v_axis) in the plane is chosen to be
      right-handed with the normal vector
    - For batched operations on multiple point clouds, use:
      `vmap(project_points_to_plane, in_axes=0, out_axes=(0, 0, 0))`
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> # Create points on a tilted plane with slight noise
    >>> n = 100
    >>> x = jnp.linspace(-1, 1, n)
    >>> y = jnp.linspace(-1, 1, n)
    >>> z = 0.5 * x + 0.3 * y + 0.01 * jnp.random.normal(size=n)
    >>> points = jnp.stack([x, y, z], axis=-1)
    >>> x_pol, y_pol, plane_data = project_points_to_plane(points)
    >>> print(f"RMS fitting error: {plane_data['fitting_error']:.6f}")
    """
    # Step 1: Compute centroid (origin of the plane)
    p0 = jnp.mean(gamma_pol, axis=0)
    centered = gamma_pol - p0
    
    # Step 2: SVD to find the plane normal
    # The normal is the right singular vector with smallest singular value
    U, S, Vt = jnp.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1, :]
    normal = normal / jnp.linalg.norm(normal)
    
    # Step 3: Construct orthonormal basis in the plane
    # Choose an arbitrary vector that is not parallel to the normal
    # If normal is mostly aligned with x-axis, use y-axis, otherwise use x-axis
    arbitrary = jnp.where(
        jnp.abs(normal[0]) < 0.9,
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0])
    )
    
    # First basis vector in plane
    u = jnp.cross(normal, arbitrary)
    u = u / jnp.linalg.norm(u)
    
    # Second basis vector in plane (completes right-handed system)
    v = jnp.cross(normal, u)
    
    # Step 4: Project points onto the plane
    # Distance from each point to the plane along normal direction
    centered_from_origin = gamma_pol - p0
    distances = jnp.dot(centered_from_origin, normal)
    projected = gamma_pol - distances[:, None] * normal[None, :]
    
    # Step 5: Compute 2D coordinates in plane basis
    relative = projected - p0
    x_pol = jnp.dot(relative, u)
    y_pol = jnp.dot(relative, v)
    
    # Compute fitting error (RMS distance from plane)
    fitting_error = jnp.sqrt(jnp.mean(distances**2))
    
    # Package plane parameters
    plane_data = {
        'origin': p0,
        'normal': normal,
        'u_axis': u,
        'v_axis': v,
        'fitting_error': fitting_error
    }
    
    return x_pol, y_pol, plane_data


@jit
def reconstruct_3d_from_plane(x_pol, y_pol, plane_data):
    """
    Reconstruct 3D points from 2D plane coordinates.
    
    This is the inverse operation of project_points_to_plane. Given 2D coordinates
    in a plane's local coordinate system and the plane parameters, reconstruct
    the 3D coordinates.
    
    Parameters
    ----------
    x_pol : ndarray, shape (n,)
        X coordinates in the plane's local coordinate system.
    y_pol : ndarray, shape (n,)
        Y coordinates in the plane's local coordinate system.
    plane_data : dict
        Dictionary containing plane parameters from project_points_to_plane:
        - 'origin': ndarray, shape (3,) - Point on plane (centroid)
        - 'u_axis': ndarray, shape (3,) - First basis vector in plane
        - 'v_axis': ndarray, shape (3,) - Second basis vector in plane
    
    Returns
    -------
    gamma_3d : ndarray, shape (n, 3)
        Reconstructed 3D points on the plane.
    
    Examples
    --------
    >>> x_pol, y_pol, plane_data = project_points_to_plane(points_3d)
    >>> reconstructed = reconstruct_3d_from_plane(x_pol, y_pol, plane_data)
    >>> # reconstructed should be close to the projection of points_3d
    """
    p0 = plane_data['origin']
    u = plane_data['u_axis']
    v = plane_data['v_axis']
    
    # Reconstruct 3D coordinates: p = p0 + x*u + y*v
    gamma_3d = p0[None, :] + x_pol[:, None] * u[None, :] + y_pol[:, None] * v[None, :]
    
    return gamma_3d


@jit
def plane_fitting_error(gamma_pol, plane_data):
    """
    Compute RMS distance of points from a fitted plane.
    
    Parameters
    ----------
    gamma_pol : ndarray, shape (n, 3)
        3D points to measure distance from plane.
    plane_data : dict
        Dictionary containing plane parameters:
        - 'origin': ndarray, shape (3,) - Point on plane
        - 'normal': ndarray, shape (3,) - Unit normal vector
    
    Returns
    -------
    rms_error : float
        Root mean square distance of points from the plane.
    max_error : float
        Maximum absolute distance of any point from the plane.
    
    Examples
    --------
    >>> x_pol, y_pol, plane_data = project_points_to_plane(points_3d)
    >>> rms_err, max_err = plane_fitting_error(points_3d, plane_data)
    """
    p0 = plane_data['origin']
    normal = plane_data['normal']
    
    # Distance from each point to the plane
    centered = gamma_pol - p0
    distances = jnp.dot(centered, normal)
    
    rms_error = jnp.sqrt(jnp.mean(distances**2))
    max_error = jnp.max(jnp.abs(distances))
    
    return rms_error, max_error
