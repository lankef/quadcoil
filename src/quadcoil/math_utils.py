import jax.numpy as jnp
import numpy as np # Don't panic, it's for type checking
from jax import jit

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

