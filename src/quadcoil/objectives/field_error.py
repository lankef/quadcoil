import jax.numpy as jnp
def f_B(quadcoil_params, current_potential_mn):
    # Calculates the integral of |Bnorm|^2
    # over the surface using a QuadcoilParams and 
    # a current potential's (Phi) Fourier coefs.
    # Here, instead of phi, we call it cp to avoid confusion
    # with the angle phi.
    plasma_normal = quadcoil_params.plasma_surface.normal()
    normN = jnp.linalg.norm(plasma_normal.reshape(-1, 3), axis=-1)
    B_normal, b_e = quadcoil_params.biot_savart()
    return(jnp.sum((B_normal @ current_potential_mn + b_e)**2) * quadcoil_params.nfp)

''' Ports of gj and b_e calculation from REGCOIL'''
def winding_surface_field_Bn(points_plasma, points_coil, normal_plasma, normal_coil, stellsym, zeta_coil, theta_coil, ndofs, m, n, nfp):
    # Ensure inputs are NumPy arrays
    points_plasma = jnp.asarray(points_plasma)
    points_coil = jnp.asarray(points_coil)
    normal_plasma = jnp.asarray(normal_plasma)
    normal_coil = jnp.asarray(normal_coil)
    zeta_coil = jnp.asarray(zeta_coil)
    theta_coil = jnp.asarray(theta_coil)
    m = jnp.asarray(m)
    n = jnp.asarray(n)

    # Precompute constants
    fak = 1e-7  # mu0 / (4 * pi)

    # Calculate gij
    diff = points_plasma[:, None, :] - points_coil[None, :, :]
    rmag2 = jnp.sum(diff**2, axis=-1)
    rmag_inv = 1.0 / jnp.sqrt(rmag2)
    rmag_inv_3 = rmag_inv**3
    rmag_inv_5 = rmag_inv**5

    npdotnc = jnp.sum(normal_plasma[:, None, :] * normal_coil[None, :, :], axis=-1)
    rdotnp = jnp.sum(diff * normal_plasma[:, None, :], axis=-1)
    rdotnc = jnp.sum(diff * normal_coil[None, :, :], axis=-1)

    gij = fak * (npdotnc * rmag_inv_3 - 3.0 * rdotnp * rdotnc * rmag_inv_5)

    # Calculate gj
    angle = 2 * jnp.pi * m[:, None] * theta_coil - 2 * jnp.pi * n[:, None] * zeta_coil * nfp
    sphi = jnp.sin(angle)  # Shape: (len(m), num_coil)
    cphi = jnp.cos(angle)  # Shape: (len(m), num_coil)

    # Reshape gij for compatibility: (num_plasma, 1, num_coil)
    gij_expanded = gij[:, None, :]  # Add an axis for compatibility with sphi and cphi

    # Compute gj using broadcasting and summing over coils
    gj_sin = jnp.sum(gij_expanded * sphi[None, :, :], axis=-1)  # Shape: (num_plasma, len(m))
    gj = gj_sin

    if not stellsym:
        gj_cos = jnp.sum(gij_expanded * cphi[None, :, :], axis=-1)  # Shape: (num_plasma, len(m))
        gj = jnp.concatenate([gj, gj_cos], axis=1)  # Shape: (num_plasma, 2 * len(m))

    # Calculate Ajk
    normal_norms = jnp.linalg.norm(normal_plasma, axis=-1, keepdims=True)  # Shape: (num_plasma, 1)
    gj_normalized = gj / normal_norms  # Normalize gj by normal_plasma

    Ajk = jnp.dot(gj_normalized.T, gj_normalized)  # Shape: (ndofs, ndofs)

    return gj, Ajk

def winding_surface_field_Bn_GI(points_plasma, points_coil, normal_plasma, zeta_coil, theta_coil, G, I, gammadash1_coil, gammadash2_coil):
    # Ensure inputs are JAX arrays
    points_plasma = jnp.asarray(points_plasma)
    points_coil = jnp.asarray(points_coil)
    normal_plasma = jnp.asarray(normal_plasma)
    gammadash1_coil = jnp.asarray(gammadash1_coil)
    gammadash2_coil = jnp.asarray(gammadash2_coil)

    # Constants
    fak = 1e-7  # mu0 / (8 * pi^2)

    # Normalize normal_plasma vectors
    nmag = jnp.linalg.norm(normal_plasma, axis=-1, keepdims=True)
    normal_plasma_normalized = normal_plasma / nmag

    # Vectorized computation of rx, ry, rz (broadcasting over plasma and coil points)
    diff = points_plasma[:, None, :] - points_coil[None, :, :]  # Shape: (num_plasma, num_coil, 3)

    # Compute rmag_inv and rmag_inv_3
    rmag = jnp.linalg.norm(diff, axis=-1)  # Shape: (num_plasma, num_coil)
    rmag_inv = 1.0/rmag  # Shape: (num_plasma, num_coil)
    rmag_inv_3 = rmag_inv**3  # Shape: (num_plasma, num_coil)

    # Compute GI vector
    GI = G * gammadash2_coil - I * gammadash1_coil  # Shape: (num_coil, 3)

    # Compute GI cross r
    GI_cross_r = jnp.cross(GI[None, :,  :], diff, axis=-1)  # Shape: (num_plasma, num_coil, 3)

    # Dot product of GI_cross_r with normal_plasma
    GIcrossr_dotn = jnp.sum(GI_cross_r * normal_plasma_normalized[:, None, :], axis=-1)  # Shape: (num_plasma, num_coil)

    # Compute B_GI
    B_GI = jnp.sum(fak * GIcrossr_dotn * rmag_inv_3, axis=1)  # Shape: (num_plasma,)

    return B_GI
     