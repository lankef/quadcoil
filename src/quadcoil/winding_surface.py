import jax.numpy as jnp
import jax
# import matplotlib.pyplot as plt
from jax import jit, lax, vmap
from jax.lax import scan
from functools import partial
from .surfacerzfourier_jax import dof_to_rz_op, SurfaceRZFourierJAX
import lineax as lx
import equinox as eqx

# @partial(jit, static_argnames=['nfp', 'stellsym', 'mpol', 'ntor', 'lam_tikhonov',])
def fit_surfacerzfourier(
        phi_grid, theta_grid, 
        r_fit, z_fit, 
        nfp:int, stellsym:bool, 
        mpol:int=5, ntor:int=5, 
        lam_tikhonov=0., 
        custom_weight=None,):
    # Fits r and z with a surface

    A_lstsq, m_2_n_2 = dof_to_rz_op(
        theta_grid=theta_grid, 
        phi_grid=phi_grid,
        nfp=nfp, 
        stellsym=stellsym, 
        mpol=mpol, 
        ntor=ntor
    )
    
    b_lstsq = jnp.concatenate([r_fit[:, :, None], z_fit[:, :, None]], axis=2)
    # Weight each point by the sum of the length of the two 
    # poloidal segments that each vertex is attached to
    # weight = jacobian # 
    max_r_slice = jnp.max(r_fit, axis=1)[:, None]
    min_r_slice = jnp.min(r_fit, axis=1)[:, None]
    # A and b of the lstsq problem.
    # A_lstsq is a function of phi_grid and theta_grid
    # b_lstsq is differentiable.
    # A_lstsq has shape: [nphi, ntheta, 2(rz), ndof]
    # b_lstsq has shape: [nphi, ntheta, 2(rz)]
    if custom_weight is not None:
        if custom_weight.shape != A_lstsq.shape[:2]:
            raise ValueError(
                'custom_weight must have the shape ' 
                + str(A_lstsq.shape[:2]) 
                + ', but it has shape ' 
                + str(custom_weight.shape)
            )
        A_lstsq = A_lstsq * custom_weight[:, :, None, None]
        b_lstsq = b_lstsq * custom_weight[:, :, None]
    A_lstsq = A_lstsq.reshape(-1, A_lstsq.shape[-1])
    b_lstsq = b_lstsq.flatten()

    # tikhonov regularization for higher harmonics
    lam = lam_tikhonov * jnp.diag(m_2_n_2)
    
    # The lineax call fulfills the same purpose as the following:
    # dofs_expand, resid, rank, s = jnp.linalg.lstsq(A_lstsq.T.dot(A_lstsq) + lam, A_lstsq.T.dot(b_lstsq))
    # but is faster and more robust to gradients.
    operator = lx.MatrixLinearOperator(A_lstsq.T.dot(A_lstsq) + lam)
    # solver = lx.QR()  # or lx.AutoLinearSolver(well_posed=None)
    solver = lx.AutoLinearSolver(well_posed=False)
    solution = lx.linear_solve(operator, A_lstsq.T.dot(b_lstsq), solver)
    return(solution.value)

# An approximation for unit normal.
# and include the endpoints
gen_rot_matrix = lambda theta: jnp.array([
    [jnp.cos(theta), -jnp.sin(theta), 0],
    [jnp.sin(theta),  jnp.cos(theta), 0],
    [0,              0,             1]
])

# @partial(jit, static_argnames=[
#     'nfp', 'stellsym', 
#     'mpol', 'ntor', 
# ])
def gen_winding_surface_offset(
        plasma_gamma, d_expand, 
        nfp, stellsym,
        unitnormal=None,
        mpol=10, ntor=10,
    ):
    # A simple winding surface generator with less intermediate quantities.
    # only works for large offset distances, where center (from the unweighted
    # avg of the quadrature points' rz coordinate) of the offset surface's rz cross sections
    # lay within the cross sections. 
    theta = 2 * jnp.pi / nfp
    rotation_matrix = gen_rot_matrix(theta)

    # Approximately calculating the normal vector. Alternatively, the normal
    # can be provided, but this will make the Jacobian matrix larger and lead to longer compile time.
    if unitnormal is None:
        xyz_rotated = plasma_gamma[0, :, :] @ rotation_matrix.T
        plasma_gamma_phi_rolled = jnp.append(plasma_gamma[1:, :, :], xyz_rotated[None, :, :], axis=0)
        delta_phi = plasma_gamma_phi_rolled - plasma_gamma
        delta_theta = jnp.roll(plasma_gamma, 1, axis=1) - plasma_gamma
        normal_approx = jnp.cross(delta_theta, delta_phi)
        unitnormal = normal_approx / jnp.linalg.norm(normal_approx, axis=-1)[:,:,None]
    
    # Copy the next field period 
    if stellsym:
        # If stellsym, then only use half of the field period for surface fitting
        len_phi = plasma_gamma.shape[0]//2
        plasma_gamma_expand = (
            plasma_gamma[:len_phi] 
            + unitnormal[:len_phi] * d_expand)
    else:
        plasma_gamma_expand = plasma_gamma + unitnormal * d_expand

    # The original uniform offset. Has self-intersections.
    # Tested to be differentiable.
    r_expand = jnp.sqrt(plasma_gamma_expand[:, :, 1]**2 + plasma_gamma_expand[:, :, 0]**2)
    phi_expand = jnp.arctan2(plasma_gamma_expand[:, :, 1], plasma_gamma_expand[:, :, 0]) / jnp.pi / 2 
    theta_expand = jnp.linspace(0, 1, plasma_gamma.shape[1], endpoint=False)[None, :] + jnp.ones_like(phi_expand)
    z_expand = plasma_gamma_expand[:, :, 2]

    # gamma_and_scalar_field_to_vtk(weight_remove_invalid[:, :, None] * plasma_gamma_expand, theta_atan, 'ws_new_to_fit.vts')
    dofs_expand = fit_surfacerzfourier(
        mpol=mpol,
        ntor=ntor,
        theta_grid=theta_expand, # theta_interp
        phi_grid=phi_expand,
        r_fit=r_expand,
        z_fit=z_expand,
        nfp=nfp, stellsym=stellsym,
        lam_tikhonov=0., 
    )

    return(dofs_expand)

def get_line_intersection(p0, p1, p2, p3):
    # Detects if two line segments given by 
    # p0 (x, y), p1 (x, y);
    # p1 (x, y), p2 (x, y)
    # intersects.
    s1 = p1 - p0
    s2 = p3 - p2
    denom = -s2[0] * s1[1] + s1[0] * s2[1]
    # Preventing division by zero
    inv_denom = jnp.where(denom!=0, 1/denom, 0)
    s = (-s1[1] * (p0[0] - p2[0]) + s1[0] * (p0[1] - p2[1])) * inv_denom
    t = ( s2[0] * (p0[1] - p2[1]) - s2[1] * (p0[0] - p2[0])) * inv_denom
    return (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1) & (denom!=0)

# @jit
def polygon_self_intersection(r_pol, z_pol):
    len_theta = len(r_pol)
    # Takes a planar polygon and removes self-intersecting regions.
    # Returns a weight array that is 1 for every point where the 
    # adjacent edges contain self-intersection.
    # Assumes that the first point in the input is on the polygon to keep.
    # shape: len_phi, 2 (r, z)
    p0_in = jnp.concatenate([r_pol[:, None], z_pol[:, None]], axis=-1)
    p1_in = jnp.roll(p0_in, -1, axis=0)
    # shape: len_phi, 4 (r0, z0, r1, z1)
    p0p1 = jnp.concatenate([p0_in, p1_in], axis=1)
    # Outer scan
    def outer_loop(carry_outer, x_outer):
        index_a, weight = carry_outer
        def inner_loop(carry, x):
            # carry is (index of p0p1, ([r0, z0, r1, z1]), index of p2p3)
            # x is ([r2, z2, r3, z3])
            index_a, r0z0r1z1, index_b = carry
            # Is the index of the second line segment
            # one greater or lower than that of the current line segment?
            # If so, get_line_intersection will throw a False positive
            # and has to be disregarded.
            is_overlapping = (
                (index_a == index_b) 
                | (index_a == (index_b+1)%len_theta)
                | ((index_a+1)%len_theta == index_b)
            )
            p0_i = r0z0r1z1[:2]
            p1_i = r0z0r1z1[2:]
            p2_i = x[:2]
            p3_i = x[2:]
            # True when intersection is present
            is_intersect = get_line_intersection(p0_i, p1_i, p2_i, p3_i)
            return (index_a, r0z0r1z1, index_b+1), is_intersect & jnp.logical_not(is_overlapping)
        _, is_intersect = scan(inner_loop, (index_a, x_outer, 0), p0p1)
        has_self_intersection = jnp.any(is_intersect)
        # flip the sign of weight if self intersection is detected. 
        # We assume that the outboard side is not self-intersecting (weight=1)
        # This will allow us to mark all self-intersecting regions with 
        # weight = -1.
        weight = jnp.where(has_self_intersection, -weight, weight)
        # if has_self_intersection:
        #     plt.plot(p0p1[:, 0], p0p1[:, 1])
        #     plt.scatter(p0p1[:, 0], p0p1[:, 1], alpha = is_intersect)
        #     plt.show()
        return (index_a + 1, weight), weight
    _, weight = scan(outer_loop, (0, 1), p0p1)
    # Convert weight = +-1 to 0 and 1
    weight = (weight+1)/2
    # Thus far we've marked all vertices where the edge
    # that follows contains self-intersection. We now also
    # change the weight of the vertices that is preceded 
    # by a self-intersecting edge.
    weight = jnp.where(jnp.roll(weight, 1)==0, 0, 1)
    return(weight)

@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
    'mpol',
    'ntor',
    'pol_interp',
    'tor_interp',
    # 'lam_tikhonov'
])
def gen_winding_surface_atan(
        plasma_gamma, d_expand, 
        nfp, stellsym,
        unitnormal=None,
        mpol=5, ntor=5,
        pol_interp=2,
        tor_interp=2,
        lam_tikhonov=1e-5,
    ):
    ''' Create uniform offset '''
    uniform_offset_dofs = gen_winding_surface_offset(
        plasma_gamma, d_expand, 
        nfp, stellsym,
        unitnormal=unitnormal,
        mpol=mpol, ntor=ntor,
    )
    ''' Interpolate to generate smooth poloidal cross sections '''
    phi_expand = jnp.linspace(0, 1/nfp, plasma_gamma.shape[0] * tor_interp)
    uniform_offset_surface_jax = SurfaceRZFourierJAX(
        nfp=nfp, stellsym=stellsym, 
        mpol=mpol, ntor=ntor, 
        quadpoints_phi=phi_expand, 
        quadpoints_theta=jnp.linspace(0, 1, plasma_gamma.shape[1] * pol_interp, endpoint=False), 
        dofs=uniform_offset_dofs
    )
    gamma_uniform = uniform_offset_surface_jax.gamma()
    ''' Trimming based on stellarator symmetry '''
    # Fit only half a field period when stellsym.
    if stellsym:
        # If stellsym, then only use half of the field period for surface fitting
        len_phi = len(phi_expand)//2
        gamma_uniform = gamma_uniform[:len_phi]
        phi_expand = phi_expand[:len_phi]
        # finding center to generate poloidal parameterization
        r_plasma = jnp.sqrt(plasma_gamma[:len_phi, :, 1]**2 + plasma_gamma[:len_phi, :, 0]**2)
        z_plasma = plasma_gamma[:len_phi, :, 2]
    else:
        gamma_uniform = gamma_uniform
        # Copy the gamma from the next and last fp.
        # finding center to generate poloidal parameterization
        r_plasma = jnp.sqrt(plasma_gamma[:, :, 1]**2 + plasma_gamma[:, :, 0]**2)
        z_plasma = plasma_gamma[:, :, 2]
    r_center = jnp.average(r_plasma, axis=-1)
    z_center = jnp.average(z_plasma, axis=-1)
    # The original uniform offset. Has self-intersections.
    # Tested to be differentiable.
    r_expand = jnp.sqrt(gamma_uniform[:, :, 1]**2 + gamma_uniform[:, :, 0]**2)
    z_expand = gamma_uniform[:, :, 2]
    ''' Removing self-intersection '''
    weight_remove_invalid = vmap(polygon_self_intersection, in_axes=0)(r_expand, z_expand)
    ''' Fitting surface'''
    theta_atan = jnp.arctan2(z_expand-z_center[:, None], r_expand-r_center[:, None])/jnp.pi/2
    theta_atan = jnp.where(theta_atan>0, theta_atan, theta_atan+1)
    phi_expand, theta_atan = jnp.broadcast_arrays(phi_expand[:, None], theta_atan)
    dofs_expand = fit_surfacerzfourier(
        mpol=mpol,
        ntor=ntor,
        theta_grid=theta_atan, # theta_interp
        phi_grid=phi_expand,
        r_fit=r_expand,
        z_fit=z_expand,
        nfp=nfp, stellsym=stellsym,
        lam_tikhonov=lam_tikhonov,
        custom_weight=weight_remove_invalid,
    )
    return(dofs_expand)

@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
    'mpol',
    'ntor',
    'pol_interp',
    'tor_interp',
    # 'lam_tikhonov'
])
def gen_winding_surface_arc(
        plasma_gamma, d_expand, 
        nfp, stellsym,
        unitnormal=None,
        mpol=5, ntor=5,
        pol_interp=2,
        tor_interp=2,
        lam_tikhonov=1e-5,
    ):
    
    # ----- Create uniform offset -----
    uniform_offset_dofs = gen_winding_surface_offset(
        plasma_gamma, d_expand, 
        nfp, stellsym,
        unitnormal=unitnormal,
        mpol=mpol, ntor=ntor,
    )
    
    # ----- Interpolate to generate smooth poloidal cross sections -----
    phi_expand = jnp.linspace(0, 1/nfp, plasma_gamma.shape[0] * tor_interp)
    uniform_offset_surface_jax = SurfaceRZFourierJAX(
        nfp=nfp, stellsym=stellsym, 
        mpol=mpol, ntor=ntor, 
        quadpoints_phi=phi_expand, 
        quadpoints_theta=jnp.linspace(0, 1, plasma_gamma.shape[1] * pol_interp, endpoint=False), 
        dofs=uniform_offset_dofs
    )
    gamma_uniform = uniform_offset_surface_jax.gamma()
    
    # ----- Trimming based on stellarator symmetry -----
    # Fit only half a field period when stellsym.
    if stellsym:
        # If stellsym, then only use half of the field period for surface fitting
        len_phi = len(phi_expand)//2
        gamma_uniform = gamma_uniform[:len_phi]
        phi_expand = phi_expand[:len_phi]
        # finding center to generate poloidal parameterization
        r_plasma = jnp.sqrt(plasma_gamma[:len_phi, :, 1]**2 + plasma_gamma[:len_phi, :, 0]**2)
        z_plasma = plasma_gamma[:len_phi, :, 2]
    else:
        gamma_uniform = gamma_uniform
        # Copy the gamma from the next and last fp.
        # finding center to generate poloidal parameterization
        r_plasma = jnp.sqrt(plasma_gamma[:, :, 1]**2 + plasma_gamma[:, :, 0]**2)
        z_plasma = plasma_gamma[:, :, 2]
    r_center = jnp.average(r_plasma, axis=-1)
    z_center = jnp.average(z_plasma, axis=-1)
    # The original uniform offset. Has self-intersections.
    # Tested to be differentiable.
    r_expand = jnp.sqrt(gamma_uniform[:, :, 1]**2 + gamma_uniform[:, :, 0]**2)
    z_expand = gamma_uniform[:, :, 2]
    ''' Removing self-intersection '''
    weight_remove_invalid = vmap(polygon_self_intersection, in_axes=0)(r_expand, z_expand)
    
    # ----- Calculating parameterization -----
    r_wrapped = jnp.pad(r_expand, pad_width=((0, 0), (0, 1)), mode='wrap')
    z_wrapped = jnp.pad(z_expand, pad_width=((0, 0), (0, 1)), mode='wrap')
    # Compute the differences along axis=1 (between successive points)
    dr = jnp.diff(r_wrapped, axis=1)
    dz = jnp.diff(z_wrapped, axis=1)
    # Compute the Euclidean distance for each segment
    segment_lengths = jnp.sqrt(dr**2 + dz**2)
    # Sum the segment lengths to get the total arclength for each curve
    arclengths = jnp.cumsum(segment_lengths, axis=1)
    theta_arc = (arclengths - arclengths[:, 0][:, None]) / arclengths[:, -1][:, None]
    phi_expand, theta_arc = jnp.broadcast_arrays(phi_expand[:, None], theta_arc)

    # ----- Fitting surface -----
    dofs_expand = fit_surfacerzfourier(
        mpol=mpol,
        ntor=ntor,
        theta_grid=theta_arc, # theta_interp
        phi_grid=phi_expand,
        r_fit=r_expand,
        z_fit=z_expand,
        nfp=nfp, stellsym=stellsym,
        lam_tikhonov=lam_tikhonov,
        custom_weight=weight_remove_invalid,
    )
    return(dofs_expand)