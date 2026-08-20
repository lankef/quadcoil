import jax.numpy as jnp
import jax
import lineax as lx
# import matplotlib.pyplot as plt
from jax import jit, lax, vmap, eval_shape
from jax.lax import scan
from functools import partial
from .surface import SurfaceRZFourierJAX
from .quadcoil_params import QuadcoilParams
from .quantities import Phi_with_net_current
from .quantities.current import _K
from .math_utils import project_points_to_plane
import optimistix as optx

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
    """Generate an offset winding surface and fit RZ-Fourier coefficients.

    The surface is constructed by offsetting ``plasma_gamma`` along an
    approximate or user-provided normal vector, then fitting the result with
    :meth:`SurfaceRZFourierJAX.fit_dofs_from_gamma`.
    """
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
    phi_expand = jnp.arctan2(plasma_gamma_expand[:, :, 1], plasma_gamma_expand[:, :, 0]) / jnp.pi / 2 
    theta_expand = jnp.linspace(0, 1, plasma_gamma.shape[1], endpoint=False)[None, :] + jnp.ones_like(phi_expand)

    # gamma_and_scalar_field_to_vtk(weight_remove_invalid[:, :, None] * plasma_gamma_expand, theta_atan, 'ws_new_to_fit.vts')
    dofs_expand = SurfaceRZFourierJAX._fit_dofs_from_gamma(
        phi_target=phi_expand,
        theta_target=theta_expand,
        gamma_target=plasma_gamma_expand,
        nfp=nfp,
        stellsym=stellsym,
        mpol=mpol,
        ntor=ntor,
        lam_tikhonov=0.,
    )

    return(dofs_expand)

def _get_line_intersection(p0, p1, p2, p3):
    """Return whether 2-D line segments ``(p0, p1)`` and ``(p2, p3)`` intersect."""
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


@jit
def bisect_phi(offset_surface, plane_data, rtol=1e-3, atol=1e-3):
    """Per-plane bisection of phi so each point lands on its fitted plane.

    For each (i, j) with i in [0, n) and j in [0, m), find phi such that

        ((offset_surface.gamma_at_point(phi, quadpoints_theta[j])
          - plane_data['origin'][i]) . plane_data['normal'][i]) == 0

    using optimistix's :class:`Bisection`.  The search bracket for slice i is
    ``[quadpoints_phi[i] - 1/(2 nfp), quadpoints_phi[i] + 1/(2 nfp)]``
    (one half field period centred on the original quadrature angle), and all
    n * m scalar bisections are vmapped so they run in parallel.

    Returns
    -------
    grid_phi : ndarray, shape (n, m)
        Phi values such that the n slices each lie on their respective plane.
    """
    nfp = offset_surface.nfp
    quadpoints_theta = offset_surface.quadpoints_theta
    quadpoints_phi = offset_surface.quadpoints_phi
    solver = optx.Bisection(rtol=rtol, atol=atol, flip='detect')

    def residual(phi, args):
        theta_j, origin_i, normal_i = args
        g = offset_surface.gamma_at_point(phi, theta_j)
        return jnp.dot(g - origin_i, normal_i)

    if offset_surface.stellsym:
        half_width = 0.25 / nfp
    else:
        half_width = 0.5 / nfp
    # half_width = 0.5
    
    def solve_one(theta_j, origin_i, normal_i, phi_center):
        sol = optx.root_find(
            residual, solver, y0=phi_center,
            args=(theta_j, origin_i, normal_i),
            options=dict(
                lower=phi_center - half_width,
                upper=phi_center + half_width,
            ),
            throw=False,
        )
        return sol.value

    # Inner vmap: over the m theta points (axis 0 of quadpoints_theta).
    # Outer vmap: over the n planes (axis 0 of origin/normal and quadpoints_phi).
    over_theta  = vmap(solve_one,  in_axes=(0, None, None, None))
    over_planes = vmap(over_theta, in_axes=(None, 0, 0, 0))
    return over_planes(
        quadpoints_theta,
        plane_data['origin'],
        plane_data['normal'],
        quadpoints_phi,
    )


def _polygon_self_intersection(r_pol, z_pol):
    """Mark self-intersecting regions of a closed planar polygon.
    Vectorized O(N^2) all-pairs implementation. Output is binary in {0, 1};
    the function is traceable end-to-end and produces clean (zero) gradients
    almost everywhere thanks to the safe-where pattern below.
    """
    N = r_pol.shape[0]
    # Edge endpoints, shape (N, 2). Edge i goes from p_curr[i] to p_next[i].
    p_curr = jnp.stack([r_pol, z_pol], axis=-1)
    p_next = jnp.roll(p_curr, -1, axis=0)
    # Pairwise broadcast:
    #   axis 0 -> "this" edge  (a)
    #   axis 1 -> "other" edge (b)
    p0 = p_curr[:, None, :]   # (N, 1, 2)
    p1 = p_next[:, None, :]   # (N, 1, 2)
    p2 = p_curr[None, :, :]   # (1, N, 2)
    p3 = p_next[None, :, :]   # (1, N, 2)
    s1 = p1 - p0              # (N, 1, 2)
    s2 = p3 - p2              # (1, N, 2)
    denom = -s2[..., 0] * s1[..., 1] + s1[..., 0] * s2[..., 1]   # (N, N)
    # Double-where: makes the backward pass NaN-safe when denom == 0
    nonzero     = denom != 0
    safe_denom  = jnp.where(nonzero, denom, 1.0)
    inv_denom   = jnp.where(nonzero, 1.0 / safe_denom, 0.0)
    dx = p0[..., 0] - p2[..., 0]                                  # (N, N)
    dy = p0[..., 1] - p2[..., 1]                                  # (N, N)
    s  = (-s1[..., 1] * dx + s1[..., 0] * dy) * inv_denom
    t  = ( s2[..., 0] * dy - s2[..., 1] * dx) * inv_denom
    intersect = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1) & nonzero
    # Drop self / cyclic-adjacent edge pairs (they share a vertex)
    idx     = jnp.arange(N)
    a, b    = idx[:, None], idx[None, :]
    overlap = (a == b) | (a == (b + 1) % N) | ((a + 1) % N == b)
    intersect = intersect & ~overlap
    bad_edge = jnp.any(intersect, axis=1)                          # (N,) bool
    # Reproduce the original parity-toggle sweep:
    # weight[i] = 1 if an even number of bad edges have been seen in [0..i], else 0.
    weight = 1 - (jnp.cumsum(bad_edge.astype(jnp.int32)) % 2)      # (N,) in {0,1}
    # Original post-processing (kept for behavior parity).
    weight = jnp.where(jnp.roll(weight, 1) == 0, 0, 1)
    return weight.astype(r_pol.dtype)

def _graham_scan(r_expand, z_expand):
    """Return a boolean mask marking points on the convex hull."""
    N = r_expand.shape[0]

    # Step 1: Find P0 (lowest z, then leftmost r)
    min_idx = jnp.lexsort((r_expand, z_expand))[0]
    P0 = jnp.array([r_expand[min_idx], z_expand[min_idx]])

    # Step 2: Compute polar angles and distances
    delta_r = r_expand - P0[0]
    delta_z = z_expand - P0[1]
    angles = jnp.arctan2(delta_z, delta_r)
    dists = delta_r**2 + delta_z**2

    # Step 3: Sort indices by angle, break ties with farthest distance
    sort_idx = jnp.lexsort((-dists, angles))
    angles_sorted = angles[sort_idx]

    # Step 4: Keep only the farthest point per unique angle using fixed-size buffer
    def keep_unique_angles():
        init_kept = jnp.zeros(N, dtype=jnp.int32).at[0].set(sort_idx[0])
        init_angle = angles[sort_idx[0]]
        init_count = jnp.array(1, dtype=jnp.int32)

        def body(i, carry):
            kept_indices, last_angle, count = carry
            idx = sort_idx[i]
            angle = angles[idx]
            is_new = angle != last_angle

            kept_indices = lax.cond(
                is_new,
                lambda k: k.at[count].set(idx),
                lambda k: k,
                kept_indices
            )
            last_angle = lax.cond(is_new, lambda _: angle, lambda a: a, last_angle)
            count = count + is_new.astype(jnp.int32)
            return (kept_indices, last_angle, count)

        kept_indices, _, count = lax.fori_loop(1, N, body, (init_kept, init_angle, init_count))
        kept_indices = lax.dynamic_slice(kept_indices, (0,), (count,))
        return kept_indices, count

    kept_idx, M = keep_unique_angles()

    # Step 5: Sort r, z arrays by remaining indices
    r_sorted = r_expand[kept_idx]
    z_sorted = z_expand[kept_idx]

    # Step 6: Graham scan using lax.while_loop
    stack = jnp.zeros(M, dtype=jnp.int32).at[:2].set(jnp.array([0, 1]))
    top = jnp.array(2, dtype=jnp.int32)

    def ccw(i, j, k):
        xi, yi = r_sorted[i], z_sorted[i]
        xj, yj = r_sorted[j], z_sorted[j]
        xk, yk = r_sorted[k], z_sorted[k]
        return (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)

    def cond(state):
        i, top, stack = state
        return i < M

    def body(state):
        i, top, stack = state

        def inner_cond(inner_state):
            top, stack = inner_state
            return jnp.logical_and(top > 1, ccw(stack[top - 2], stack[top - 1], i) <= 0)

        def inner_body(inner_state):
            top, stack = inner_state
            return (top - 1, stack)

        top_new, stack_new = lax.while_loop(inner_cond, inner_body, (top, stack))
        stack_new = stack_new.at[top_new].set(i)
        return (i + 1, top_new + 1, stack_new)

    _, final_top, final_stack = lax.while_loop(cond, body, (2, top, stack))

    # Step 7: Map final hull indices back to original array
    hull_idx = kept_idx[final_stack[:final_top]]
    is_on_hull = jnp.zeros(N, dtype=bool).at[hull_idx].set(True)
    return is_on_hull

@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
    'mpol',
    'ntor',
    'pol_interp',
    'tor_interp',
    'rule',
])
def gen_winding_surface_arc(
        plasma_gamma, d_expand, 
        nfp, stellsym,
        unitnormal=None,
        mpol=5, ntor=5,
        pol_interp=2,
        tor_interp=2,
        lam_tikhonov=1e-5,
        rule='self-intersection',
    ):
    """Generate winding-surface DOFs using arclength poloidal parameterization.

    After creating a uniform offset, invalid points are filtered with
    ``rule`` and the remaining samples are fit with
    :meth:`SurfaceRZFourierJAX.fit_dofs_from_gamma`.
    """
    
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
    r_expand = jnp.sqrt(gamma_uniform[:, :, 0]**2 + gamma_uniform[:, :, 1]**2)
    z_expand = gamma_uniform[:, :, 2]
    ''' Removing self-intersection '''
    if rule == 'self-intersection':
        rule_f = _polygon_self_intersection
    elif rule == 'hull':
        rule_f = _graham_scan
    else:
        raise ValueError('rule must to be \'intersection\' '
                         'or \'hull\'. The current value is: '+ rule)
    weight_remove_invalid = vmap(rule_f, in_axes=0)(r_expand, z_expand)
    
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
    dofs_expand = SurfaceRZFourierJAX._fit_dofs_from_gamma(
        phi_target=phi_expand,
        theta_target=theta_arc,
        gamma_target=gamma_uniform,
        nfp=nfp,
        stellsym=stellsym,
        mpol=mpol,
        ntor=ntor,
        lam_tikhonov=lam_tikhonov,
        custom_weight=weight_remove_invalid,
    )
    return(dofs_expand)

# ----- Meshing -----
def _f_K_integrand(qp, dofs):
    # The linear operator corresponding to the f_K.
    # objective. The eval_surface in qp should be an 
    # offset surface if meshing for an offset surface.
    K_val = _K(qp, dofs)
    da = qp.eval_surface.da()
    return jnp.sqrt(da / 2 * qp.nfp)[:, :, None] * K_val

def qp_nescoil_like_meshing(qp, weights=None, f_affine=_f_K_integrand):
    """
    Solves the following problem:
    min(|r x grad zeta|^2 + |r x grad theta|^2)
    as 2 linear least-squares problems. 

    Generates smooth zeta and theta contours that are 
    nearly perpendicular to each other.

    Styled after qp_nescoil.
    """
    # Generating blank phi for defining A and b
    phi0 = jnp.zeros(qp.ndofs)
    # Default values for weights.
    if weights is None:
        weights = jnp.ones_like(eval_shape(f_affine, qp, {'phi': phi0}))
    neg_b = weights * f_affine(qp, {'phi': phi0})# = A @ 0 - b = -b
    def A_fn(phi):
        return weights * f_affine(qp, {'phi': phi}) - neg_b
    operator = lx.FunctionLinearOperator(A_fn, phi0)
    solution = lx.linear_solve(operator, -neg_b, solver=lx.SVD())
    return solution.value

@partial(jit, static_argnames=['mpol', 'ntor', 'skip_tor'])
def least_square_meshing(
    surf, 
    quadpoints_phi_offset, 
    quadpoints_theta_offset, 
    mpol, ntor, 
    weights=None,
    f_affine=_f_K_integrand,
    skip_tor=False, # For RZ surfaces
):
    qp_dummy_kwargs = {
        'plasma_surface': surf,
        'winding_surface': surf.copy_and_set_quadpoints(
            quadpoints_phi=jnp.linspace(
                0, 1, 
                len(surf.quadpoints_phi) * surf.nfp, 
                endpoint=False
            ),
            quadpoints_theta=surf.quadpoints_theta
        ),
        'Bnormal_plasma': 0,
        'mpol': mpol,
        'ntor': ntor,
        'quadpoints_phi': quadpoints_phi_offset,
        'quadpoints_theta': quadpoints_theta_offset,
        'stellsym': surf.stellsym,
    }
    theta_qp = QuadcoilParams(
        net_poloidal_current_amperes=0,
        net_toroidal_current_amperes=1,
        **qp_dummy_kwargs
    )
    theta_dofs = {'phi': qp_nescoil_like_meshing(theta_qp, weights=weights, f_affine=f_affine)}
    theta_val = Phi_with_net_current(theta_qp, theta_dofs)
    if skip_tor:
        return theta_val
    phi_qp = QuadcoilParams(
        net_poloidal_current_amperes=1,
        net_toroidal_current_amperes=0,
        **qp_dummy_kwargs
    )
    phi_dofs = {'phi': qp_nescoil_like_meshing(phi_qp, weights=weights, f_affine=f_affine)}
    phi_val = Phi_with_net_current(phi_qp, phi_dofs)
    return phi_val, theta_val, phi_qp, phi_dofs, theta_dofs, theta_val
    