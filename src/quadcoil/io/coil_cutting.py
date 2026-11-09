
# Import packages.
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Importing simsopt 
from quadcoil import QuadcoilParams
from quadcoil.quantities import Phi_with_net_current

def coil_zeta_theta_from_qp(
    qp:QuadcoilParams,
    dofs,
    coils_per_half_period=5,
    theta_shift=0):
    """
    Extract coil contours from quadcoil solution.
    
    Requires matplotlib for contour extraction.
    
    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for coil cutting. "
            "Install with: pip install matplotlib or pip install quadcoil[coil-cutting]"
        )
    
    stellsym = qp.stellsym
    nzeta_coil = len(qp.quadpoints_phi)
    nfp = qp.nfp 
    theta = qp.quadpoints_theta * 2 * np.pi 
    zeta = qp.quadpoints_phi * 2 * np.pi 
    net_poloidal_current_amperes = qp.net_poloidal_current_amperes 

    # ------------------------
    # Load current potential
    # ------------------------
    current_potential = Phi_with_net_current(qp, dofs)

    # Now just generate a new monotonic array with the correct first value:
    theta = theta[0] + np.linspace(0,2*np.pi,len(theta),endpoint=False)

    d = 2*np.pi/nfp
    # We add a field period to the "left" and "right" of the 
    # present field period so that no contours are cutoff.
    zeta_3 = np.concatenate((zeta-d, zeta, zeta+d))
    # If the net current is non-neglegible, use the net current to calculate spacing.
    # Else, use the maximum phi to calculate spacing.
    # we normalize the current potential correspondingly into "data",
    # and also pad it to contain 3 field periods.
    if abs(net_poloidal_current_amperes) > np.finfo(float).eps:
        data = current_potential / net_poloidal_current_amperes * nfp  
        data_3 = np.concatenate((data-1,data,data+1))
    else:
        data = current_potential / np.max(current_potential)
        data_3 = np.concatenate((data,data,data))

    # We only generate contours based on the "middle"
    # field period. 
    d = 0.5/coils_per_half_period
    level_shift = theta_shift % 1 * d
    if stellsym:
        if theta_shift != 0:
            print('Warning: non-zero theta-shift is not permitted when stell-sym is true.')
        levels = np.linspace(0,0.5,coils_per_half_period,endpoint=False)
    else:
        levels = np.linspace(0,1,coils_per_half_period*2,endpoint=False) + level_shift
    # The contours are symmetric w.r.t. 2pi/nfp/2, i.e. stellarator
    # symmetric, by default.
    levels = levels + d/2
    fig = plt.figure()
    cdata = plt.contour(zeta_3, theta, np.transpose(data_3), levels, colors='k')
    plt.close(fig)
    num_coils_found = len(cdata.levels)
    if num_coils_found != 2*coils_per_half_period:
        print("Warning: The expected number of levels != 2 * coils_per_half_period.")

    contour_zeta=[]
    contour_theta=[]
    num_coils = 0
    for j in range(num_coils_found):
        paths_in_level = cdata.allsegs[j]
        # Each level can contain many paths.
        for p in paths_in_level:
            plt.plot(p[:, 0], p[:, 1])
            # These are the zeta (toroidal angles)
            # and theta of the paths.
            contour_zeta.append(p[:, 0])
            contour_theta.append(p[:, 1])
            num_coils += 1

    print('Cutting complete. Number of coils/half field period:', num_coils)
    return(contour_zeta, contour_theta)

# This script assumes the contours do not zig-zag back and forth across the theta=0 line,
# after shifting the current potential by theta_shift grid points.
# The nescin file is used to provide the coil winding surface, so make sure this is consistent with the regcoil run.
# ilambda is the index in the lambda scan which you want to select.
# def cut_coil(qp, qpst):
# filename = 'regcoil_out.li383.nc' # sys.argv[1]
# TODO: these tow have a lot of duplicate code. Clean up when finalizing how QUADCOIL is integrated.
def gamma_current_from_qp(
    qp: QuadcoilParams,
    dofs,
    coils_per_half_period=1,
    theta_shift=0,
):
    """Contour a QUADCOIL solution into centerline samples and currents.

    Parameters
    ----------
    qp : QuadcoilParams
        QUADCOIL problem configuration.
    dofs : dict
        Optimized QUADCOIL degrees of freedom.
    coils_per_half_period : int, optional
        Number of contours per half field period.
    theta_shift : float, optional
        Fractional shift applied to contour levels.

    Returns
    -------
    gammas : list of ndarray, each shape (n_j, 3)
        Sampled XYZ centerlines (metres), one per cut coil.
    currents : list of float
        Current [A] for each cut coil (uniform magnitude).
    """
    contour_zeta, contour_theta = coil_zeta_theta_from_qp(
        qp=qp,
        dofs=dofs,
        coils_per_half_period=coils_per_half_period,
        theta_shift=theta_shift
    )
    num_coils = len(contour_zeta)
    net_poloidal_current_amperes = qp.net_poloidal_current_amperes

    # ------------------------
    # Load surface shape
    # ------------------------
    contour_R = []
    contour_Z = []
    for j in range(num_coils):
        contour_R.append(np.zeros_like(contour_zeta[j]))
        contour_Z.append(np.zeros_like(contour_zeta[j]))

    surf = qp.winding_surface.to_simsopt()
    for m in range(surf.mpol+1): # 0 to mpol
        for i in range(2*surf.ntor+1):
            n = i-surf.ntor
            crc = surf.rc[m, i]
            czs = surf.zs[m, i]
            if surf.stellsym:
                crs = 0
                czc = 0
            else: # Returns ValueError for stellsym cases
                crs = surf.get_rs(m, n)
                czc = surf.get_zc(m, n)
            for j in range(num_coils):
                angle = m*contour_theta[j] - n*contour_zeta[j]*surf.nfp
                # Was filled with zeroes.
                # Are lists because contou lengths are not uniform.
                contour_R[j] = contour_R[j] + crc*np.cos(angle) + crs*np.sin(angle)
                contour_Z[j] = contour_Z[j] + czs*np.sin(angle) + czc*np.cos(angle)

    gammas = []
    for j in range(num_coils):
        x = contour_R[j] * np.cos(contour_zeta[j])
        y = contour_R[j] * np.sin(contour_zeta[j])
        gammas.append(np.column_stack([x, y, contour_Z[j]]))

    coil_current = net_poloidal_current_amperes / num_coils / qp.nfp
    if qp.stellsym:
        coil_current = coil_current / 2
    currents = [coil_current] * num_coils
    return gammas, currents


def curve_from_gamma(gamma, order, ppp=20):
    """Fit one simsopt ``CurveXYZFourier`` to a sampled centerline.

    Parameters
    ----------
    gamma : ndarray, shape (n, 3)
        Sampled XYZ points along the curve (one period, ordered).
    order : int
        Fourier order. Must satisfy ``order < n / 2``.
    ppp : int, optional
        Quadrature points per Fourier period for the fitted curve.

    Returns
    -------
    simsopt.geo.CurveXYZFourier
        Curve whose dofs are the least-squares Fourier fit to ``gamma``.

    Raises
    ------
    ValueError
        If ``order`` is not smaller than ``n / 2``.
    ImportError
        If simsopt is not installed.
    """
    try:
        from simsopt.geo import CurveXYZFourier
    except ImportError:
        raise ImportError('Simsopt is required to use the coil-cutting features.')

    gamma = np.asarray(gamma)
    n = gamma.shape[0]
    if not (order < n / 2):
        raise ValueError("order must be smaller than sample point number /2")

    # Design matrix for DOF order [c0, s1, c1, ..., s_order, c_order].
    # Angle convention matches JaxCurveXYZFourier: 2π * quadpoint.
    phi = np.linspace(0.0, 1.0, n, endpoint=False)
    angle = 2.0 * np.pi * phi
    A = np.empty((n, 2 * order + 1))
    A[:, 0] = 1.0
    for m in range(1, order + 1):
        A[:, 2 * m - 1] = np.sin(m * angle)
        A[:, 2 * m] = np.cos(m * angle)

    dofs = []
    for k in range(3):
        dofs.append(np.linalg.lstsq(A, gamma[:, k], rcond=None)[0])

    curve = CurveXYZFourier(order * ppp, order)
    curve.local_x = np.concatenate(dofs)
    return curve


def _rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion ``[q0, qi, qj, qk]``."""
    R = np.asarray(R)
    t = np.trace(R)
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        q0 = 0.25 * s
        qi = (R[2, 1] - R[1, 2]) / s
        qj = (R[0, 2] - R[2, 0]) / s
        qk = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q0 = (R[2, 1] - R[1, 2]) / s
        qi = 0.25 * s
        qj = (R[0, 1] + R[1, 0]) / s
        qk = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q0 = (R[0, 2] - R[2, 0]) / s
        qi = (R[0, 1] + R[1, 0]) / s
        qj = 0.25 * s
        qk = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q0 = (R[1, 0] - R[0, 1]) / s
        qi = (R[0, 2] + R[2, 0]) / s
        qj = (R[1, 2] + R[2, 1]) / s
        qk = 0.25 * s
    q = np.array([q0, qi, qj, qk])
    return q / np.linalg.norm(q)


def planar_curve_from_gamma(gamma, order=10, ppp=20):
    """Fit one simsopt ``CurvePlanarFourier`` to a sampled centerline.

    Finds the nearest plane to ``gamma``, projects the points onto it, and
    least-squares fits the polar Fourier series. Sample angles are taken as
    equally spaced in the simsopt parameterization, ``φ_i = 2π i / n``.

    Parameters
    ----------
    gamma : ndarray, shape (n, 3)
        Sampled XYZ points along the curve (one period, ordered).
    order : int, optional
        Fourier order. Must satisfy ``order < n / 2``.
    ppp : int, optional
        Quadrature points per Fourier period for the fitted curve.

    Returns
    -------
    simsopt.geo.CurvePlanarFourier
        Planar curve whose dofs are the least-squares Fourier fit to ``gamma``.

    Raises
    ------
    ValueError
        If ``gamma`` is not shape ``(n, 3)`` or ``order`` is not smaller
        than ``n / 2``.
    ImportError
        If simsopt is not installed.
    """
    try:
        from simsopt.geo import CurvePlanarFourier
    except ImportError:
        raise ImportError('Simsopt is required to use the coil-cutting features.')

    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim != 2 or gamma.shape[1] != 3:
        raise ValueError("gamma must have shape (n_points, 3)")
    n = gamma.shape[0]
    if not (order < n / 2):
        raise ValueError("order must be smaller than sample point number /2")

    center = np.mean(gamma, axis=0)
    centered = gamma - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    normal = normal / np.linalg.norm(normal)

    proj = gamma - np.outer(centered @ normal, normal)

    v0 = proj[0] - center
    v0_norm = np.linalg.norm(v0)
    if v0_norm < 1e-14:
        e1 = Vt[0] - np.dot(Vt[0], normal) * normal
        e1_norm = np.linalg.norm(e1)
        if e1_norm < 1e-14:
            e1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(e1) < 1e-14:
                e1 = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        e1 = e1 / np.linalg.norm(e1)
    else:
        e1 = v0 / v0_norm

    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)
    if n >= 2 and np.dot(proj[1] - center, e2) < 0.0:
        normal = -normal
        e2 = -e2

    R = np.column_stack((e1, e2, normal))
    q = _rotation_matrix_to_quaternion(R)

    u = (proj - center) @ e1
    v = (proj - center) @ e2
    r = np.hypot(u, v)

    phi = 2.0 * np.pi * np.linspace(0.0, 1.0, n, endpoint=False)
    A = np.empty((n, 2 * order + 1))
    A[:, 0] = 1.0
    for m in range(1, order + 1):
        A[:, m] = np.cos(m * phi)
        A[:, order + m] = np.sin(m * phi)
    rc_rs = np.linalg.lstsq(A, r, rcond=None)[0]

    dofs = np.concatenate([rc_rs, q, center])
    curve = CurvePlanarFourier(order * ppp, order)
    curve.set_dofs(dofs)
    return curve


def simsopt_coil_from_qp(
    qp, dofs, coils_per_half_period, theta_shift=0,
    method=gamma_current_from_qp,
    base_mode=False,
    order=10, ppp=40):
    '''
    Cut a QUADCOIL current-potential solution into a Simsopt coil set.

    This helper contours the full current potential on ``qp.winding_surface``,
    fits each contour with a ``simsopt.geo.CurveXYZFourier``, assigns a
    ``simsopt.field.Current``, and, by default, expands the base curves using
    Simsopt field-period and stellarator symmetries.

    Parameters
    ----------
    qp : QuadcoilParams
        QUADCOIL problem configuration returned by ``quadcoil.quadcoil``.
    dofs : dict
        Optimized QUADCOIL degrees of freedom, typically the ``dofs_opt``
        dictionary returned by ``quadcoil.quadcoil``.
    coils_per_half_period : int
        Number of current-potential contours to cut per half field period.
    theta_shift : float, optional
        Fractional shift applied to the contour levels. Nonzero shifts are
        ignored when stellarator symmetry is enabled.
    method : callable, optional
        Function used to generate centerlines and currents from ``qp`` and
        ``dofs``. By default, ``gamma_current_from_qp`` is used.
    base_mode : bool, optional
        If ``True``, return the base curves and currents before symmetry
        expansion. If ``False``, return the symmetry-expanded Simsopt coil set.
    order : int, optional
        Fourier order for the fitted ``CurveXYZFourier`` curves.
    ppp : int, optional
        Quadrature points per Fourier period used for the fitted curves.

    Returns
    -------
    list
        A list of Simsopt coils generated by ``coils_via_symmetries`` when
        ``base_mode=False``.
    tuple
        ``(curves, currents)`` when ``base_mode=True``.
    '''
    try:     
        from simsopt.field import Current # , Coil
        from simsopt.field.coil import coils_via_symmetries
    except:
        raise ImportError('Simsopt is required to use the coil-cutting features.')
    gammas, currents_scalar_list = method(
        qp=qp,
        dofs=dofs,
        coils_per_half_period=coils_per_half_period,
        theta_shift=theta_shift,
    )
    curves = [curve_from_gamma(g, order=order, ppp=ppp) for g in gammas]
    currents = [Current(I) for I in currents_scalar_list]
    if base_mode:
        return curves, currents
    coils = coils_via_symmetries(curves, currents, qp.nfp, qp.stellsym)
    return coils
