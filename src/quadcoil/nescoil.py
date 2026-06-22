import jax.numpy as jnp
from jax import jit
from functools import partial
import lineax as lx
from quadcoil.quantity.magnetic_field import _Bnormal
from quadcoil import SurfaceRZFourierJAX, QuadcoilParams
from quadcoil.wrapper import _resolve_quadpoints


@jit
def _f_B_integrand(qp, dofs):
    # The nescoil objective.
    Bnormal_val = _Bnormal(qp, dofs)
    return jnp.sqrt(qp.plasma_surface.da()/2 * qp.nfp) * Bnormal_val


@jit
def qp_nescoil(qp):
    """
    Solve for phi minimizing sum(|_f_B_integrand(qp, {'phi': phi})|^2).

    _f_B_integrand is affine in phi: A @ phi - b.
    Setting phi=0 gives -b, so b = -_f_B_integrand(qp, {'phi': 0}).
    The purely linear part is A_fn(phi) = _f_B_integrand(..., phi) - _f_B_integrand(..., 0).
    We solve A @ phi = b in the least-squares sense via SVD.
    """
    phi0 = jnp.zeros(qp.ndofs)
    neg_b = _f_B_integrand(qp, {'phi': phi0})  # = A @ 0 - b = -b

    def A_fn(phi):
        return _f_B_integrand(qp, {'phi': phi}) - neg_b

    operator = lx.FunctionLinearOperator(A_fn, phi0)
    solution = lx.linear_solve(operator, -neg_b, solver=lx.SVD())
    return solution.value

NESCOIL_STATIC_ARGNAMES = [
    'nfp',
    'stellsym',
    'mpol',
    'ntor',
    'plasma_mpol',
    'plasma_ntor',
    'plasma_stellsym',
    'surface_type',
    'winding_surface_mode',
    'winding_theta_mode',
    'winding_mpol',
    'winding_ntor',
    'winding_stellsym',
]

@partial(jit, static_argnames=NESCOIL_STATIC_ARGNAMES)
def nescoil(
    nfp: int,
    stellsym: bool,
    plasma_mpol: int,
    plasma_ntor: int,
    plasma_dofs,
    net_poloidal_current_amperes: float,

    # -- Defaults --

    net_toroidal_current_amperes: float = 0.,
    mpol: int = 6,
    ntor: int = 4,
    quadpoints_phi=None,
    quadpoints_theta=None,

    plasma_stellsym: bool = True,
    plasma_quadpoints_phi=None,
    plasma_quadpoints_theta=None,
    Bnormal_plasma=None,

    plasma_coil_distance: float = None,
    surface_type: str = 'SurfaceRZFourier',

    winding_dofs=None,
    winding_mpol: int = 6,
    winding_ntor: int = 5,
    winding_quadpoints_phi=None,
    winding_quadpoints_theta=None,
    winding_stellsym: bool = True,
    winding_phi_interp: int = 2,
    winding_theta_interp: int = 5,
    winding_theta_rule_subsample: int = None,
    winding_lam_tikhonov: float = 1e-5,
    winding_surface_mode: str = 'self-intersection',
    winding_theta_mode: str = 'arclen',
):
    r'''
    Solves a NESCOIL problem: finds the current potential :math:`\Phi_{sv}`
    on the winding surface that minimizes the normal field error
    :math:`\sum |B_n|^2` on the plasma boundary.

    Parameters
    ----------
    nfp : int
        (Static) The number of field periods.
    stellsym : bool
        (Static) Whether the coils have stellarator symmetry.
    plasma_mpol : int
        (Static) The number of poloidal Fourier harmonics in the plasma boundary.
    plasma_ntor : int
        (Static) The number of toroidal Fourier harmonics in the plasma boundary.
    plasma_dofs : ndarray
        The plasma surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` convention.
    net_poloidal_current_amperes : float
        The net poloidal current :math:`G`.
    net_toroidal_current_amperes : float, optional, default=0
        The net toroidal current :math:`I`.
    mpol : int, optional, default=6
        (Static) The number of poloidal Fourier harmonics in the current potential :math:`\Phi_{sv}`.
    ntor : int, optional, default=4
        (Static) The number of toroidal Fourier harmonics in :math:`\Phi_{sv}`.
    quadpoints_phi : ndarray, shape (nphi,), optional, default=None
        The poloidal quadrature points on the winding surface to evaluate the objective at.
        Uses one period from the winding surface by default.
    quadpoints_theta : ndarray, shape (ntheta,), optional, default=None
        The toroidal quadrature points on the winding surface to evaluate the objective at.
        Uses one period from the winding surface by default.
    plasma_stellsym : bool, optional, default=True
        (Static) Whether the plasma has stellarator symmetry.
    plasma_quadpoints_phi : ndarray, shape (nphi_plasma,), optional, default=None
        Will be set based on the shape of ``Bnormal_plasma`` if provided,
        or default to ``jnp.linspace(0, 1/nfp, 32, endpoint=False)`` otherwise.
    plasma_quadpoints_theta : ndarray, shape (ntheta_plasma,), optional, default=None
        Will be set based on the shape of ``Bnormal_plasma`` if provided,
        or default to ``jnp.linspace(0, 1, 34, endpoint=False)`` otherwise.
    Bnormal_plasma : ndarray, shape (nphi_plasma, ntheta_plasma), optional, default=None
        The external normal magnetic field on the plasma surface (e.g. from a fixed toroidal current).
        Will be treated as zero if not provided.
    plasma_coil_distance : float, optional, default=None
        The coil-plasma distance used to auto-generate the winding surface.
        Must be provided if ``winding_dofs`` is not.
    surface_type : str, optional, default='SurfaceRZFourier'
        (Static) The surface type string (reserved for future use).
    winding_dofs : ndarray, shape (ndof_winding,), optional, default=None
        The winding surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()``
        convention. Will be generated from ``plasma_coil_distance`` if not provided.
    winding_mpol : int, optional, default=6
        (Static) The number of poloidal Fourier harmonics in the winding surface.
    winding_ntor : int, optional, default=5
        (Static) The number of toroidal Fourier harmonics in the winding surface.
    winding_quadpoints_phi : ndarray, shape (nphi_winding,), optional, default=None
        Will be set to ``jnp.linspace(0, 1, 32*nfp, endpoint=False)`` by default.
    winding_quadpoints_theta : ndarray, shape (ntheta_winding,), optional, default=None
        Will be set to ``jnp.linspace(0, 1, 34, endpoint=False)`` by default.
    winding_stellsym : bool, optional, default=True
        (Static) Whether the winding surface has stellarator symmetry.

    Returns
    -------
    qp : QuadcoilParams
        The object storing the plasma and winding surfaces.
    dofs : {'phi': ndarray}, shape (ndofs,)
        The current potential coefficients minimizing :math:`\sum |B_n|^2`.
    '''
    # ----- Default parameters -----
    (
        plasma_quadpoints_phi, plasma_quadpoints_theta,
        winding_quadpoints_phi, winding_quadpoints_theta,
        quadpoints_phi, quadpoints_theta,
    ) = _resolve_quadpoints(
        nfp=nfp,
        Bnormal_plasma=Bnormal_plasma,
        plasma_quadpoints_phi=plasma_quadpoints_phi,
        plasma_quadpoints_theta=plasma_quadpoints_theta,
        winding_quadpoints_phi=winding_quadpoints_phi,
        winding_quadpoints_theta=winding_quadpoints_theta,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        plasma_coil_distance=plasma_coil_distance,
        winding_dofs=winding_dofs,
    )

    plasma_surface = SurfaceRZFourierJAX(
        nfp=nfp, stellsym=plasma_stellsym, 
        mpol=plasma_mpol, ntor=plasma_ntor, 
        quadpoints_phi=plasma_quadpoints_phi, 
        quadpoints_theta=plasma_quadpoints_theta,
        dofs=plasma_dofs
    )
    # winding surface is provided. 
    # Its dofs will be among x.
    if plasma_coil_distance is None:
        winding_surface = SurfaceRZFourierJAX(
            nfp=nfp, stellsym=winding_stellsym, 
            mpol=winding_mpol, ntor=winding_ntor, 
            quadpoints_phi=winding_quadpoints_phi, 
            quadpoints_theta=winding_quadpoints_theta,
            dofs=winding_dofs
        )
    # winding surface is not provided; auto-generate from plasma_coil_distance.
    else:
        winding_surface = plasma_surface.gen_winding_surface(
            d_expand=plasma_coil_distance,
            mpol=winding_mpol,
            ntor=winding_ntor,
            phi_interp=winding_phi_interp,
            theta_interp=winding_theta_interp,
            theta_rule_subsample=winding_theta_rule_subsample,
            quadpoints_phi=winding_quadpoints_phi,
            quadpoints_theta=winding_quadpoints_theta,
            lam_tikhonov=winding_lam_tikhonov,
            winding_surface_mode=winding_surface_mode,
            theta_mode=winding_theta_mode,
        )
    if Bnormal_plasma is None:
        Bnormal_plasma_temp = jnp.zeros((
            len(plasma_quadpoints_phi), 
            len(plasma_quadpoints_theta)
        ))
    else:
        Bnormal_plasma_temp = Bnormal_plasma
    
    qp_temp = QuadcoilParams(
        plasma_surface=plasma_surface, 
        winding_surface=winding_surface, 
        net_poloidal_current_amperes=net_poloidal_current_amperes, 
        net_toroidal_current_amperes=net_toroidal_current_amperes,
        Bnormal_plasma=Bnormal_plasma_temp,
        mpol=mpol, 
        ntor=ntor, 
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta, 
        stellsym=stellsym,
    )
    phi = qp_nescoil(qp_temp)
    return qp_temp, {'phi': phi}