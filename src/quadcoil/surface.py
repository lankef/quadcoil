from functools import partial, lru_cache
from jax import jit, tree_util, grad, jacfwd, vmap
from jax.scipy.special import factorial
from math import comb
from .math_utils import norm_helper, is_ndarray
import numpy as np
import jax.numpy as jnp
import sys

class SurfaceJAX:
    """Abstract base class for JAX-native toroidal surfaces.

    Subclasses must implement :meth:`gammadash` and register themselves as JAX
    pytrees.  All geometric quantities (normals, curvatures, etc.) are derived
    from ``gammadash`` and defined here so that every concrete surface type
    shares the same interface without code duplication.

    Attributes
    ----------
    quadpoints_phi, quadpoints_theta : jnp.ndarray, shape (nphi,) / (ntheta,)
        Quadrature grid in [0, 1).
    phi_mesh, theta_mesh : jnp.ndarray, shape (nphi, ntheta)
        Meshgrid counterparts (phi varies along axis-0).
    dphi, dtheta : float
        Grid spacings.
    """

    def __init__(self, nfp: int, stellsym: bool, mpol: int, ntor: int,
                 quadpoints_phi: jnp.ndarray, quadpoints_theta: jnp.ndarray,
                 dofs: jnp.ndarray):
        if not is_ndarray(quadpoints_phi, 1):
            raise TypeError(
                'quadpoints_phi has incorrect type or shape: '
                + str(type(quadpoints_phi))
            )
        if not is_ndarray(quadpoints_theta, 1):
            raise TypeError(
                'quadpoints_theta has incorrect type or shape: '
                + str(type(quadpoints_theta))
            )
        if not is_ndarray(dofs, 1):
            raise TypeError('dofs has incorrect type or shape: ' + str(type(dofs)))
        self.nfp = nfp
        self.stellsym = stellsym
        self.mpol = mpol
        self.ntor = ntor
        self.dofs = dofs
        self.quadpoints_phi = quadpoints_phi
        self.quadpoints_theta = quadpoints_theta
        self.theta_mesh, self.phi_mesh = jnp.meshgrid(quadpoints_theta, quadpoints_phi)
        self.dphi = quadpoints_phi[1] - quadpoints_phi[0]
        self.dtheta = quadpoints_theta[1] - quadpoints_theta[0]

    # ------------------------------------------------------------------
    # Concrete methods built on the abstract interface
    # ------------------------------------------------------------------

    @classmethod
    def dof_to_gamma(cls, dofs, phi_grid, theta_grid, nfp, stellsym,
                     dash1_order=0, dash2_order=0,
                     mpol: int = 10, ntor: int = 10):
        """Map DOF vector to gamma (or derivatives) on the quadrature grid."""
        return cls._dof_to_gamma_op(
            phi_grid=phi_grid,
            theta_grid=theta_grid,
            nfp=nfp,
            stellsym=stellsym,
            dash1_order=dash1_order,
            dash2_order=dash2_order,
            mpol=mpol,
            ntor=ntor,
        ) @ dofs

    @partial(jit, static_argnames=['a', 'b'])
    def gammadash(self, a: int, b: int) -> jnp.ndarray:
        """Surface position or mixed partial derivative.

        Parameters
        ----------
        a : int
            Order of the phi derivative (0, 1, or 2).
        b : int
            Order of the theta derivative (0, 1, or 2).

        Returns
        -------
        jnp.ndarray, shape (nphi, ntheta, 3)
            The quantity ``d^(a+b) gamma / d phi^a d theta^b`` evaluated on
            the quadrature grid.  Derivatives are with respect to the
            *normalised* angles in [0, 1).

        Notes
        -----
        Forwards to :meth:`gammadash_at_point` with the *separable*
        meshgrid form ``phi=quadpoints_phi[:, None]``,
        ``theta=quadpoints_theta[None, :]``.  This is strictly a faster
        backend than the historical ``_dof_to_gamma_op @ dofs`` path
        (5x-17x for RZ/XYZ Fourier surfaces, 30x-40x for the separable
        XYZ tensor Fourier surface) and produces output that matches the
        operator approach to round-off.  The operator
        (``cls._dof_to_gamma_op``) remains available and is still used
        internally by ``_fit_dofs_from_gamma``.
        """
        return self.gammadash_at_point(
            self.quadpoints_phi[:, None],
            self.quadpoints_theta[None, :],
            a, b,
        )

    @partial(jit, static_argnames=['a', 'b'])
    def gammadash_at_point(self, phi, theta, a: int, b: int) -> jnp.ndarray:
        """Broadcastable evaluation of ``d^(a+b) gamma / dphi^a dtheta^b``.

        Unlike :meth:`gammadash`, this never materialises the
        ``(nphi, ntheta, 3, ndof)`` operator: it computes the basis
        functions at the requested ``(phi, theta)`` points and contracts
        directly against ``self.dofs``.

        Parameters
        ----------
        phi, theta : jnp.ndarray
            Broadcast-compatible arrays of normalised angles in ``[0, 1)``.
        a : int
            Order of the phi derivative (0, 1, 2, or 3).
        b : int
            Order of the theta derivative (0, 1, 2, or 3).

        Returns
        -------
        jnp.ndarray, shape ``broadcast(phi, theta).shape + (3,)``
            The mixed partial of gamma at each requested point.

        Notes
        -----
        Reproduces ``self.gammadash(a, b)`` bit-for-bit when called with
        the fully expanded meshgrid
        ``phi = quadpoints_phi[:, None] + 0 * quadpoints_theta[None, :]``,
        ``theta = quadpoints_theta[None, :] + 0 * quadpoints_phi[:, None]``.

        Subclasses must override this method.
        """
        raise NotImplementedError(
            "gammadash_at_point() is not implemented for "
            f"{type(self).__name__}."
        )

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    gamma           = lambda self: self.gammadash(0, 0)
    gammadash1      = lambda self: self.gammadash(1, 0)
    gammadash2      = lambda self: self.gammadash(0, 1)
    gammadash1dash1 = lambda self: self.gammadash(2, 0)
    gammadash1dash2 = lambda self: self.gammadash(1, 1)
    gammadash2dash2 = lambda self: self.gammadash(0, 2)

    gamma_at_point           = lambda self, phi, theta: self.gammadash_at_point(phi, theta, 0, 0)
    gammadash1_at_point      = lambda self, phi, theta: self.gammadash_at_point(phi, theta, 1, 0)
    gammadash2_at_point      = lambda self, phi, theta: self.gammadash_at_point(phi, theta, 0, 1)
    gammadash1dash1_at_point = lambda self, phi, theta: self.gammadash_at_point(phi, theta, 2, 0)
    gammadash1dash2_at_point = lambda self, phi, theta: self.gammadash_at_point(phi, theta, 1, 1)
    gammadash2dash2_at_point = lambda self, phi, theta: self.gammadash_at_point(phi, theta, 0, 2)

    # ------------------------------------------------------------------
    # Geometric quantities
    # ------------------------------------------------------------------

    @jit
    def normal(self):
        dg1 = self.gammadash1()
        dg2 = self.gammadash2()
        return jnp.cross(dg1, dg2, axis=-1)

    @jit
    def unitnormal(self):
        normal = self.normal()
        return normal / jnp.linalg.norm(normal, axis=-1)[:, :, None]

    @jit
    def unitnormaldash_legacy(self):
        """d(unitnormal)/dphi and d(unitnormal)/dtheta (legacy hard-coded implementation).
        
        This is the original hard-coded gradient implementation, preserved for
        performance comparison. For production code, use unitnormaldash(a, b) instead.

        Returns
        -------
        (unitnormaldash1, unitnormaldash2), each (nphi, ntheta, 3)
        """
        normal = self.normal()
        dg1 = self.gammadash1()
        dg2 = self.gammadash2()
        dg12 = self.gammadash1dash2()
        dg22 = self.gammadash2dash2()
        _, inv_normN = norm_helper(normal)
        dg1_inv_n_dash1, dg1_inv_n_dash2, _, _ = self.dga_inv_n_dashb()

        dg1_inv_n = dg1 * inv_normN[:, :, None]
        unitnormaldash1 = (
            jnp.cross(dg1_inv_n_dash1, dg2, axis=-1)
            + jnp.cross(dg1_inv_n, dg12, axis=-1)
        )
        unitnormaldash2 = (
            jnp.cross(dg1_inv_n_dash2, dg2, axis=-1)
            + jnp.cross(dg1_inv_n, dg22, axis=-1)
        )
        return unitnormaldash1, unitnormaldash2

    @partial(jit, static_argnames=['a', 'b'])
    def unitnormaldash_at_point(self, phi, theta, a: int, b: int) -> jnp.ndarray:
        """Broadcastable mixed derivative of the unit normal at arbitrary points.

        Built on top of :meth:`gammadash_at_point` and JAX forward-mode
        autodiff (``vmap`` of repeated ``jacfwd`` over scalar phi/theta).
        Output shape: ``broadcast(phi, theta).shape + (3,)``.

        Reproduces ``self.unitnormaldash(a, b)`` when called with the
        fully-expanded meshgrid ``phi=quadpoints_phi[:, None] + 0*theta_1d``,
        ``theta=quadpoints_theta[None, :] + 0*phi_1d``.

        Parameters
        ----------
        phi, theta : jnp.ndarray
            Broadcast-compatible normalised angles in ``[0, 1)``.
        a, b : int
            Order of the phi / theta derivative.

        Notes
        -----
        For ``(a, b) == (0, 0)`` we take an analytic fast path that
        evaluates ``cross(gammadash1_at_point, gammadash2_at_point) / norm``
        directly on the broadcasted arrays (no autodiff, no vmap). For all
        other orders we vmap a per-point scalar function and apply
        ``jacfwd`` ``a + b`` times.
        """
        if a == 0 and b == 0:
            g1 = self.gammadash_at_point(phi, theta, 1, 0)
            g2 = self.gammadash_at_point(phi, theta, 0, 1)
            n = jnp.cross(g1, g2, axis=-1)
            return n / jnp.linalg.norm(n, axis=-1, keepdims=True)

        def n_at_point(phi_s, theta_s):
            g1 = self.gammadash_at_point(phi_s, theta_s, 1, 0)
            g2 = self.gammadash_at_point(phi_s, theta_s, 0, 1)
            n = jnp.cross(g1, g2)
            return n / jnp.linalg.norm(n)

        deriv_fn = n_at_point
        for _ in range(a):
            deriv_fn = jacfwd(deriv_fn, argnums=0)
        for _ in range(b):
            deriv_fn = jacfwd(deriv_fn, argnums=1)

        phi_b, theta_b = jnp.broadcast_arrays(phi, theta)
        flat_phi = phi_b.ravel()
        flat_theta = theta_b.ravel()
        result = vmap(deriv_fn)(flat_phi, flat_theta)        # (N, 3)
        return result.reshape(phi_b.shape + (3,))

    def unitnormal_at_point(self, phi, theta) -> jnp.ndarray:
        """Convenience: ``unitnormaldash_at_point(phi, theta, 0, 0)``."""
        return self.unitnormaldash_at_point(phi, theta, 0, 0)

    @partial(jit, static_argnames=['a', 'b'])
    def unitnormaldash(self, a: int, b: int) -> jnp.ndarray:
        """Compute d^(a+b)(unitnormal) / dphi^a dtheta^b using autodiff.
        
        Uses nested automatic differentiation for arbitrary-order derivatives.
        
        Parameters
        ----------
        a : int
            Order of derivative with respect to phi
        b : int
            Order of derivative with respect to theta
            
        Returns
        -------
        jnp.ndarray, shape (nphi, ntheta, 3)
            The derivative d^(a+b)(unitnormal) / dphi^a dtheta^b
            
        Examples
        --------
        >>> surf.unitnormaldash(0, 0)  # Returns unitnormal
        >>> surf.unitnormaldash(1, 0)  # Returns d(unitnormal)/dphi
        >>> surf.unitnormaldash(0, 1)  # Returns d(unitnormal)/dtheta
        >>> surf.unitnormaldash(2, 0)  # Returns d²(unitnormal)/dphi²
        """
        if a == 0 and b == 0:
            return self.unitnormal()
        
        # General autodiff implementation for all derivatives
        def shifted_unitnormal(dphi, dtheta):
            shifted_surface = self.copy_and_set_quadpoints(
                self.quadpoints_phi + dphi,
                self.quadpoints_theta + dtheta
            )
            return shifted_surface.unitnormal()
        
        # Build up derivatives by composing jacfwd
        result_fn = shifted_unitnormal
        
        # Apply 'a' derivatives with respect to dphi (argnums=0)
        for _ in range(a):
            result_fn = jacfwd(result_fn, argnums=0)
        
        # Apply 'b' derivatives with respect to dtheta (argnums=1)
        for _ in range(b):
            result_fn = jacfwd(result_fn, argnums=1)
        
        # Evaluate at dphi=0, dtheta=0
        return result_fn(0.0, 0.0)

    @jit
    def first_fund_form(self):
        """First fundamental form [E, F, G], shape (nphi, ntheta, 3)."""
        dg1 = self.gammadash1()
        dg2 = self.gammadash2()
        E = jnp.sum(dg1 * dg1, axis=-1)
        F = jnp.sum(dg1 * dg2, axis=-1)
        G = jnp.sum(dg2 * dg2, axis=-1)
        return jnp.stack([E, F, G], axis=-1)

    @jit
    def second_fund_form(self):
        """Second fundamental form [e, f, g], shape (nphi, ntheta, 3)."""
        un  = self.unitnormal()
        d11 = self.gammadash1dash1()
        d12 = self.gammadash1dash2()
        d22 = self.gammadash2dash2()
        e = jnp.sum(un * d11, axis=-1)
        f = jnp.sum(un * d12, axis=-1)
        g = jnp.sum(un * d22, axis=-1)
        return jnp.stack([e, f, g], axis=-1)

    @jit
    def surface_curvatures(self):
        """Mean (H), Gaussian (K), and principal (κ₁, κ₂) curvatures.

        Returns
        -------
        jnp.ndarray, shape (nphi, ntheta, 4)
            Stacked [H, K, kappa1, kappa2].
        """
        first  = self.first_fund_form()
        second = self.second_fund_form()
        E, F, G = first[..., 0], first[..., 1], first[..., 2]
        e, f, g = second[..., 0], second[..., 1], second[..., 2]
        det = E * G - F * F
        H = (e * G - 2 * F * f + g * E) / (2 * det)
        K = (e * g - f * f) / det
        disc = jnp.sqrt(H * H - K)
        return jnp.stack([H, K, H + disc, H - disc], axis=-1)

    @jit
    def da(self):
        """Area element: |N| * dphi * dtheta."""
        normN = jnp.linalg.norm(self.normal(), axis=-1)
        return self.dphi * self.dtheta * normN

    @jit
    def integrate(self, scalar_field):
        """Integrate a scalar field over the surface."""
        return jnp.sum(scalar_field * self.da())

    @jit
    def area(self):
        return jnp.sum(self.da())

    # ------------------------------------------------------------------
    # Helper functions for calculating quantities
    # ------------------------------------------------------------------
    
    @jit
    def grad_helper(self):
        """Contravariant vectors grad-phi and grad-theta.

        Returns
        -------
        (grad1, grad2) each of shape (nphi, ntheta, 3)
        """
        dg2 = self.gammadash2()
        dg1 = self.gammadash1()
        dg1xdg2 = jnp.cross(dg1, dg2, axis=-1)
        denom = jnp.sum(dg1xdg2 ** 2, axis=-1)
        grad1 = jnp.cross(dg2,  dg1xdg2, axis=-1) / denom[:, :, None]
        grad2 = jnp.cross(dg1, -dg1xdg2, axis=-1) / denom[:, :, None]
        return grad1, grad2

    @jit
    def dga_inv_n_dashb(self):
        """Derivatives of (1/|N|) * (dγ/dphi) and (1/|N|) * (dγ/dtheta).

        Returns
        -------
        (dg1_inv_n_dash1, dg1_inv_n_dash2,
         dg2_inv_n_dash1, dg2_inv_n_dash2)
            Each of shape (nphi, ntheta, 3).
        """
        normal = self.normal()
        dg1 = self.gammadash1()
        dg2 = self.gammadash2()
        dg11 = self.gammadash1dash1()
        dg12 = self.gammadash1dash2()
        dg22 = self.gammadash2dash2()

        normaldash1 = jnp.cross(dg11, dg2) + jnp.cross(dg1, dg12)
        normaldash2 = jnp.cross(dg12, dg2) + jnp.cross(dg1, dg22)

        _, inv_normN = norm_helper(normal)
        denominator = jnp.sum(normal ** 2, axis=-1) ** 1.5
        inv_normN_dash1 = -jnp.sum(normal * normaldash1, axis=-1) / denominator
        inv_normN_dash2 = -jnp.sum(normal * normaldash2, axis=-1) / denominator

        inv_n = inv_normN[:, :, None]
        inv_n_d1 = inv_normN_dash1[:, :, None]
        inv_n_d2 = inv_normN_dash2[:, :, None]

        dg1_inv_n_dash1 = dg11 * inv_n  + dg1 * inv_n_d1
        dg1_inv_n_dash2 = dg12 * inv_n  + dg1 * inv_n_d2
        dg2_inv_n_dash1 = dg12 * inv_n  + dg2 * inv_n_d1
        dg2_inv_n_dash2 = dg22 * inv_n  + dg2 * inv_n_d2
        return dg1_inv_n_dash1, dg1_inv_n_dash2, dg2_inv_n_dash1, dg2_inv_n_dash2

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_simsopt(cls, surface_simsopt):
        # Get the class name of the input surface and append "JAX"
        simsopt_class_name = type(surface_simsopt).__name__
        jax_class_name = simsopt_class_name + "JAX"

        # Look up the JAX class dynamically from the current module
        current_module = sys.modules[__name__]
        jax_cls = getattr(current_module, jax_class_name, None)

        if jax_cls is None:
            raise TypeError(
                f"No JAX equivalent found for '{simsopt_class_name}': "
                f"'{jax_class_name}' is not defined in this module."
            )

        # Delegate to the JAX class's own from_simsopt
        return jax_cls.from_simsopt(surface_simsopt)
        
    def get_dofs(self):
        return self.dofs.copy()

    def plot(self, **kwargs):
        try:
            self.to_simsopt().plot(**kwargs)
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Simsopt must be installed to use plot().')

    def copy_and_set_quadpoints(self, quadpoints_phi, quadpoints_theta):
        return type(self)(
            nfp=self.nfp,
            stellsym=self.stellsym,
            mpol=self.mpol,
            ntor=self.ntor,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            dofs=self.dofs,
        )

    # ------------------------------------------------------------------
    # Winding surface generators
    # ------------------------------------------------------------------

    def uniform_offset(
        self, d_expand: float,
        quadpoints_phi=None,
        quadpoints_theta=None,
    ):
        return SurfaceOffsetJAX(
            base_surface=self, 
            d_expand=d_expand,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )

    # ------------------------------------------------------------------
    # Winding surface helpers
    # ------------------------------------------------------------------

    @classmethod
    @partial(jit, static_argnames=['cls', 'nfp', 'stellsym', 'mpol', 'ntor'])
    def _fit_dofs_from_gamma(
            cls,
            phi_target, theta_target,
            gamma_target,
            nfp: int, stellsym: bool,
            mpol: int = 5, ntor: int = 5,
            lam_tikhonov=0.,
            custom_weight=None
    ):
        """Fit surface DOFs to sampled gamma points.

        Calls :meth:`_build_surface_fit_matrices` (subclass-specific) to
        obtain the operator and target, then solves the weighted
        least-squares problem with optional Tikhonov regularization.

        Parameters
        ----------
        phi_target : array, shape (nphi, ntheta)
            Target phi coordinates (normalized to [0, 1]).
        theta_target : array, shape (nphi, ntheta)
            Target theta coordinates (normalized to [0, 1]).
        gamma_target : array, shape (nphi, ntheta, 3)
            Target surface positions in Cartesian coordinates [x, y, z].
        nfp : int
            Number of field periods.
        stellsym : bool
            Stellarator symmetry flag.
        mpol : int, optional
            Maximum poloidal mode number.
        ntor : int, optional
            Maximum toroidal mode number.
        lam_tikhonov : float, optional
            Tikhonov regularization parameter for higher harmonics.
        custom_weight : array, shape (nphi, ntheta), optional
            Custom weights for fitting points.

        Returns
        -------
        dofs : array
            Fitted DOF vector for this surface type.
        """
        from .math_utils import safe_linear_solve
        A_lstsq, b_lstsq, m_2_n_2 = cls._build_surface_fit_matrices(
            phi_target, theta_target, gamma_target,
            nfp, stellsym, mpol, ntor,
        )
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
        lam = lam_tikhonov * jnp.diag(m_2_n_2)
        return safe_linear_solve(
            A=A_lstsq.T.dot(A_lstsq) + lam,
            b=A_lstsq.T.dot(b_lstsq),
        )

    @staticmethod
    def _dof_to_gamma_op(phi_grid, theta_grid, nfp, stellsym,
                        dash1_order=0, dash2_order=0,
                        mpol: int = 10, ntor: int = 10):
        """Operator mapping DOFs to gamma (or derivatives) on the grid.

        Returns an array of shape ``(nphi, ntheta, 3, ndof)`` such that
        ``op @ dofs`` gives the surface position (or derivative).

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def _build_surface_fit_matrices(
            phi_target, theta_target, gamma_target,
            nfp: int, stellsym: bool,
            mpol: int = 5, ntor: int = 5):
        """Build the least-squares matrices for surface fitting.

        Must be implemented by subclasses.

        Returns
        -------
        A_lstsq : array, shape (nphi, ntheta, k, ndof)
            The linear operator mapping DOFs to the fitting target.
        b_lstsq : array, shape (nphi, ntheta, k)
            The target vector.
        m_2_n_2 : array, shape (ndof,)
            Mode-number weights ``m^2 + n^2`` for Tikhonov regularization.
        """
        raise NotImplementedError
    
# ======================================================================
# SurfaceRZFourierJAX
# ======================================================================

@tree_util.register_pytree_node_class
class SurfaceRZFourierJAX(SurfaceJAX):
    """JAX-native surface in cylindrical Fourier (RZ) coordinates.

    Representation::

        r(phi, theta) = sum_{m,n} [rc_{mn} cos(m*theta - nfp*n*phi)
                                  + rs_{mn} sin(m*theta - nfp*n*phi)]
        z(phi, theta) = sum_{m,n} [zc_{mn} cos(m*theta - nfp*n*phi)
                                  + zs_{mn} sin(m*theta - nfp*n*phi)]

    The DOF vector is ``[rc, zs]`` for stellarator-symmetric surfaces and
    ``[rc, rs, zc, zs]`` otherwise, matching simsopt's convention exactly.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def gen_offset_dofs(self, d_expand,
            mpol=5, ntor=5, smoothing='intersection',
            pol_interp=2, tor_interp=2,
            lam_tikhonov=1e-5):
        from .winding_surface import gen_winding_surface_offset, gen_winding_surface_arc
        if smoothing == 'none':
            return gen_winding_surface_offset(
                plasma_gamma=self.gamma(),
                d_expand=d_expand,
                nfp=self.nfp,
                stellsym=self.stellsym,
                mpol=mpol,
                ntor=ntor,
            )
        elif smoothing in ('intersection', 'hull'):
            rule = 'self-intersection' if smoothing == 'intersection' else 'hull'
            return gen_winding_surface_arc(
                plasma_gamma=self.gamma(),
                d_expand=d_expand,
                nfp=self.nfp,
                stellsym=self.stellsym,
                mpol=mpol,
                ntor=ntor,
                pol_interp=pol_interp,
                tor_interp=tor_interp,
                lam_tikhonov=lam_tikhonov,
                rule=rule,
            )
        else:
            raise ValueError(
                "smoothing must be 'none', 'intersection', or 'hull'. "
                "Got: " + repr(smoothing)
            )

    def from_simsopt(simsopt_surf):
        return SurfaceRZFourierJAX(
            nfp=simsopt_surf.nfp,
            stellsym=simsopt_surf.stellsym,
            mpol=simsopt_surf.mpol,
            ntor=simsopt_surf.ntor,
            quadpoints_phi=jnp.array(simsopt_surf.quadpoints_phi),
            quadpoints_theta=jnp.array(simsopt_surf.quadpoints_theta),
            dofs=jnp.array(simsopt_surf.get_dofs()),
        )

    def to_simsopt(self):
        try:
            from simsopt.geo import SurfaceRZFourier
        except ImportError:
            raise ModuleNotFoundError(
                'Simsopt must be installed to export surface with to_simsopt().'
            )
        surf = SurfaceRZFourier(
            nfp=self.nfp,
            stellsym=self.stellsym,
            mpol=self.mpol,
            ntor=self.ntor,
            quadpoints_phi=np.array(self.quadpoints_phi),
            quadpoints_theta=np.array(self.quadpoints_theta),
        )
        surf.set_dofs(np.array(self.dofs))
        return surf

    def from_desc(desc_surf, quadpoints_phi, quadpoints_theta):
        try:
            from desc.vmec_utils import ptolemy_identity_rev
        except ImportError:
            raise ModuleNotFoundError('DESC must be installed to load surface from DESC.')
        mm, nn, rs_raw, rc_raw = ptolemy_identity_rev(
            desc_surf.R_basis.modes[:, 1],
            desc_surf.R_basis.modes[:, 2],
            desc_surf.R_lmn,
        )
        mm, nn, zs_raw, zc_raw = ptolemy_identity_rev(
            desc_surf.Z_basis.modes[:, 1],
            desc_surf.Z_basis.modes[:, 2],
            desc_surf.Z_lmn,
        )
        mpol = desc_surf.M
        ntor = desc_surf.N
        stellsym = desc_surf.sym
        nfp = desc_surf.NFP
        rc = rc_raw.flatten()
        rs = rs_raw.flatten()[1:]
        zc = zc_raw.flatten()
        zs = zs_raw.flatten()[1:]
        if stellsym:
            dofs = jnp.concatenate([rc, zs])
        else:
            dofs = jnp.concatenate([rc, rs, zc, zs])
        return SurfaceRZFourierJAX(
            nfp=nfp,
            stellsym=stellsym,
            mpol=mpol,
            ntor=ntor,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            dofs=dofs,
        )

    def to_desc(self):
        try:
            from desc.vmec_utils import ptolemy_identity_fwd
            from desc.geometry import FourierRZToroidalSurface
        except ImportError:
            raise ModuleNotFoundError('DESC must be installed to export surface to DESC.')
        if self.stellsym:
            len_sin = len(self.dofs) // 2
            rc = self.dofs[:-len_sin]
            zs = jnp.insert(self.dofs[-len_sin:], 0, 0.)
            zc = jnp.zeros_like(rc)
            rs = jnp.zeros_like(rc)
        else:
            half_len = len(self.dofs) // 2
            len_sin = half_len // 2
            rcrs = self.dofs[:half_len]
            zczs = self.dofs[half_len:]
            rc = rcrs[:-len_sin]
            rs = jnp.insert(rcrs[-len_sin:], 0, 0.)
            zc = zczs[:-len_sin]
            zs = jnp.insert(zczs[-len_sin:], 0, 0.)
        mc, _, nc, _ = make_rzfourier_mc_ms_nc_ns(self.mpol, self.ntor)
        Rm, Rn, R_lmn = ptolemy_identity_fwd(mc, nc, rs, rc)
        Zm, Zn, Z_lmn = ptolemy_identity_fwd(mc, nc, zs, zc)
        modes_R = jnp.vstack([Rm, Rn]).T
        modes_Z = jnp.vstack([Zm, Zn]).T
        return FourierRZToroidalSurface(
            R_lmn.flatten(), Z_lmn.flatten(),
            modes_R.astype(int), modes_Z.astype(int),
            NFP=self.nfp, sym=self.stellsym,
            M=self.mpol, N=self.ntor, rho=1,
        )

    # ------------------------------------------------------------------
    # JAX pytree protocol
    # ------------------------------------------------------------------

    def tree_flatten(self):
        children = (
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.dofs,
            self.theta_mesh,
            self.phi_mesh,
            self.dphi,
            self.dtheta,
        )
        aux_data = {
            'nfp': self.nfp,
            'stellsym': self.stellsym,
            'mpol': self.mpol,
            'ntor': self.ntor,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            nfp=aux_data['nfp'],
            stellsym=aux_data['stellsym'],
            mpol=aux_data['mpol'],
            ntor=aux_data['ntor'],
            quadpoints_phi=children[0],
            quadpoints_theta=children[1],
            dofs=children[2],
        )

    # ------------------------------------------------------------------
    # Static methods for DOF operations
    # ------------------------------------------------------------------

    @staticmethod
    @partial(jit, static_argnames=['nfp', 'stellsym', 'dash1_order', 'dash2_order', 'mpol', 'ntor'])
    def dof_to_rz_op(
            phi_grid, theta_grid,
            nfp: int, stellsym: bool,
            dash1_order=0, dash2_order=0,
            mpol: int = 10, ntor: int = 10):
        """Operator mapping DOF vector -> (R, Z) on the quadrature grid."""
        mc, ms, nc, ns = make_rzfourier_mc_ms_nc_ns(mpol, ntor)
        total_neg = (dash1_order + dash2_order) // 2
        derivative_factor_c = (
            (-nc[:, None, None] * jnp.pi * 2 * nfp) ** dash1_order
            * (mc[:, None, None] * jnp.pi * 2) ** dash2_order
        ) * (-1) ** total_neg
        derivative_factor_s = (
            (-ns[:, None, None] * jnp.pi * 2 * nfp) ** dash1_order
            * (ms[:, None, None] * jnp.pi * 2) ** dash2_order
        ) * (-1) ** total_neg
        if (dash1_order + dash2_order) % 2 == 0:
            cmn = derivative_factor_c * jnp.cos(
                mc[:, None, None] * jnp.pi * 2 * theta_grid[None, :, :]
                - nc[:, None, None] * jnp.pi * 2 * nfp * phi_grid[None, :, :]
            )
            smn = derivative_factor_s * jnp.sin(
                ms[:, None, None] * jnp.pi * 2 * theta_grid[None, :, :]
                - ns[:, None, None] * jnp.pi * 2 * nfp * phi_grid[None, :, :]
            )
        else:
            cmn = -derivative_factor_c * jnp.sin(
                mc[:, None, None] * theta_grid[None, :, :] * jnp.pi * 2
                - nc[:, None, None] * phi_grid[None, :, :] * jnp.pi * 2 * nfp
            )
            smn = derivative_factor_s * jnp.cos(
                ms[:, None, None] * theta_grid[None, :, :] * jnp.pi * 2
                - ns[:, None, None] * phi_grid[None, :, :] * jnp.pi * 2 * nfp
            )
        m_2_n_2 = jnp.concatenate([mc, ms]) ** 2 + jnp.concatenate([nc, ns]) ** 2
        if not stellsym:
            m_2_n_2 = jnp.tile(m_2_n_2, 2)
        if stellsym:
            r_operator = cmn
            z_operator = smn
        else:
            r_operator = jnp.concatenate([cmn, smn], axis=0)
            z_operator = jnp.concatenate([cmn, smn], axis=0)
        r_operator_padded = jnp.concatenate([r_operator, jnp.zeros_like(z_operator)], axis=0)
        z_operator_padded = jnp.concatenate([jnp.zeros_like(r_operator), z_operator], axis=0)
        A_lstsq = jnp.concatenate(
            [r_operator_padded[:, :, :, None], z_operator_padded[:, :, :, None]], axis=3
        )
        A_lstsq = jnp.moveaxis(A_lstsq, 0, -1)
        return A_lstsq, m_2_n_2

    @staticmethod
    def _dof_to_gamma_op(
            phi_grid, theta_grid,
            nfp, stellsym,
            dash1_order=0, dash2_order=0,
            mpol: int = 10, ntor: int = 10):
        """Operator of shape (nphi, ntheta, 3, ndof) mapping dofs -> gamma."""
        dof_to_x = 0.
        dof_to_y = 0.
        for dash1_order_rz in range(dash1_order + 1):
            dash1_order_trig = dash1_order - dash1_order_rz
            dof_to_rz_dash, _ = SurfaceRZFourierJAX.dof_to_rz_op(
                phi_grid=phi_grid,
                theta_grid=theta_grid,
                nfp=nfp,
                stellsym=stellsym,
                dash1_order=dash1_order_rz,
                dash2_order=dash2_order,
                mpol=mpol,
                ntor=ntor,
            )
            dof_to_r_dash = dof_to_rz_dash[:, :, 0, :]
            if dash1_order_rz == dash1_order:
                dof_to_z = dof_to_rz_dash[:, :, 1, :]
            total_neg = dash1_order_trig // 2
            binomial_coef = (
                factorial(dash1_order)
                / factorial(dash1_order_rz)
                / factorial(dash1_order_trig)
            )
            derivative_factor = (
                binomial_coef * (-1) ** total_neg * (jnp.pi * 2) ** dash1_order_trig
            )
            if dash1_order_trig % 2 == 0:
                dof_to_x += derivative_factor * dof_to_r_dash * jnp.cos(phi_grid * jnp.pi * 2)[:, :, None]
                dof_to_y += derivative_factor * dof_to_r_dash * jnp.sin(phi_grid * jnp.pi * 2)[:, :, None]
            else:
                dof_to_x += -derivative_factor * dof_to_r_dash * jnp.sin(phi_grid * jnp.pi * 2)[:, :, None]
                dof_to_y +=  derivative_factor * dof_to_r_dash * jnp.cos(phi_grid * jnp.pi * 2)[:, :, None]
        return jnp.concatenate(
            [dof_to_x[:, :, None, :], dof_to_y[:, :, None, :], dof_to_z[:, :, None, :]], axis=2
        )

    @staticmethod
    @partial(jit, static_argnames=['nfp', 'stellsym', 'mpol', 'ntor'])
    def _build_surface_fit_matrices(
            phi_target, theta_target, gamma_target,
            nfp: int, stellsym: bool,
            mpol: int = 5, ntor: int = 5):
        r_fit = jnp.sqrt(gamma_target[:, :, 0]**2 + gamma_target[:, :, 1]**2)
        z_fit = gamma_target[:, :, 2]
        A_lstsq, m_2_n_2 = SurfaceRZFourierJAX.dof_to_rz_op(
            theta_grid=theta_target,
            phi_grid=phi_target,
            nfp=nfp,
            stellsym=stellsym,
            mpol=mpol,
            ntor=ntor,
        )
        b_lstsq = jnp.concatenate([r_fit[:, :, None], z_fit[:, :, None]], axis=2)
        return A_lstsq, b_lstsq, m_2_n_2

    # ------------------------------------------------------------------
    # Broadcastable evaluator
    # ------------------------------------------------------------------

    @partial(jit, static_argnames=['a', 'b'])
    def gammadash_at_point(self, phi, theta, a: int, b: int) -> jnp.ndarray:
        """Direct broadcastable evaluation of d^(a+b) gamma / dphi^a dtheta^b.

        Computes the cos/sin mode tables at the requested (phi, theta) and
        contracts them directly against the rc/rs/zc/zs slices of
        ``self.dofs``, then applies the Leibniz rule to rotate (R, Z) into
        Cartesian (x, y, z).  This avoids the (nphi, ntheta, 3, ndof) operator
        used by ``gammadash``.

        Reproduces ``self.gammadash(a, b)`` bit-for-bit when called with the
        fully expanded meshgrid ``phi=quadpoints_phi[:, None] + 0*theta_1d``,
        ``theta=quadpoints_theta[None, :] + 0*phi_1d``.
        """
        nfp = self.nfp
        stellsym = self.stellsym
        mpol = self.mpol
        ntor = self.ntor
        dofs = self.dofs

        mc, ms, nc, ns = make_rzfourier_mc_ms_nc_ns(mpol, ntor)
        n_c = mc.shape[0]
        n_s = ms.shape[0]

        # Slice DOFs to match the layout used in ``dof_to_rz_op``.
        if stellsym:
            rc = dofs[:n_c]
            zs = dofs[n_c:]
            rs_use = None
            zc_use = None
        else:
            rc = dofs[:n_c]
            rs_use = dofs[n_c:n_c + n_s]
            zc_use = dofs[n_c + n_s:n_c + n_s + n_c]
            zs = dofs[n_c + n_s + n_c:]

        pi2 = 2.0 * jnp.pi
        pi2nfp = pi2 * nfp
        phi_e = phi[..., None]      # broadcast_shape + (1,)
        theta_e = theta[..., None]  # broadcast_shape + (1,)

        def compute_rz(k_phi, k_theta):
            """(R, Z) for derivative orders (k_phi, k_theta)."""
            ang_c = mc * pi2 * theta_e - nc * pi2nfp * phi_e
            ang_s = ms * pi2 * theta_e - ns * pi2nfp * phi_e

            total_neg = (k_phi + k_theta) // 2
            sign = (-1) ** total_neg
            fac_c = sign * (-nc * pi2nfp) ** k_phi * (mc * pi2) ** k_theta
            fac_s = sign * (-ns * pi2nfp) ** k_phi * (ms * pi2) ** k_theta

            if (k_phi + k_theta) % 2 == 0:
                basis_c = fac_c * jnp.cos(ang_c)
                basis_s = fac_s * jnp.sin(ang_s)
            else:
                basis_c = -fac_c * jnp.sin(ang_c)
                basis_s = fac_s * jnp.cos(ang_s)

            if stellsym:
                R = basis_c @ rc
                Z = basis_s @ zs
            else:
                R = basis_c @ rc + basis_s @ rs_use
                Z = basis_c @ zc_use + basis_s @ zs
            return R, Z

        cosphi = jnp.cos(pi2 * phi)
        sinphi = jnp.sin(pi2 * phi)

        dof_to_x = 0.0
        dof_to_y = 0.0
        Z_final = None
        for k in range(a + 1):
            a_trig = a - k
            R_k, Z_k = compute_rz(k, b)
            if k == a:
                Z_final = Z_k
            binomial_coef = comb(a, k)
            total_neg = a_trig // 2
            derivative_factor = binomial_coef * (-1) ** total_neg * pi2 ** a_trig
            if a_trig % 2 == 0:
                dof_to_x = dof_to_x + derivative_factor * R_k * cosphi
                dof_to_y = dof_to_y + derivative_factor * R_k * sinphi
            else:
                dof_to_x = dof_to_x - derivative_factor * R_k * sinphi
                dof_to_y = dof_to_y + derivative_factor * R_k * cosphi

        return jnp.stack([dof_to_x, dof_to_y, Z_final], axis=-1)


# ======================================================================
# SurfaceXYZTensorFourierJAX
# ======================================================================

@tree_util.register_pytree_node_class
class SurfaceXYZTensorFourierJAX(SurfaceJAX):
    r"""JAX-native surface in Cartesian tensor-product Fourier coordinates.

    Matches :class:`simsopt.geo.SurfaceXYZTensorFourier` exactly.

    Representation::

        x_hat(theta, phi) = sum_{i,j} x_{ij} w_i(theta) v_j(phi)
        y_hat(theta, phi) = sum_{i,j} y_{ij} w_i(theta) v_j(phi)
        x(phi, theta) = x_hat * cos(phi_rad) - y_hat * sin(phi_rad)
        y(phi, theta) = x_hat * sin(phi_rad) + y_hat * cos(phi_rad)
        z(theta, phi) = sum_{i,j} z_{ij} w_i(theta) v_j(phi)

    where ``phi_rad = 2*pi*phi_normalised``, and the toroidal basis is::

        v_j : j=0..ntor  -> cos(nfp*j*phi_rad)
              j=ntor+1..2*ntor -> sin(nfp*(j-ntor)*phi_rad)

    and the poloidal basis is::

        w_i : i=0..mpol  -> cos(i*theta_rad)
              i=mpol+1..2*mpol -> sin((i-mpol)*theta_rad)

    The DOF vector is ``[x_active, y_active, z_active]`` where the active
    coefficients follow simsopt's ``get_dofs()`` ordering (row-major over
    (m, n), skipping stellarator-symmetric zeros).

    Stellarator symmetry rules
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    * **x**: keep ``(n <= ntor and m <= mpol)`` OR ``(n > ntor and m > mpol)``
    * **y, z**: keep ``(n <= ntor and m > mpol)`` OR ``(n > ntor and m <= mpol)``

    Parameters
    ----------
    nfp : int
    stellsym : bool
    mpol, ntor : int
    quadpoints_phi, quadpoints_theta : array-like 1-D, values in [0, 1)
    dofs : 1-D array
        Active Fourier coefficients in simsopt ordering.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def from_simsopt(simsopt_surf):
        """Load from a :class:`simsopt.geo.SurfaceXYZTensorFourier` instance."""
        return SurfaceXYZTensorFourierJAX(
            nfp=simsopt_surf.nfp,
            stellsym=simsopt_surf.stellsym,
            mpol=simsopt_surf.mpol,
            ntor=simsopt_surf.ntor,
            quadpoints_phi=jnp.array(simsopt_surf.quadpoints_phi),
            quadpoints_theta=jnp.array(simsopt_surf.quadpoints_theta),
            dofs=jnp.array(simsopt_surf.get_dofs()),
        )

    def to_simsopt(self):
        """Convert to :class:`simsopt.geo.SurfaceXYZTensorFourier`."""
        try:
            from simsopt.geo import SurfaceXYZTensorFourier
        except ImportError:
            raise ModuleNotFoundError(
                'Simsopt must be installed to export surface with to_simsopt().'
            )
        surf = SurfaceXYZTensorFourier(
            nfp=self.nfp,
            stellsym=self.stellsym,
            mpol=self.mpol,
            ntor=self.ntor,
            quadpoints_phi=np.array(self.quadpoints_phi),
            quadpoints_theta=np.array(self.quadpoints_theta),
        )
        surf.set_dofs(np.array(self.dofs))
        return surf

    def to_RZFourier(self):
        """Convert to :class:`SurfaceRZFourierJAX` via a least-squares fit."""
        simsopt_rz = self.to_simsopt().to_RZFourier()
        return SurfaceRZFourierJAX.from_simsopt(simsopt_rz)

    # ------------------------------------------------------------------
    # DOF utilities
    # ------------------------------------------------------------------

    def num_dofs(self):
        """Total number of active DOFs."""
        rx, cx, ry, cy, rz, cz = _xyztensor_active_indices(
            self.mpol, self.ntor, self.stellsym
        )
        return len(rx) + len(ry) + len(rz)

    # ------------------------------------------------------------------
    # JAX pytree protocol
    # ------------------------------------------------------------------

    def tree_flatten(self):
        children = (
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.dofs,
            self.theta_mesh,
            self.phi_mesh,
            self.dphi,
            self.dtheta,
        )
        aux_data = {
            'nfp': self.nfp,
            'stellsym': self.stellsym,
            'mpol': self.mpol,
            'ntor': self.ntor,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            nfp=aux_data['nfp'],
            stellsym=aux_data['stellsym'],
            mpol=aux_data['mpol'],
            ntor=aux_data['ntor'],
            quadpoints_phi=children[0],
            quadpoints_theta=children[1],
            dofs=children[2],
        )

    # ------------------------------------------------------------------
    # Static methods for DOF operations
    # ------------------------------------------------------------------

    @staticmethod
    def _dof_to_gamma_op(
            phi_grid, theta_grid,
            nfp: int, stellsym: bool,
            dash1_order=0, dash2_order=0,
            mpol: int = 5, ntor: int = 5):
        """Operator of shape (nphi, ntheta, 3, ndof) mapping dofs -> gamma.
        
        For XYZ tensor Fourier surfaces, returns the operator that maps
        active DOF vector to gamma evaluated on the grid.
        
        Note: This uses xyztensor_gammadash to build the operator by
        calling it with unit vectors for each DOF.
        """
        rows_x, cols_x, rows_y, cols_y, rows_z, cols_z = _xyztensor_active_indices(
            mpol, ntor, stellsym
        )
        ndof_x = len(rows_x)
        ndof_y = len(rows_y)
        ndof_z = len(rows_z)
        ndof_total = ndof_x + ndof_y + ndof_z

        quadpoints_phi = phi_grid[:, 0]      # (nphi,)
        quadpoints_theta = theta_grid[0, :]  # (ntheta,)
        nphi = quadpoints_phi.shape[0]
        ntheta = quadpoints_theta.shape[0]

        # ----------------------------------------------------------------
        # Vectorised construction of the (nphi, ntheta, 3, ndof) operator.
        #
        # For each active DOF (r, c), the corresponding column of the
        # operator is V_a[:, c] * W_b[:, r] (outer product), with Leibniz-
        # rule trig factors for the x and y channels:
        #
        #   d^a x / dphi^a = sum_k C(a,k) [xhat^(k) * D^(a-k)cos
        #                                  - yhat^(k) * D^(a-k)sin]
        #   d^a y / dphi^a = sum_k C(a,k) [xhat^(k) * D^(a-k)sin
        #                                  + yhat^(k) * D^(a-k)cos]
        #
        # where xhat^(k)[i, j, idx] = V_k[i, cols_x[idx]] * W_b[j, rows_x[idx]]
        # and similarly for yhat (over y DOF indices).  z is not rotated, so
        # z_op[i, j, idx] = V_a[i, cols_z[idx]] * W_b[j, rows_z[idx]].
        # ----------------------------------------------------------------
        Wb = _xyztensor_W(quadpoints_theta, mpol, dash2_order)            # (ntheta, 2*mpol+1)
        Vks = [
            _xyztensor_V(quadpoints_phi, ntor, nfp, k)
            for k in range(dash1_order + 1)
        ]                                                                  # each (nphi, 2*ntor+1)

        # Active gather indices (numpy ints -> static JAX gather).
        cols_x_j = jnp.asarray(cols_x)
        rows_x_j = jnp.asarray(rows_x)
        cols_y_j = jnp.asarray(cols_y)
        rows_y_j = jnp.asarray(rows_y)
        cols_z_j = jnp.asarray(cols_z)
        rows_z_j = jnp.asarray(rows_z)

        # Per-channel hat operators of shape (nphi, ntheta, ndof_*).
        # Va_x[k][i, idx] = Vk[i, cols_x[idx]];  Wb_x[j, idx] = Wb[j, rows_x[idx]]
        Wb_x = Wb[:, rows_x_j]               # (ntheta, ndof_x)
        Wb_y = Wb[:, rows_y_j]               # (ntheta, ndof_y)
        Wb_z = Wb[:, rows_z_j]               # (ntheta, ndof_z)

        xhat_ops = [Vk[:, cols_x_j][:, None, :] * Wb_x[None, :, :] for Vk in Vks]
        yhat_ops = [Vk[:, cols_y_j][:, None, :] * Wb_y[None, :, :] for Vk in Vks]
        z_op_z = Vks[dash1_order][:, cols_z_j][:, None, :] * Wb_z[None, :, :]

        # Derivatives of cos(phi_rad) and sin(phi_rad) w.r.t. phi_norm.
        pi2 = 2.0 * jnp.pi
        phi_r = pi2 * quadpoints_phi
        cosphi = jnp.cos(phi_r)[:, None, None]   # (nphi, 1, 1)
        sinphi = jnp.sin(phi_r)[:, None, None]

        def _dcos(k):
            r = k % 4
            if r == 0: return cosphi
            if r == 1: return -pi2 * sinphi
            if r == 2: return -(pi2 ** 2) * cosphi
            return (pi2 ** 3) * sinphi

        def _dsin(k):
            r = k % 4
            if r == 0: return sinphi
            if r == 1: return pi2 * cosphi
            if r == 2: return -(pi2 ** 2) * sinphi
            return -(pi2 ** 3) * cosphi

        # Leibniz combinations for x and y; sums are short (a+1 terms).
        x_op_xpart = sum(
            comb(dash1_order, k) * xhat_ops[k] * _dcos(dash1_order - k)
            for k in range(dash1_order + 1)
        )
        x_op_ypart = sum(
            comb(dash1_order, k) * yhat_ops[k] * (-_dsin(dash1_order - k))
            for k in range(dash1_order + 1)
        )
        y_op_xpart = sum(
            comb(dash1_order, k) * xhat_ops[k] * _dsin(dash1_order - k)
            for k in range(dash1_order + 1)
        )
        y_op_ypart = sum(
            comb(dash1_order, k) * yhat_ops[k] * _dcos(dash1_order - k)
            for k in range(dash1_order + 1)
        )

        zeros_x = jnp.zeros((nphi, ntheta, ndof_x))
        zeros_y = jnp.zeros((nphi, ntheta, ndof_y))
        zeros_z = jnp.zeros((nphi, ntheta, ndof_z))

        # Pack along DOF axis in the order [x_dofs, y_dofs, z_dofs].
        x_op = jnp.concatenate([x_op_xpart, x_op_ypart, zeros_z], axis=-1)
        y_op = jnp.concatenate([y_op_xpart, y_op_ypart, zeros_z], axis=-1)
        z_op = jnp.concatenate([zeros_x,    zeros_y,    z_op_z], axis=-1)

        operator = jnp.stack([x_op, y_op, z_op], axis=-2)  # (nphi, ntheta, 3, ndof)
        return operator

    @staticmethod
    @partial(jit, static_argnames=['nfp', 'stellsym', 'mpol', 'ntor'])
    def _build_surface_fit_matrices(
            phi_target, theta_target, gamma_target,
            nfp: int, stellsym: bool,
            mpol: int = 5, ntor: int = 5):
        A_lstsq = SurfaceXYZTensorFourierJAX._dof_to_gamma_op(
            phi_grid=phi_target,
            theta_grid=theta_target,
            nfp=nfp,
            stellsym=stellsym,
            mpol=mpol,
            ntor=ntor,
        )
        b_lstsq = gamma_target
        rows_x, cols_x, rows_y, cols_y, rows_z, cols_z = _xyztensor_active_indices(
            mpol, ntor, stellsym
        )
        m_x = jnp.array([i if i <= mpol else i - mpol - 1 for i in rows_x])
        n_x = jnp.array([j if j <= ntor else j - ntor - 1 for j in cols_x])
        m_y = jnp.array([i if i <= mpol else i - mpol - 1 for i in rows_y])
        n_y = jnp.array([j if j <= ntor else j - ntor - 1 for j in cols_y])
        m_z = jnp.array([i if i <= mpol else i - mpol - 1 for i in rows_z])
        n_z = jnp.array([j if j <= ntor else j - ntor - 1 for j in cols_z])
        m_2_n_2 = jnp.concatenate([
            m_x**2 + n_x**2,
            m_y**2 + n_y**2,
            m_z**2 + n_z**2,
        ])
        return A_lstsq, b_lstsq, m_2_n_2

    # ------------------------------------------------------------------
    # Broadcastable evaluator
    # ------------------------------------------------------------------

    @partial(jit, static_argnames=['a', 'b'])
    def gammadash_at_point(self, phi, theta, a: int, b: int) -> jnp.ndarray:
        """Direct broadcastable evaluation of d^(a+b) gamma / dphi^a dtheta^b.

        Reconstructs the full (2*mpol+1, 2*ntor+1) coefficient matrices,
        evaluates the poloidal/toroidal basis at the requested points, and
        contracts straight to ``(xhat, yhat, z)`` before applying the same
        Leibniz rotation as :func:`xyztensor_gammadash`.
        """
        nfp = self.nfp
        stellsym = self.stellsym
        mpol = self.mpol
        ntor = self.ntor
        dofs = self.dofs

        rows_x, cols_x, rows_y, cols_y, rows_z, cols_z = _xyztensor_active_indices(
            mpol, ntor, stellsym
        )
        ndof_x = len(rows_x)
        ndof_y = len(rows_y)

        x_dofs = dofs[:ndof_x]
        y_dofs = dofs[ndof_x:ndof_x + ndof_y]
        z_dofs = dofs[ndof_x + ndof_y:]

        shape = (2 * mpol + 1, 2 * ntor + 1)
        x_full = jnp.zeros(shape).at[rows_x, cols_x].set(x_dofs)
        y_full = jnp.zeros(shape).at[rows_y, cols_y].set(y_dofs)
        z_full = jnp.zeros(shape).at[rows_z, cols_z].set(z_dofs)

        Wb = _xyztensor_W_at_point(theta, mpol, b)              # S_theta + (2*mpol+1,)
        Vks = [_xyztensor_V_at_point(phi, ntor, nfp, k) for k in range(a + 1)]
                                                                # each S_phi + (2*ntor+1,)

        def hat(Vk_arr, M):
            """sum_{i, j} Wb[..., i] * Vk_arr[..., j] * M[i, j]."""
            VkMT = jnp.tensordot(Vk_arr, M, axes=[[-1], [1]])    # S_phi + (2*mpol+1,)
            return jnp.sum(Wb * VkMT, axis=-1)                    # broadcast S_theta with S_phi

        xhat_list = [hat(Vks[k], x_full) for k in range(a + 1)]
        yhat_list = [hat(Vks[k], y_full) for k in range(a + 1)]
        z_a = hat(Vks[a], z_full)

        pi2 = 2.0 * jnp.pi
        cosphi = jnp.cos(pi2 * phi)
        sinphi = jnp.sin(pi2 * phi)

        def _dcos(k):
            r = k % 4
            if r == 0: return cosphi
            if r == 1: return -pi2 * sinphi
            if r == 2: return -(pi2 ** 2) * cosphi
            return (pi2 ** 3) * sinphi

        def _dsin(k):
            r = k % 4
            if r == 0: return sinphi
            if r == 1: return pi2 * cosphi
            if r == 2: return -(pi2 ** 2) * sinphi
            return -(pi2 ** 3) * cosphi

        res_x = sum(
            comb(a, k) * (xhat_list[k] * _dcos(a - k) - yhat_list[k] * _dsin(a - k))
            for k in range(a + 1)
        )
        res_y = sum(
            comb(a, k) * (xhat_list[k] * _dsin(a - k) + yhat_list[k] * _dcos(a - k))
            for k in range(a + 1)
        )

        return jnp.stack([res_x, res_y, z_a], axis=-1)


# ======================================================================
# SurfaceXYZFourierJAX
# ======================================================================

@tree_util.register_pytree_node_class
class SurfaceXYZFourierJAX(SurfaceJAX):
    r"""JAX-native surface in Cartesian Fourier (XYZ) coordinates.

    Matches :class:`simsopt.geo.SurfaceXYZFourier` exactly.

    Representation::

        x_hat(phi, theta) = sum_{m,n} [xc_{mn} cos(m*theta - nfp*n*phi)
                                      + xs_{mn} sin(m*theta - nfp*n*phi)]
        y_hat(phi, theta) = sum_{m,n} [yc_{mn} cos(m*theta - nfp*n*phi)
                                      + ys_{mn} sin(m*theta - nfp*n*phi)]
        z(phi, theta)     = sum_{m,n} [zc_{mn} cos(m*theta - nfp*n*phi)
                                      + zs_{mn} sin(m*theta - nfp*n*phi)]
        x = x_hat * cos(2*pi*phi) - y_hat * sin(2*pi*phi)
        y = x_hat * sin(2*pi*phi) + y_hat * cos(2*pi*phi)

    Under stellarator symmetry the ``xs``, ``yc``, and ``zc`` terms are zero.

    The DOF vector is ``[xc, ys, zs]`` for stellarator-symmetric surfaces and
    ``[xc, xs, yc, ys, zc, zs]`` otherwise, matching simsopt's convention
    exactly. The (m, n) mode indexing follows :func:`make_rzfourier_mc_ms_nc_ns`.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def from_simsopt(simsopt_surf):
        """Load from a :class:`simsopt.geo.SurfaceXYZFourier` instance."""
        return SurfaceXYZFourierJAX(
            nfp=simsopt_surf.nfp,
            stellsym=simsopt_surf.stellsym,
            mpol=simsopt_surf.mpol,
            ntor=simsopt_surf.ntor,
            quadpoints_phi=jnp.array(simsopt_surf.quadpoints_phi),
            quadpoints_theta=jnp.array(simsopt_surf.quadpoints_theta),
            dofs=jnp.array(simsopt_surf.get_dofs()),
        )

    def to_simsopt(self):
        """Convert to :class:`simsopt.geo.SurfaceXYZFourier`."""
        try:
            from simsopt.geo import SurfaceXYZFourier
        except ImportError:
            raise ModuleNotFoundError(
                'Simsopt must be installed to export surface with to_simsopt().'
            )
        surf = SurfaceXYZFourier(
            nfp=self.nfp,
            stellsym=self.stellsym,
            mpol=self.mpol,
            ntor=self.ntor,
            quadpoints_phi=np.array(self.quadpoints_phi),
            quadpoints_theta=np.array(self.quadpoints_theta),
        )
        surf.set_dofs(np.array(self.dofs))
        return surf

    # ------------------------------------------------------------------
    # JAX pytree protocol
    # ------------------------------------------------------------------

    def tree_flatten(self):
        children = (
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.dofs,
            self.theta_mesh,
            self.phi_mesh,
            self.dphi,
            self.dtheta,
        )
        aux_data = {
            'nfp': self.nfp,
            'stellsym': self.stellsym,
            'mpol': self.mpol,
            'ntor': self.ntor,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            nfp=aux_data['nfp'],
            stellsym=aux_data['stellsym'],
            mpol=aux_data['mpol'],
            ntor=aux_data['ntor'],
            quadpoints_phi=children[0],
            quadpoints_theta=children[1],
            dofs=children[2],
        )

    # ------------------------------------------------------------------
    # Static methods for DOF operations
    # ------------------------------------------------------------------

    @staticmethod
    @partial(jit, static_argnames=['nfp', 'stellsym', 'dash1_order', 'dash2_order', 'mpol', 'ntor'])
    def dof_to_xhatz_op(
            phi_grid, theta_grid,
            nfp: int, stellsym: bool,
            dash1_order=0, dash2_order=0,
            mpol: int = 10, ntor: int = 10):
        """Operator mapping DOF vector -> (x_hat, y_hat, z) on the quadrature grid.

        Returns
        -------
        A_lstsq : array, shape (nphi, ntheta, 3, ndof)
            Operator such that ``A_lstsq @ dofs`` gives (x_hat, y_hat, z).
        m_2_n_2 : array, shape (ndof,)
            Per-DOF mode-number penalty weights ``m^2 + n^2``.
        """
        mc, ms, nc, ns = make_rzfourier_mc_ms_nc_ns(mpol, ntor)
        total_neg = (dash1_order + dash2_order) // 2
        derivative_factor_c = (
            (-nc[:, None, None] * jnp.pi * 2 * nfp) ** dash1_order
            * (mc[:, None, None] * jnp.pi * 2) ** dash2_order
        ) * (-1) ** total_neg
        derivative_factor_s = (
            (-ns[:, None, None] * jnp.pi * 2 * nfp) ** dash1_order
            * (ms[:, None, None] * jnp.pi * 2) ** dash2_order
        ) * (-1) ** total_neg
        if (dash1_order + dash2_order) % 2 == 0:
            cmn = derivative_factor_c * jnp.cos(
                mc[:, None, None] * jnp.pi * 2 * theta_grid[None, :, :]
                - nc[:, None, None] * jnp.pi * 2 * nfp * phi_grid[None, :, :]
            )
            smn = derivative_factor_s * jnp.sin(
                ms[:, None, None] * jnp.pi * 2 * theta_grid[None, :, :]
                - ns[:, None, None] * jnp.pi * 2 * nfp * phi_grid[None, :, :]
            )
        else:
            cmn = -derivative_factor_c * jnp.sin(
                mc[:, None, None] * theta_grid[None, :, :] * jnp.pi * 2
                - nc[:, None, None] * phi_grid[None, :, :] * jnp.pi * 2 * nfp
            )
            smn = derivative_factor_s * jnp.cos(
                ms[:, None, None] * theta_grid[None, :, :] * jnp.pi * 2
                - ns[:, None, None] * phi_grid[None, :, :] * jnp.pi * 2 * nfp
            )
        mc_2_nc_2 = mc ** 2 + nc ** 2
        ms_2_ns_2 = ms ** 2 + ns ** 2
        if stellsym:
            # DOF layout: [xc (n_c), ys (n_s), zs (n_s)]
            m_2_n_2 = jnp.concatenate([mc_2_nc_2, ms_2_ns_2, ms_2_ns_2])
        else:
            # DOF layout: [xc (n_c), xs (n_s), yc (n_c), ys (n_s), zc (n_c), zs (n_s)]
            cs = jnp.concatenate([mc_2_nc_2, ms_2_ns_2])
            m_2_n_2 = jnp.concatenate([cs, cs, cs])
        # For stellsym: DOFs = [xc, ys, zs] -> xhat uses cmn, yhat uses smn, z uses smn
        # For non-stellsym: DOFs = [xc, xs, yc, ys, zc, zs]
        #   xhat = xc*cmn + xs*smn, yhat = yc*cmn + ys*smn, z = zc*cmn + zs*smn
        if stellsym:
            xhat_operator = cmn             # shape (n_c_modes, nphi, ntheta)
            yhat_operator = smn             # shape (n_s_modes, nphi, ntheta)
            z_operator = smn
        else:
            xhat_operator = jnp.concatenate([cmn, smn], axis=0)
            yhat_operator = jnp.concatenate([cmn, smn], axis=0)
            z_operator    = jnp.concatenate([cmn, smn], axis=0)

        # Build per-component operators padded with zeros for the other components
        # stellsym: [xhat_block | yhat_block | z_block] -> 3 separate blocks
        # non-stellsym: [xhat_full | yhat_full | z_full] -> 3 blocks of same size
        if stellsym:
            n_c = cmn.shape[0]
            n_s = smn.shape[0]
            xhat_padded = jnp.concatenate([
                xhat_operator,
                jnp.zeros_like(yhat_operator),
                jnp.zeros_like(z_operator),
            ], axis=0)
            yhat_padded = jnp.concatenate([
                jnp.zeros_like(xhat_operator),
                yhat_operator,
                jnp.zeros_like(z_operator),
            ], axis=0)
            z_padded = jnp.concatenate([
                jnp.zeros_like(xhat_operator),
                jnp.zeros_like(yhat_operator),
                z_operator,
            ], axis=0)
        else:
            n_full = xhat_operator.shape[0]
            xhat_padded = jnp.concatenate([
                xhat_operator,
                jnp.zeros((n_full, *xhat_operator.shape[1:])),
                jnp.zeros((n_full, *xhat_operator.shape[1:])),
            ], axis=0)
            yhat_padded = jnp.concatenate([
                jnp.zeros((n_full, *yhat_operator.shape[1:])),
                yhat_operator,
                jnp.zeros((n_full, *yhat_operator.shape[1:])),
            ], axis=0)
            z_padded = jnp.concatenate([
                jnp.zeros((n_full, *z_operator.shape[1:])),
                jnp.zeros((n_full, *z_operator.shape[1:])),
                z_operator,
            ], axis=0)

        # A_lstsq shape: (nphi, ntheta, 3, ndof)
        A_lstsq = jnp.stack(
            [xhat_padded, yhat_padded, z_padded], axis=-1
        )  # (ndof, nphi, ntheta, 3)
        A_lstsq = jnp.moveaxis(A_lstsq, 0, -1)  # (nphi, ntheta, 3, ndof)
        return A_lstsq, m_2_n_2

    @staticmethod
    def _dof_to_gamma_op(
            phi_grid, theta_grid,
            nfp: int, stellsym: bool,
            dash1_order=0, dash2_order=0,
            mpol: int = 10, ntor: int = 10):
        """Operator of shape (nphi, ntheta, 3, ndof) mapping dofs -> gamma.

        Applies the Leibniz rule to differentiate
        ``x = x_hat * cos(phi_rad) - y_hat * sin(phi_rad)`` and similarly for y.
        """
        dof_to_x = 0.
        dof_to_y = 0.
        for dash1_order_xhatz in range(dash1_order + 1):
            dash1_order_trig = dash1_order - dash1_order_xhatz
            dof_to_xhatz_dash, _ = SurfaceXYZFourierJAX.dof_to_xhatz_op(
                phi_grid=phi_grid,
                theta_grid=theta_grid,
                nfp=nfp,
                stellsym=stellsym,
                dash1_order=dash1_order_xhatz,
                dash2_order=dash2_order,
                mpol=mpol,
                ntor=ntor,
            )
            # dof_to_xhatz_dash: (nphi, ntheta, 3, ndof); channels: [xhat, yhat, z]
            dof_to_xhat_dash = dof_to_xhatz_dash[:, :, 0, :]  # (nphi, ntheta, ndof)
            dof_to_yhat_dash = dof_to_xhatz_dash[:, :, 1, :]
            if dash1_order_xhatz == dash1_order:
                dof_to_z = dof_to_xhatz_dash[:, :, 2, :]

            total_neg = dash1_order_trig // 2
            binomial_coef = (
                factorial(dash1_order)
                / factorial(dash1_order_xhatz)
                / factorial(dash1_order_trig)
            )
            derivative_factor = (
                binomial_coef * (-1) ** total_neg * (jnp.pi * 2) ** dash1_order_trig
            )
            # Leibniz rule:
            #   d^a x / dphi^a = sum_k C(a,k) xhat^(k) * (d^(a-k) cos) - yhat^(k) * (d^(a-k) sin)
            #   d^a y / dphi^a = sum_k C(a,k) xhat^(k) * (d^(a-k) sin) + yhat^(k) * (d^(a-k) cos)
            if dash1_order_trig % 2 == 0:
                dof_to_x += (
                    derivative_factor
                    * dof_to_xhat_dash
                    * jnp.cos(phi_grid * jnp.pi * 2)[:, :, None]
                )
                dof_to_x -= (
                    derivative_factor
                    * dof_to_yhat_dash
                    * jnp.sin(phi_grid * jnp.pi * 2)[:, :, None]
                )
                dof_to_y += (
                    derivative_factor
                    * dof_to_xhat_dash
                    * jnp.sin(phi_grid * jnp.pi * 2)[:, :, None]
                )
                dof_to_y += (
                    derivative_factor
                    * dof_to_yhat_dash
                    * jnp.cos(phi_grid * jnp.pi * 2)[:, :, None]
                )
            else:
                dof_to_x -= (
                    derivative_factor
                    * dof_to_xhat_dash
                    * jnp.sin(phi_grid * jnp.pi * 2)[:, :, None]
                )
                dof_to_x -= (
                    derivative_factor
                    * dof_to_yhat_dash
                    * jnp.cos(phi_grid * jnp.pi * 2)[:, :, None]
                )
                dof_to_y += (
                    derivative_factor
                    * dof_to_xhat_dash
                    * jnp.cos(phi_grid * jnp.pi * 2)[:, :, None]
                )
                dof_to_y -= (
                    derivative_factor
                    * dof_to_yhat_dash
                    * jnp.sin(phi_grid * jnp.pi * 2)[:, :, None]
                )
        return jnp.concatenate(
            [dof_to_x[:, :, None, :], dof_to_y[:, :, None, :], dof_to_z[:, :, None, :]], axis=2
        )

    @staticmethod
    @partial(jit, static_argnames=['nfp', 'stellsym', 'mpol', 'ntor'])
    def _build_surface_fit_matrices(
            phi_target, theta_target, gamma_target,
            nfp: int, stellsym: bool,
            mpol: int = 5, ntor: int = 5):
        phi_rad = phi_target * jnp.pi * 2
        cos_phi = jnp.cos(phi_rad)
        sin_phi = jnp.sin(phi_rad)
        x_cart = gamma_target[:, :, 0]
        y_cart = gamma_target[:, :, 1]
        z_cart = gamma_target[:, :, 2]
        xhat = x_cart * cos_phi + y_cart * sin_phi
        yhat = -x_cart * sin_phi + y_cart * cos_phi
        A_lstsq, m_2_n_2 = SurfaceXYZFourierJAX.dof_to_xhatz_op(
            theta_grid=theta_target,
            phi_grid=phi_target,
            nfp=nfp,
            stellsym=stellsym,
            mpol=mpol,
            ntor=ntor,
        )
        b_lstsq = jnp.stack([xhat, yhat, z_cart], axis=-1)
        return A_lstsq, b_lstsq, m_2_n_2

    # ------------------------------------------------------------------
    # Broadcastable evaluator
    # ------------------------------------------------------------------

    @partial(jit, static_argnames=['a', 'b'])
    def gammadash_at_point(self, phi, theta, a: int, b: int) -> jnp.ndarray:
        """Direct broadcastable evaluation of d^(a+b) gamma / dphi^a dtheta^b.

        Operates on (xhat, yhat, z) directly in mode space and rotates to
        Cartesian (x, y, z) via the Leibniz rule on
        ``x = xhat*cos(2pi phi) - yhat*sin(2pi phi)``,
        ``y = xhat*sin(2pi phi) + yhat*cos(2pi phi)``.
        """
        nfp = self.nfp
        stellsym = self.stellsym
        mpol = self.mpol
        ntor = self.ntor
        dofs = self.dofs

        mc, ms, nc, ns = make_rzfourier_mc_ms_nc_ns(mpol, ntor)
        n_c = mc.shape[0]
        n_s = ms.shape[0]

        # Slice DOFs to match the layout used in ``dof_to_xhatz_op``.
        if stellsym:
            xc = dofs[:n_c]
            ys = dofs[n_c:n_c + n_s]
            zs = dofs[n_c + n_s:]
            xs_use = None
            yc_use = None
            zc_use = None
        else:
            i0 = 0
            xc = dofs[i0:i0 + n_c]; i0 += n_c
            xs_use = dofs[i0:i0 + n_s]; i0 += n_s
            yc_use = dofs[i0:i0 + n_c]; i0 += n_c
            ys = dofs[i0:i0 + n_s]; i0 += n_s
            zc_use = dofs[i0:i0 + n_c]; i0 += n_c
            zs = dofs[i0:i0 + n_s]

        pi2 = 2.0 * jnp.pi
        pi2nfp = pi2 * nfp
        phi_e = phi[..., None]
        theta_e = theta[..., None]

        def compute_xyhat_z(k_phi, k_theta):
            """(xhat, yhat, z) for derivative orders (k_phi, k_theta)."""
            ang_c = mc * pi2 * theta_e - nc * pi2nfp * phi_e
            ang_s = ms * pi2 * theta_e - ns * pi2nfp * phi_e

            total_neg = (k_phi + k_theta) // 2
            sign = (-1) ** total_neg
            fac_c = sign * (-nc * pi2nfp) ** k_phi * (mc * pi2) ** k_theta
            fac_s = sign * (-ns * pi2nfp) ** k_phi * (ms * pi2) ** k_theta

            if (k_phi + k_theta) % 2 == 0:
                basis_c = fac_c * jnp.cos(ang_c)
                basis_s = fac_s * jnp.sin(ang_s)
            else:
                basis_c = -fac_c * jnp.sin(ang_c)
                basis_s = fac_s * jnp.cos(ang_s)

            if stellsym:
                xhat = basis_c @ xc
                yhat = basis_s @ ys
                z = basis_s @ zs
            else:
                xhat = basis_c @ xc + basis_s @ xs_use
                yhat = basis_c @ yc_use + basis_s @ ys
                z = basis_c @ zc_use + basis_s @ zs
            return xhat, yhat, z

        cosphi = jnp.cos(pi2 * phi)
        sinphi = jnp.sin(pi2 * phi)

        dof_to_x = 0.0
        dof_to_y = 0.0
        Z_final = None
        for k in range(a + 1):
            a_trig = a - k
            xhat_k, yhat_k, z_k = compute_xyhat_z(k, b)
            if k == a:
                Z_final = z_k
            binomial_coef = comb(a, k)
            total_neg = a_trig // 2
            derivative_factor = binomial_coef * (-1) ** total_neg * pi2 ** a_trig
            if a_trig % 2 == 0:
                dof_to_x = dof_to_x + derivative_factor * (xhat_k * cosphi - yhat_k * sinphi)
                dof_to_y = dof_to_y + derivative_factor * (xhat_k * sinphi + yhat_k * cosphi)
            else:
                dof_to_x = dof_to_x + derivative_factor * (-xhat_k * sinphi - yhat_k * cosphi)
                dof_to_y = dof_to_y + derivative_factor * (xhat_k * cosphi - yhat_k * sinphi)

        return jnp.stack([dof_to_x, dof_to_y, Z_final], axis=-1)


# ======================================================================
# Helper functions for SurfaceRZFourierJAX
# ======================================================================

@partial(jit, static_argnames=['mpol', 'ntor'])
def make_rzfourier_mc_ms_nc_ns(mpol: int, ntor: int):
    ms = jnp.concatenate([
        jnp.zeros(ntor),
        jnp.repeat(jnp.arange(1, mpol + 1), ntor * 2 + 1)
    ])
    ns = jnp.concatenate([
        jnp.arange(1, ntor + 1),
        jnp.tile(jnp.arange(-ntor, ntor + 1), mpol)
    ])
    mc = jnp.concatenate([jnp.zeros(1), ms])
    nc = jnp.concatenate([jnp.zeros(1), ns])
    return mc, ms, nc, ns


# ======================================================================
# Helper functions for SurfaceXYZTensorFourierJAX
# ======================================================================

@lru_cache(maxsize=None)
def _xyztensor_active_indices(mpol: int, ntor: int, stellsym: bool):
    """Return active (m, n) index arrays for each coordinate.

    Ordering matches simsopt's ``get_dofs()`` / ``set_dofs_impl()`` exactly:
    iterate m = 0..2*mpol, then n = 0..2*ntor, skip where appropriate.

    Returns
    -------
    (rows_x, cols_x, rows_y, cols_y, rows_z, cols_z)
        Six 1-D numpy int arrays.
    """
    rows_x, cols_x = [], []
    rows_y, cols_y = [], []

    for m in range(2 * mpol + 1):
        for n in range(2 * ntor + 1):
            # x (dim=0): skip if (n<=ntor and m>mpol) or (n>ntor and m<=mpol)
            skip_x = stellsym and (
                (n <= ntor and m > mpol) or (n > ntor and m <= mpol)
            )
            # y, z (dim=1,2): skip if (n<=ntor and m<=mpol) or (n>ntor and m>mpol)
            skip_yz = stellsym and (
                (n <= ntor and m <= mpol) or (n > ntor and m > mpol)
            )
            if not skip_x:
                rows_x.append(m); cols_x.append(n)
            if not skip_yz:
                rows_y.append(m); cols_y.append(n)

    rows_x = np.array(rows_x, dtype=np.intp)
    cols_x = np.array(cols_x, dtype=np.intp)
    rows_y = np.array(rows_y, dtype=np.intp)
    cols_y = np.array(cols_y, dtype=np.intp)
    # y and z have the same mask
    return rows_x, cols_x, rows_y, cols_y, rows_y.copy(), cols_y.copy()


def _xyztensor_V_at_point(phi, ntor: int, nfp: int, order: int):
    """Toroidal basis at arbitrary-shaped phi (broadcastable).

    Same basis as :func:`_xyztensor_V` but accepts ``phi`` of any shape ``S``
    and returns an array of shape ``S + (2*ntor+1,)``.
    """
    pi2 = 2.0 * jnp.pi
    phi_r = pi2 * phi[..., None]                     # S + (1,)

    n_cos = jnp.arange(ntor + 1)                     # (ntor+1,)
    n_sin = jnp.arange(1, ntor + 1)                  # (ntor,)

    ang_cos = nfp * n_cos * phi_r                    # S + (ntor+1,)
    ang_sin = nfp * n_sin * phi_r                    # S + (ntor,)

    fc = nfp * n_cos * pi2                           # (ntor+1,)
    fs = nfp * n_sin * pi2                           # (ntor,)

    r = order % 4
    if r == 0:
        v_cos = jnp.cos(ang_cos)
        v_sin = jnp.sin(ang_sin)
    elif r == 1:
        v_cos = -fc * jnp.sin(ang_cos)
        v_sin =  fs * jnp.cos(ang_sin)
    elif r == 2:
        v_cos = -(fc ** 2) * jnp.cos(ang_cos)
        v_sin = -(fs ** 2) * jnp.sin(ang_sin)
    else:  # r == 3
        v_cos =  (fc ** 3) * jnp.sin(ang_cos)
        v_sin = -(fs ** 3) * jnp.cos(ang_sin)

    return jnp.concatenate([v_cos, v_sin], axis=-1)  # S + (2*ntor+1,)


def _xyztensor_W_at_point(theta, mpol: int, order: int):
    """Poloidal basis at arbitrary-shaped theta (broadcastable).

    Same basis as :func:`_xyztensor_W` but accepts ``theta`` of any shape ``S``
    and returns an array of shape ``S + (2*mpol+1,)``.
    """
    pi2 = 2.0 * jnp.pi
    theta_r = pi2 * theta[..., None]                 # S + (1,)

    m_cos = jnp.arange(mpol + 1)                     # (mpol+1,)
    m_sin = jnp.arange(1, mpol + 1)                  # (mpol,)

    ang_cos = m_cos * theta_r                        # S + (mpol+1,)
    ang_sin = m_sin * theta_r                        # S + (mpol,)

    fc = m_cos * pi2                                 # (mpol+1,)
    fs = m_sin * pi2                                 # (mpol,)

    r = order % 4
    if r == 0:
        w_cos = jnp.cos(ang_cos)
        w_sin = jnp.sin(ang_sin)
    elif r == 1:
        w_cos = -fc * jnp.sin(ang_cos)
        w_sin =  fs * jnp.cos(ang_sin)
    elif r == 2:
        w_cos = -(fc ** 2) * jnp.cos(ang_cos)
        w_sin = -(fs ** 2) * jnp.sin(ang_sin)
    else:  # r == 3
        w_cos =  (fc ** 3) * jnp.sin(ang_cos)
        w_sin = -(fs ** 3) * jnp.cos(ang_sin)

    return jnp.concatenate([w_cos, w_sin], axis=-1)  # S + (2*mpol+1,)


def _xyztensor_V(quadpoints_phi, ntor: int, nfp: int, order: int):
    """Toroidal basis functions (or their `order`-th derivative w.r.t. phi_norm).

    Basis::

        v_j(phi_norm):
            j = 0..ntor       ->  cos(nfp * j * 2π * phi_norm)
            j = ntor+1..2*ntor -> sin(nfp * (j-ntor) * 2π * phi_norm)

    Parameters
    ----------
    quadpoints_phi : (nphi,) array, values in [0, 1)
    order : 0, 1, or 2

    Returns
    -------
    V : (nphi, 2*ntor+1)
    """
    pi2 = 2.0 * jnp.pi
    phi_r = pi2 * quadpoints_phi[:, None]          # (nphi, 1)

    n_cos = jnp.arange(ntor + 1)                   # 0..ntor
    n_sin = jnp.arange(1, ntor + 1)                # 1..ntor

    ang_cos = nfp * n_cos[None, :] * phi_r         # (nphi, ntor+1)
    ang_sin = nfp * n_sin[None, :] * phi_r         # (nphi, ntor)

    # Frequencies w.r.t. phi_norm (include 2π already absorbed into phi_r)
    fc = (nfp * n_cos * pi2)[None, :]              # (1, ntor+1)
    fs = (nfp * n_sin * pi2)[None, :]              # (1, ntor)

    # d^k cos(f*phi_norm)/dphi_norm^k:
    #   k%4==0: cos,  k%4==1: -f*sin,  k%4==2: -f²*cos,  k%4==3: f³*sin
    # d^k sin(f*phi_norm)/dphi_norm^k:
    #   k%4==0: sin,  k%4==1:  f*cos,  k%4==2: -f²*sin,  k%4==3: -f³*cos
    r = order % 4
    if r == 0:
        v_cos = jnp.cos(ang_cos)
        v_sin = jnp.sin(ang_sin)
    elif r == 1:
        v_cos = -fc * jnp.sin(ang_cos)
        v_sin =  fs * jnp.cos(ang_sin)
    elif r == 2:
        v_cos = -(fc ** 2) * jnp.cos(ang_cos)
        v_sin = -(fs ** 2) * jnp.sin(ang_sin)
    else:  # r == 3
        v_cos =  (fc ** 3) * jnp.sin(ang_cos)
        v_sin = -(fs ** 3) * jnp.cos(ang_sin)

    return jnp.concatenate([v_cos, v_sin], axis=1)   # (nphi, 2*ntor+1)


def _xyztensor_W(quadpoints_theta, mpol: int, order: int):
    """Poloidal basis functions (or their `order`-th derivative w.r.t. theta_norm).

    Basis::

        w_i(theta_norm):
            i = 0..mpol       ->  cos(i * 2π * theta_norm)
            i = mpol+1..2*mpol ->  sin((i-mpol) * 2π * theta_norm)

    Returns
    -------
    W : (ntheta, 2*mpol+1)
    """
    pi2 = 2.0 * jnp.pi
    theta_r = pi2 * quadpoints_theta[:, None]      # (ntheta, 1)

    m_cos = jnp.arange(mpol + 1)                   # 0..mpol
    m_sin = jnp.arange(1, mpol + 1)                # 1..mpol

    ang_cos = m_cos[None, :] * theta_r             # (ntheta, mpol+1)
    ang_sin = m_sin[None, :] * theta_r             # (ntheta, mpol)

    fc = (m_cos * pi2)[None, :]                    # (1, mpol+1)
    fs = (m_sin * pi2)[None, :]                    # (1, mpol)

    r = order % 4
    if r == 0:
        w_cos = jnp.cos(ang_cos)
        w_sin = jnp.sin(ang_sin)
    elif r == 1:
        w_cos = -fc * jnp.sin(ang_cos)
        w_sin =  fs * jnp.cos(ang_sin)
    elif r == 2:
        w_cos = -(fc ** 2) * jnp.cos(ang_cos)
        w_sin = -(fs ** 2) * jnp.sin(ang_sin)
    else:  # r == 3
        w_cos =  (fc ** 3) * jnp.sin(ang_cos)
        w_sin = -(fs ** 3) * jnp.cos(ang_sin)

    return jnp.concatenate([w_cos, w_sin], axis=1)   # (ntheta, 2*mpol+1)


@partial(jit, static_argnames=['nfp', 'stellsym', 'a', 'b', 'mpol', 'ntor'])
def xyztensor_gammadash(
        dofs, quadpoints_phi, quadpoints_theta,
        nfp: int, stellsym: bool,
        a: int, b: int,
        mpol: int, ntor: int):
    """Compute ``d^(a+b) gamma / dphi^a dtheta^b`` for XYZ tensor Fourier surface.

    Uses the Leibniz product rule to differentiate
    ``x = x_hat * cos(phi_rad) - y_hat * sin(phi_rad)`` and similarly for y,
    then combines with the theta derivative contained in the W basis.

    Parameters
    ----------
    dofs : 1-D jax array
    quadpoints_phi, quadpoints_theta : 1-D jax arrays in [0, 1)
    nfp, stellsym, a, b, mpol, ntor : static

    Returns
    -------
    jnp.ndarray, shape (nphi, ntheta, 3)
    """
    # ------------------------------------------------------------------
    # 1. Reconstruct full coefficient matrices from active DOFs
    # ------------------------------------------------------------------
    rows_x, cols_x, rows_y, cols_y, rows_z, cols_z = _xyztensor_active_indices(
        mpol, ntor, stellsym
    )
    ndof_x = len(rows_x)
    ndof_y = len(rows_y)

    x_dofs = dofs[:ndof_x]
    y_dofs = dofs[ndof_x: ndof_x + ndof_y]
    z_dofs = dofs[ndof_x + ndof_y:]

    shape = (2 * mpol + 1, 2 * ntor + 1)
    x_full = jnp.zeros(shape).at[rows_x, cols_x].set(x_dofs)
    y_full = jnp.zeros(shape).at[rows_y, cols_y].set(y_dofs)
    z_full = jnp.zeros(shape).at[rows_z, cols_z].set(z_dofs)

    # ------------------------------------------------------------------
    # 2. Build basis function matrices
    # ------------------------------------------------------------------
    # W^(b): theta basis with b-th derivative, shape (ntheta, 2*mpol+1)
    Wb = _xyztensor_W(quadpoints_theta, mpol, b)

    # V^(k) for k = 0..a: phi basis with k-th derivative, shape (nphi, 2*ntor+1)
    Vk = [_xyztensor_V(quadpoints_phi, ntor, nfp, k) for k in range(a + 1)]

    # xhat^(k,b) = V^(k) @ X.T @ W^(b).T  ->  (nphi, ntheta)
    def hat(Vk_mat, M):
        return (Vk_mat @ M.T) @ Wb.T

    xhat = [hat(Vk[k], x_full) for k in range(a + 1)]
    yhat = [hat(Vk[k], y_full) for k in range(a + 1)]
    zhat_a = hat(Vk[a], z_full)

    # ------------------------------------------------------------------
    # 3. Derivatives of cos/sin(phi_rad) w.r.t. phi_norm
    # ------------------------------------------------------------------
    pi2 = 2.0 * jnp.pi
    phi_r = pi2 * quadpoints_phi          # (nphi,)
    cosphi = jnp.cos(phi_r)[:, None]     # (nphi, 1) for broadcasting
    sinphi = jnp.sin(phi_r)[:, None]

    def _deriv_cos(k):
        """d^k cos(phi_rad) / dphi_norm^k, shape (nphi, 1)."""
        r = k % 4
        if r == 0: return cosphi
        if r == 1: return -pi2 * sinphi
        if r == 2: return -(pi2 ** 2) * cosphi
        return (pi2 ** 3) * sinphi

    def _deriv_sin(k):
        """d^k sin(phi_rad) / dphi_norm^k, shape (nphi, 1)."""
        r = k % 4
        if r == 0: return sinphi
        if r == 1: return pi2 * cosphi
        if r == 2: return -(pi2 ** 2) * sinphi
        return -(pi2 ** 3) * cosphi

    # ------------------------------------------------------------------
    # 4. Apply Leibniz rule:
    #    d^a x / dphi^a = sum_k C(a,k) * [xhat^(k) * d^(a-k) cos - yhat^(k) * d^(a-k) sin]
    #    d^a y / dphi^a = sum_k C(a,k) * [xhat^(k) * d^(a-k) sin + yhat^(k) * d^(a-k) cos]
    # ------------------------------------------------------------------
    res_x = sum(
        comb(a, k) * (xhat[k] * _deriv_cos(a - k) - yhat[k] * _deriv_sin(a - k))
        for k in range(a + 1)
    )
    res_y = sum(
        comb(a, k) * (xhat[k] * _deriv_sin(a - k) + yhat[k] * _deriv_cos(a - k))
        for k in range(a + 1)
    )

    return jnp.stack([res_x, res_y, zhat_a], axis=-1)


# ======================================================================
# A special SurfaceJAX subclass that represents an uniform offset surface.
# ======================================================================

@tree_util.register_pytree_node_class
class SurfaceOffsetJAX(SurfaceJAX):
    """Subclass that applies a fixed normal offset to any SurfaceJAX subclass.
    
    This class inherits from SurfaceJAX and offsets all geometric quantities by
    a fixed distance along the surface normal. It maintains the full SurfaceJAX
    interface and can be used anywhere a SurfaceJAX is expected.
    
    Parameters
    ----------
    base_surface : SurfaceJAX
        The underlying surface to offset.
    d_expand : float
        Distance to offset along the unit normal (positive = outward).
    
    Examples
    --------
    >>> plasma_surf = SurfaceRZFourierJAX(...)
    >>> winding_surf = SurfaceOffsetJAX(plasma_surf, d_expand=0.2)
    >>> gamma_offset = winding_surf.gamma()
    >>> isinstance(winding_surf, SurfaceJAX)  # Returns True
    True
    
    Notes
    -----
    DOF-related methods (e.g., `get_dofs()`, `from_simsopt()`, `_fit_dofs_from_gamma()`)
    raise NotImplementedError since offset surfaces don't have independent DOFs.
    """
    
    def __init__(
        self, 
        base_surface, 
        d_expand: float,
        quadpoints_phi=None,
        quadpoints_theta=None,
    ):
        if quadpoints_phi is None:
            quadpoints_phi = base_surface.quadpoints_phi
        if quadpoints_theta is None:
            quadpoints_theta = base_surface.quadpoints_theta
        # Call parent constructor with base surface parameters
        super().__init__(
            nfp=base_surface.nfp,
            stellsym=base_surface.stellsym,
            mpol=base_surface.mpol,
            ntor=base_surface.ntor,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            dofs=base_surface.dofs,
        )
        self.base_surface = base_surface.copy_and_set_quadpoints(
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        self.d_expand = d_expand
    
    @partial(jit, static_argnames=['a', 'b'])
    def gammadash(self, a: int, b: int) -> jnp.ndarray:
        """Surface position or mixed partial derivative with offset applied.
        
        For an offset surface, gamma_offset = gamma + d * unitnormal, so:
        d^(a+b)(gamma_offset) / dphi^a dtheta^b = 
            d^(a+b)(gamma) / dphi^a dtheta^b + d * d^(a+b)(unitnormal) / dphi^a dtheta^b
        
        Parameters
        ----------
        a : int
            Order of the phi derivative.
        b : int
            Order of the theta derivative.
        
        Returns
        -------
        jnp.ndarray, shape (nphi, ntheta, 3)
            The quantity ``d^(a+b) (gamma + d*unitnormal) / d phi^a d theta^b``.
        """
        if self.d_expand == 0.:
            return self.base_surface.gammadash(a, b)
        
        return (self.base_surface.gammadash(a, b) + 
                self.d_expand * self.base_surface.unitnormaldash(a, b))

    @partial(jit, static_argnames=['a', 'b'])
    def gammadash_at_point(self, phi, theta, a: int, b: int) -> jnp.ndarray:
        """Broadcastable derivative of ``gamma_offset = gamma + d * unitnormal``.

        Reuses the base surface's :meth:`gammadash_at_point` and
        :meth:`unitnormaldash_at_point`, so the cost is the cost of the base
        surface's at-point evaluators plus a single AD pass for the normal.
        """
        if self.d_expand == 0.:
            return self.base_surface.gammadash_at_point(phi, theta, a, b)
        return (
            self.base_surface.gammadash_at_point(phi, theta, a, b)
            + self.d_expand
            * self.base_surface.unitnormaldash_at_point(phi, theta, a, b)
        )

    def copy_and_set_quadpoints(self, quadpoints_phi, quadpoints_theta):
        """Create a new offset surface with different quadrature points."""
        new_base = self.base_surface.copy_and_set_quadpoints(
            quadpoints_phi, quadpoints_theta
        )
        return SurfaceOffsetJAX(new_base, self.d_expand)
    
    # ------------------------------------------------------------------
    # Methods from SurfaceJAX not supported by offset surfaces
    # ------------------------------------------------------------------
    
    def get_dofs(self):
        raise NotImplementedError(
            "get_dofs() is not supported for SurfaceOffsetJAX. "
            "Offset surfaces don't have independent DOFs - use base_surface.get_dofs() instead."
        )
    
    @classmethod
    def dof_to_gamma(cls, dofs, phi_grid, theta_grid, nfp, stellsym,
                     dash1_order=0, dash2_order=0, mpol: int = 10, ntor: int = 10):
        raise NotImplementedError(
            "dof_to_gamma() is not supported for SurfaceOffsetJAX. "
            "Offset surfaces are created from existing surfaces, not DOFs."
        )
    
    @staticmethod
    def _dof_to_gamma_op(phi_grid, theta_grid, nfp, stellsym,
                         dash1_order=0, dash2_order=0, mpol: int = 10, ntor: int = 10):
        raise NotImplementedError(
            "_dof_to_gamma_op() is not supported for SurfaceOffsetJAX. "
            "Offset surfaces are created from existing surfaces, not DOFs."
        )
    
    @staticmethod
    def _build_surface_fit_matrices(phi_target, theta_target, gamma_target,
                                     nfp: int, stellsym: bool,
                                     mpol: int = 5, ntor: int = 5):
        raise NotImplementedError(
            "_build_surface_fit_matrices() is not supported for SurfaceOffsetJAX. "
            "Offset surfaces cannot be fitted from target gamma points."
        )
    
    @classmethod
    def _fit_dofs_from_gamma(cls, phi_target, theta_target, gamma_target,
                             nfp: int, stellsym: bool,
                             mpol: int = 5, ntor: int = 5,
                             lam_tikhonov=0., custom_weight=None):
        raise NotImplementedError(
            "_fit_dofs_from_gamma() is not supported for SurfaceOffsetJAX. "
            "Offset surfaces cannot be fitted from target gamma points."
        )
    
    def uniform_offset(
        self, d_expand: float,
        quadpoints_phi=None,
        quadpoints_theta=None,
    ):
        raise NotImplementedError(
            "uniform_offset() is not supported for SurfaceOffsetJAX. "
            "Use SurfaceOffsetJAX directly to create offset surfaces."
        )
    
    def gen_winding_surface(self, d_expand, unitnormal=None,
                           mpol=7, ntor=7, pol_interp=2, tor_interp=2,
                           lam_tikhonov=1e-5, rule='self-intersection'):
        raise NotImplementedError(
            "gen_winding_surface() is not supported for SurfaceOffsetJAX. "
            "Use base_surface.gen_winding_surface() instead."
        )
    
    @classmethod
    def from_simsopt(cls, surface_simsopt):
        raise NotImplementedError(
            "from_simsopt() is not supported for SurfaceOffsetJAX. "
            "Create the base surface from simsopt first, then wrap with SurfaceOffsetJAX."
        )
    
    def to_simsopt(self):
        raise NotImplementedError(
            "to_simsopt() is not supported for SurfaceOffsetJAX. "
            "Offset surfaces cannot be directly converted to simsopt format."
        )
    
    def plot(self, **kwargs):
        raise NotImplementedError(
            "plot() is not supported for SurfaceOffsetJAX. "
            "To visualize, evaluate gamma() and plot the point cloud, "
            "or fit to a new surface and plot that."
        )
    
    # ------------------------------------------------------------------
    # JAX pytree protocol
    # ------------------------------------------------------------------
    
    def tree_flatten(self):
        """Flatten for JAX transformations."""
        children = (self.base_surface,)
        aux_data = {'d_expand': self.d_expand}
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten for JAX transformations."""
        return cls(children[0], aux_data['d_expand'])