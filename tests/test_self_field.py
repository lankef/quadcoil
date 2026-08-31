import unittest
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from quadcoil import QuadcoilParams, SurfaceRZFourierJAX, project_arr_cylindrical
from quadcoil.quantity.current import _K
from quadcoil.quantity.force import (
    _force_cyl, _force_cyl_legacy, _force_xyz, _force_integrands_xyz,
)
from quadcoil.quantity.self_field import (
    _B_self, _B_self_cyl, _B2_self, _B_self_norm, f_max_B2_self, f_l1_B_self_norm,
)
from load_test_data import load_data

winding_surface, plasma_surface, cp, cpst, qp = load_data()
np.random.seed(0)
DOFS = {'phi': jnp.array(np.random.randn(qp.ndofs) * 1e6)}


def _relerr(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


class QuadcoilBSelfTest(unittest.TestCase):
    '''
    Tests that _B_self / _B_self_cyl is the self-field generating the
    Robin-Volpe self-force implemented in quantity/force.py, i.e.
    K x B_self = L'.
    '''

    def test_cross_product_xyz(self):
        '''
        K x B_self must reproduce the Robin-Volpe force to machine precision,
        because the force's integrand tensors are exactly the duals of
        B_self's integrand vectors (see _B_self_integrands_xyz), and the two
        share the same quadrature weights.
        '''
        # Build the kernels directly against the winding 1fp slice.
        gamma_x = qp.winding_surface.gamma()
        n_phi_1fp = gamma_x.shape[0] // qp.nfp
        gamma_y = gamma_x[:n_phi_1fp]
        da_x = qp.winding_surface.da()
        unitnormal_x = qp.winding_surface.unitnormal()

        diff = gamma_y[:, :, None, None, :] - gamma_x[None, None, :, :, :]
        n_phix, n_thetax = gamma_x.shape[:2]
        i, j = np.meshgrid(np.arange(n_phi_1fp), np.arange(n_thetax), indexing='ij')
        mask = np.zeros((n_phi_1fp, n_thetax, n_phix, n_thetax), dtype=bool)
        mask[i, j, i, j] = True
        dist_sq = jnp.sum(diff**2, axis=-1)
        dist = jnp.sqrt(jnp.where(mask, 1.0, dist_sq))
        single_kernel = jnp.where(mask, 0., da_x[None, None] / dist)
        double_kernel = jnp.where(
            mask, 0.,
            da_x[None, None] * jnp.sum(diff * unitnormal_x[None, None], axis=-1) / dist**3
        )

        single_integrand, double_integrand = _force_integrands_xyz(
            qp, DOFS, winding_surface_mode=True
        )
        operator = (
            jnp.einsum('ijkl,klab->ijab', single_kernel, single_integrand)
            + jnp.einsum('ijkl,klab->ijab', double_kernel, double_integrand)
        )
        K_1fp = _K(qp, DOFS, winding_surface_mode='divide')
        force_reference = jnp.einsum('ija,ijab->ijb', K_1fp, operator)

        force_from_B = jnp.cross(K_1fp, _B_self(qp, DOFS), axis=-1)
        print('xyz K x B_self vs Robin-Volpe force, rel. err.:',
              _relerr(force_from_B, force_reference))
        self.assertTrue(_relerr(force_from_B, force_reference) < 1e-12)

    def test_cross_product_cyl(self):
        '''
        The same identity in cylindrical components, against _force_cyl_legacy
        (which evaluates on the winding 1fp grid, same as _B_self_cyl).  Both
        sides use identical quadrature weights and project at the same
        evaluation points, so this is exact rather than merely convergent.

        Note: _force_cyl (the public function) evaluates on eval_surface with a
        cylindrical nfp fold, which is a different computation; it is tested
        separately in test_force_cyl_axisymmetric_torus and
        test_mismatched_eval_grid_force.
        '''
        gamma_x = qp.winding_surface.gamma()
        gamma_y = gamma_x[:gamma_x.shape[0] // qp.nfp]
        K_1fp_cyl = project_arr_cylindrical(
            gamma_y,
            _K(qp, DOFS, winding_surface_mode='divide'),
        )
        force_from_B = jnp.cross(K_1fp_cyl, _B_self_cyl(qp, DOFS), axis=-1)
        force_reference = _force_cyl_legacy(qp, DOFS)
        print('cylindrical K x B_self vs _force_cyl_legacy, rel. err.:',
              _relerr(force_from_B, force_reference))
        self.assertTrue(_relerr(force_from_B, force_reference) < 1e-12)

    def test_cylindrical_basis_is_orthonormal(self):
        '''
        The cross product of two vectors' components equals the components of
        their cross product only in an orthonormal, right-handed basis. That is
        what makes test_cross_product_cyl exact, so guard it directly: the
        projection must preserve lengths and commute with the cross product.
        '''
        gamma_x = qp.winding_surface.gamma()
        gamma = gamma_x[:gamma_x.shape[0] // qp.nfp]
        u = jnp.array(np.random.randn(*gamma.shape))
        v = jnp.array(np.random.randn(*gamma.shape))
        u_cyl = project_arr_cylindrical(gamma, u)
        v_cyl = project_arr_cylindrical(gamma, v)
        self.assertTrue(_relerr(
            jnp.linalg.norm(u_cyl, axis=-1),
            jnp.linalg.norm(u, axis=-1)
        ) < 1e-12)
        self.assertTrue(_relerr(
            jnp.cross(u_cyl, v_cyl, axis=-1),
            project_arr_cylindrical(gamma, jnp.cross(u, v, axis=-1))
        ) < 1e-12)

    def test_affine_in_phi_mn(self):
        '''
        B_self must be affine in phi_mn: linear in the single-valued part, plus
        a constant from the net poloidal/toroidal currents.
        '''
        phi_a = jnp.array(np.random.randn(qp.ndofs) * 1e6)
        phi_b = jnp.array(np.random.randn(qp.ndofs) * 1e6)
        B_0 = _B_self(qp, {'phi': jnp.zeros(qp.ndofs)})
        B_a = _B_self(qp, {'phi': phi_a}) - B_0
        B_b = _B_self(qp, {'phi': phi_b}) - B_0
        B_sum = _B_self(qp, {'phi': 2.5 * phi_a - 1.5 * phi_b}) - B_0
        print('affinity in phi_mn, rel. err.:', _relerr(B_sum, 2.5 * B_a - 1.5 * B_b))
        self.assertTrue(_relerr(B_sum, 2.5 * B_a - 1.5 * B_b) < 1e-10)

    def test_axisymmetric_torus(self):
        '''
        Physical check against an exactly solvable case. On a circular
        axisymmetric torus carrying only a net poloidal current G, the field is
        purely toroidal, mu_0*G/(2*pi*R) inside the winding surface and zero
        outside. The self-field is the average of the two one-sided limits, so
        |B_self| = mu_0*G/(4*pi*R) = 1e-7*G/R, with no R or Z component.

        The single-layer kernel is only weakly singular, so the trapezoidal
        quadrature converges at first order in the grid spacing. We check both
        the value and that rate.
        '''
        R0, minor_radius, G = 1.0, 0.3, 1e6
        errors = []
        for n_grid in [24, 48, 96]:
            surface = SurfaceRZFourierJAX(
                nfp=1, stellsym=True, mpol=1, ntor=0,
                quadpoints_phi=jnp.linspace(0, 1, n_grid, endpoint=False),
                quadpoints_theta=jnp.linspace(0, 1, n_grid, endpoint=False),
                dofs=jnp.array([R0, minor_radius, minor_radius]),
            )
            qp_torus = QuadcoilParams(
                plasma_surface=surface, winding_surface=surface,
                net_poloidal_current_amperes=G,
                net_toroidal_current_amperes=0.,
                mpol=1, ntor=1,
            )
            gamma = qp_torus.winding_surface.gamma()  # same as 1fp slice since nfp=1
            B_val = _B_self_cyl(qp_torus, {'phi': jnp.zeros(qp_torus.ndofs)})
            B_phi_exact = 1e-7 * G / jnp.linalg.norm(gamma[:, :, :2], axis=-1)
            scale = jnp.max(jnp.abs(B_phi_exact))
            # The sign follows the orientation of the surface normal.
            errors.append(float(jnp.max(jnp.abs(jnp.abs(B_val[:, :, 1]) - B_phi_exact)) / scale))
            self.assertTrue(jnp.max(jnp.abs(B_val[:, :, 0])) / scale < 1e-10)
            self.assertTrue(jnp.max(jnp.abs(B_val[:, :, 2])) / scale < 1e-10)
        print('axisymmetric torus B_phi rel. err. at n = 24, 48, 96:', errors)
        self.assertTrue(errors[0] < 0.1)
        # First-order convergence: halving the grid spacing halves the error.
        self.assertTrue(errors[1] / errors[0] < 0.6)
        self.assertTrue(errors[2] / errors[1] < 0.6)

    def test_force_cyl_axisymmetric_torus(self):
        '''
        The same torus seen from the force side. K is poloidal and B_self is
        toroidal, so the self-force is pure magnetic pressure: normal to the
        surface, of magnitude |K| * 1e-7*G/R.
        '''
        R0, minor_radius, G = 1.0, 0.3, 1e6
        errors = []
        for n_grid in [24, 48, 96]:
            surface = SurfaceRZFourierJAX(
                nfp=1, stellsym=True, mpol=1, ntor=0,
                quadpoints_phi=jnp.linspace(0, 1, n_grid, endpoint=False),
                quadpoints_theta=jnp.linspace(0, 1, n_grid, endpoint=False),
                dofs=jnp.array([R0, minor_radius, minor_radius]),
            )
            qp_torus = QuadcoilParams(
                plasma_surface=surface, winding_surface=surface,
                net_poloidal_current_amperes=G,
                net_toroidal_current_amperes=0.,
                mpol=1, ntor=1,
            )
            dofs = {'phi': jnp.zeros(qp_torus.ndofs)}
            gamma = qp_torus.winding_surface.gamma()  # same as 1fp slice since nfp=1
            force = _force_cyl(qp_torus, dofs)
            normal = project_arr_cylindrical(gamma, qp_torus.eval_surface.unitnormal())
            exact = (
                jnp.linalg.norm(_K(qp_torus, dofs, winding_surface_mode=False), axis=-1)
                * 1e-7 * G / jnp.linalg.norm(gamma[:, :, :2], axis=-1)
            )
            scale = jnp.max(jnp.abs(exact))
            # Pure magnetic pressure: no component along the surface.
            self.assertTrue(
                jnp.max(jnp.abs(jnp.cross(force, normal, axis=-1))) / scale < 1e-10)
            errors.append(float(
                jnp.max(jnp.abs(jnp.abs(jnp.sum(force * normal, -1)) - exact)) / scale))
        print('axisymmetric torus force rel. err. at n = 24, 48, 96:', errors)
        self.assertTrue(errors[0] < 0.1)
        self.assertTrue(errors[1] / errors[0] < 0.6)
        self.assertTrue(errors[2] / errors[1] < 0.6)


    def test_mismatched_eval_grid(self):
        '''
        Regression test: winding theta 34, eval theta 32 puts theta=0.5 in
        both grids.  The index-based self-mask alone does not catch those
        coincident off-diagonal pairs, but _B_self always uses
        gamma_x[:n_phi_1fp] so mismatched quadpoints_theta must not affect
        _B_self or _B2_self at all.
        '''
        nfp = qp.winding_surface.nfp
        n_phi = len(qp.winding_surface.quadpoints_phi) // nfp
        ws34 = qp.winding_surface.copy_and_set_quadpoints(
            quadpoints_phi=jnp.linspace(0, 1, n_phi * nfp, endpoint=False),
            quadpoints_theta=jnp.linspace(0, 1, 34, endpoint=False),
        )
        qp_mismatch = QuadcoilParams(
            plasma_surface=qp.plasma_surface,
            winding_surface=ws34,
            net_poloidal_current_amperes=qp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=qp.net_toroidal_current_amperes,
            quadpoints_phi=jnp.linspace(0, 1 / nfp, n_phi, endpoint=False),
            quadpoints_theta=jnp.linspace(0, 1, 32, endpoint=False),
            mpol=qp.mpol, ntor=qp.ntor,
        )
        B2_mismatch = _B2_self(qp_mismatch, {'phi': jnp.zeros(qp_mismatch.ndofs)})
        max_B2 = float(jnp.max(B2_mismatch))
        print('mismatched-grid max B2_self:', max_B2)
        self.assertTrue(jnp.isfinite(B2_mismatch).all())
        self.assertTrue(max_B2 < 1e10, f'B2_self blew up: {max_B2}')

    def test_mismatched_eval_grid_force(self):
        '''
        Regression test: winding theta 34, eval theta 32 creates coincident
        source/eval pairs that are NOT on the index-diagonal.  The extra
        dist_sq == 0 guard in _singular_layer_kernels (which backs both
        _integrate_B_self and the force path) must prevent NaN.
        '''
        nfp = qp.winding_surface.nfp
        n_phi = len(qp.winding_surface.quadpoints_phi) // nfp
        ws34 = qp.winding_surface.copy_and_set_quadpoints(
            quadpoints_phi=jnp.linspace(0, 1, n_phi * nfp, endpoint=False),
            quadpoints_theta=jnp.linspace(0, 1, 34, endpoint=False),
        )
        qp_mismatch = QuadcoilParams(
            plasma_surface=qp.plasma_surface,
            winding_surface=ws34,
            net_poloidal_current_amperes=qp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=qp.net_toroidal_current_amperes,
            quadpoints_phi=jnp.linspace(0, 1 / nfp, n_phi, endpoint=False),
            quadpoints_theta=jnp.linspace(0, 1, 32, endpoint=False),
            mpol=qp.mpol, ntor=qp.ntor,
        )
        dofs = {'phi': jnp.zeros(qp_mismatch.ndofs)}
        force_xyz = _force_xyz(qp_mismatch, dofs)
        force_cyl = _force_cyl(qp_mismatch, dofs)
        max_force_xyz = float(jnp.max(jnp.abs(force_xyz)))
        max_force_cyl = float(jnp.max(jnp.abs(force_cyl)))
        print('mismatched-grid max |force_xyz|:', max_force_xyz)
        print('mismatched-grid max |force_cyl|:', max_force_cyl)
        self.assertTrue(jnp.isfinite(force_xyz).all(), 'force_xyz contains non-finite values')
        self.assertTrue(jnp.isfinite(force_cyl).all(), 'force_cyl contains non-finite values')
        self.assertTrue(max_force_xyz < 1e15, f'force_xyz blew up: {max_force_xyz}')
        self.assertTrue(max_force_cyl < 1e15, f'force_cyl blew up: {max_force_cyl}')

    def test_bs_chunk_size_parity(self):
        '''Chunked self-field kernels must match the fully vectorized path.'''
        qp_chunk = QuadcoilParams(
            plasma_surface=qp.plasma_surface,
            winding_surface=qp.winding_surface,
            net_poloidal_current_amperes=qp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=qp.net_toroidal_current_amperes,
            Bnormal_plasma=qp.Bnormal_plasma,
            mpol=qp.mpol,
            ntor=qp.ntor,
            quadpoints_phi=qp.quadpoints_phi,
            quadpoints_theta=qp.quadpoints_theta,
            stellsym=qp.stellsym,
            bs_chunk_size=5,
        )
        B_full = _B_self(qp, DOFS)
        B_chunk = _B_self(qp_chunk, DOFS)
        print('chunked vs full _B_self rel. err.:', _relerr(B_chunk, B_full))
        self.assertTrue(_relerr(B_chunk, B_full) < 1e-12)

        B2_full = _B2_self(qp, DOFS)
        B2_chunk = _B2_self(qp_chunk, DOFS)
        self.assertTrue(_relerr(B2_chunk, B2_full) < 1e-12)

        fmax_full = f_max_B2_self(qp, DOFS)
        fmax_chunk = f_max_B2_self(qp_chunk, DOFS)
        self.assertTrue(_relerr(fmax_chunk, fmax_full) < 1e-12)

        force_full = _force_xyz(qp, DOFS)
        force_chunk = _force_xyz(qp_chunk, DOFS)
        print('chunked vs full _force_xyz rel. err.:', _relerr(force_chunk, force_full))
        self.assertTrue(_relerr(force_chunk, force_full) < 1e-12)

    def test_B_self_norm_and_l1(self):
        '''
        _B_self_norm is n · B_self on the winding 1fp grid, and
        f_l1_B_self_norm is the surface L-1 integral of that scalar.
        '''
        gamma_x = qp.winding_surface.gamma()
        n_phi_1fp = gamma_x.shape[0] // qp.nfp
        unitnormal_1fp = qp.winding_surface.unitnormal()[:n_phi_1fp]
        da_1fp = qp.winding_surface.da()[:n_phi_1fp]

        Bn_ref = jnp.sum(unitnormal_1fp * _B_self(qp, DOFS), axis=-1)
        Bn = _B_self_norm(qp, DOFS)
        self.assertTrue(_relerr(Bn, Bn_ref) < 1e-12)

        l1_ref = jnp.sum(da_1fp * jnp.abs(Bn)) * qp.nfp
        l1 = f_l1_B_self_norm(qp, DOFS)
        print('f_l1_B_self_norm vs hand integral, rel. err.:', _relerr(l1, l1_ref))
        self.assertTrue(_relerr(l1, l1_ref) < 1e-12)


if __name__ == "__main__":
    unittest.main()
