import unittest
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from quadcoil import QuadcoilParams, SurfaceRZFourierJAX, project_arr_cylindrical
from quadcoil.quantity import K_cyl
from quadcoil.quantity.current import _K
from quadcoil.quantity.force import _force_cyl, _force_cyl_legacy, _force_integrands_xyz
from quadcoil.quantity.self_field import _B_self, _B_self_cyl
from load_test_data import load_data

winding_surface, plasma_surface, cp, cpst, qp = load_data()
np.random.seed(0)
DOFS = {'phi': jnp.array(np.random.randn(qp.ndofs) * 1e6)}


def _self_mask(qp):
    '''
    The (n_phiy, n_thetay, n_phix, n_thetax) mask marking the coincident
    source/evaluation points, which _integrate_force removes structurally.
    '''
    n_phiy, n_thetay = qp.eval_surface.gamma().shape[:2]
    n_phix, n_thetax = qp.winding_surface.gamma().shape[:2]
    mask = np.zeros((n_phiy, n_thetay, n_phix, n_thetax), dtype=bool)
    i, j = np.meshgrid(np.arange(n_phiy), np.arange(n_thetay), indexing='ij')
    mask[i, j, i, j] = True
    return mask


def _kernels(qp):
    ''' The single- and double-layer kernels, times the area element. '''
    gamma_x = qp.winding_surface.gamma()
    diff = qp.eval_surface.gamma()[:, :, None, None, :] - gamma_x[None, None, :, :, :]
    da_x = qp.winding_surface.da()
    unitnormal_x = qp.winding_surface.unitnormal()
    mask = _self_mask(qp)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10 * mask)
    single = jnp.where(mask, 0., da_x[None, None] / dist)
    double = jnp.where(
        mask, 0.,
        da_x[None, None] * jnp.sum(diff * unitnormal_x[None, None], axis=-1) / dist**3
    )
    return single, double


def _force_xyz_reference(qp, dofs):
    '''
    A direct, x/y/z-component discretization of the Robin-Volpe self-force,
    evaluated over the whole winding surface at once. This uses neither the
    field-period folding nor the cylindrical projection of _force_cyl, so it is
    an independent reference for the tensor contraction itself.
    '''
    single_integrand, double_integrand = _force_integrands_xyz(
        qp, dofs, winding_surface_mode=True
    )
    single_kernel, double_kernel = _kernels(qp)
    operator = (
        jnp.einsum('ijkl,klab->ijab', single_kernel, single_integrand)
        + jnp.einsum('ijkl,klab->ijab', double_kernel, double_integrand)
    )
    return jnp.einsum('ija,ijab->ijb', _K(qp, dofs, winding_surface_mode=False), operator)


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
        K_val = _K(qp, DOFS, winding_surface_mode=False)
        force_from_B = jnp.cross(K_val, _B_self(qp, DOFS), axis=-1)
        force_reference = _force_xyz_reference(qp, DOFS)
        print('xyz K x B_self vs Robin-Volpe force, rel. err.:',
              _relerr(force_from_B, force_reference))
        self.assertTrue(_relerr(force_from_B, force_reference) < 1e-12)

    def test_cross_product_cyl(self):
        '''
        The same identity in cylindrical components, against _force_cyl itself.
        Both sides use identical quadrature weights, so this is exact rather
        than convergent -- provided project_arr_cylindrical uses an orthonormal
        basis, which test_cylindrical_basis_is_orthonormal checks separately.
        '''
        force_from_B = _force_cyl(qp, DOFS)
        force_reference = _force_cyl_legacy(qp, DOFS)
        print('cylindrical K x B_self vs _force_cyl, rel. err.:',
              _relerr(force_from_B, force_reference))
        self.assertTrue(_relerr(force_from_B, force_reference) < 1e-12)

    def test_cylindrical_basis_is_orthonormal(self):
        '''
        The cross product of two vectors' components equals the components of
        their cross product only in an orthonormal, right-handed basis. That is
        what makes test_cross_product_cyl exact, so guard it directly: the
        projection must preserve lengths and commute with the cross product.
        '''
        gamma = qp.eval_surface.gamma()
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
        B_0 = _B_self_cyl(qp, {'phi': jnp.zeros(qp.ndofs)})
        B_a = _B_self_cyl(qp, {'phi': phi_a}) - B_0
        B_b = _B_self_cyl(qp, {'phi': phi_b}) - B_0
        B_sum = _B_self_cyl(qp, {'phi': 2.5 * phi_a - 1.5 * phi_b}) - B_0
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
            gamma = qp_torus.eval_surface.gamma()
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
            gamma = qp_torus.eval_surface.gamma()
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


if __name__ == "__main__":
    unittest.main()
