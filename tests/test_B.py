import unittest
from jax import jvp, tree_util
import jax.numpy as jnp
import numpy as np
from quadcoil import QuadcoilParams
from quadcoil.quantity import winding_surface_B, Bnormal, f_B, f_K
from quadcoil.quantity.magnetic_field import _winding_surface_B
from load_test_data import load_data, compare
try:
    from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
    from simsopt.field.magneticfieldclasses import WindingSurfaceField
    CPF_AVAILABLE = True
except ImportError:
    CPF_AVAILABLE = False

winding_surface, plasma_surface, cp, cpst, qp = load_data()
np.random.seed(1)
_PHI = jnp.array(np.random.randn(qp.ndofs) * 1e6)


def _qp_with_chunk(src, bs_chunk_size):
    return QuadcoilParams(
        plasma_surface=src.plasma_surface,
        winding_surface=src.winding_surface,
        net_poloidal_current_amperes=src.net_poloidal_current_amperes,
        net_toroidal_current_amperes=src.net_toroidal_current_amperes,
        Bnormal_plasma=src.Bnormal_plasma,
        mpol=src.mpol,
        ntor=src.ntor,
        quadpoints_phi=src.quadpoints_phi,
        quadpoints_theta=src.quadpoints_theta,
        stellsym=src.stellsym,
        bs_chunk_size=bs_chunk_size,
    )


class QuadcoilBTest(unittest.TestCase):

    """
    Testing the operators in f_b_and_k_operators. Thest includes:
    - f_B_operator_and_current_scale 
    The integrated normal field error f_B at the surface
    - K_operator_cylindrical
    The surface current K in a cylindrical coordinate
    - K_operator
    The surface current K in the xyz coordinate
    - K_theta
    The surface current K along the theta direction
    """

    @unittest.skipIf(not CPF_AVAILABLE, "Skipping B test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_winding_surface_B(self):
        # quadcoil implementation
        B_test = winding_surface_B(qp, {'phi': cp.get_dofs()})
        # simsopt implementation
        Bfield = WindingSurfaceField(cp)
        points = plasma_surface.gamma().reshape(-1, 3)
        Bfield.set_points(points)
        B_ans = Bfield.B()
        self.assertTrue(compare(B_test.reshape(-1, 3), B_ans))

    @unittest.skipIf(not CPF_AVAILABLE, "Skipping Bnormal test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_B_normal(self):
        B_GI_test = Bnormal(qp, {'phi': jnp.zeros_like(cp.get_dofs())})
        self.assertTrue(compare(B_GI_test.flatten(), cpst.B_GI))

    @unittest.skipIf(not CPF_AVAILABLE, "Skipping f_B, f_K test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_f_B_and_f_K(self):
        phi, f_B_ans, f_K_ans = cpst.solve_tikhonov()
        f_B_val = f_B(qp, {'phi': phi})
        f_K_val = f_K(qp, {'phi': phi}) 
        self.assertTrue(compare(f_B_val, f_B_ans))
        self.assertTrue(compare(f_K_val, f_K_ans))

    def test_bs_chunk_size_parity(self):
        '''Chunked winding-surface B and f_B must match the unchunked kernels.'''
        dofs = {'phi': _PHI}
        qp_chunk = _qp_with_chunk(qp, 7)
        children, aux = tree_util.tree_flatten(qp_chunk)
        qp_roundtrip = tree_util.tree_unflatten(aux, children)
        self.assertEqual(qp_roundtrip.bs_chunk_size, 7)

        B_full = _winding_surface_B(qp, dofs)
        B_chunk = _winding_surface_B(qp_chunk, dofs)
        self.assertTrue(compare(B_full, B_chunk, err=1e-12))

        f_full = f_B(qp, dofs)
        f_chunk = f_B(qp_chunk, dofs)
        self.assertTrue(compare(f_full, f_chunk, err=1e-12))

        tang = jnp.ones_like(_PHI)
        _, df_full = jvp(lambda phi: f_B(qp, {'phi': phi}), (_PHI,), (tang,))
        _, df_chunk = jvp(lambda phi: f_B(qp_chunk, {'phi': phi}), (_PHI,), (tang,))
        self.assertTrue(compare(df_full, df_chunk, err=1e-10))


if __name__ == "__main__":
    unittest.main()