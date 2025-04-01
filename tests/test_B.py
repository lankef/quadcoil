import unittest
from quadcoil.objective import winding_surface_B, Bnormal, f_B, f_K
import jax.numpy as jnp
from load_test_data import load_data, compare
try:
    from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
    from simsopt.field.magneticfieldclasses import WindingSurfaceField
    CPF_AVAILABLE = True
except ImportError:
    CPF_AVAILABLE = False

winding_surface, plasma_surface, cp, cpst, qp = load_data()
    
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
        B_test = winding_surface_B(qp, cp.get_dofs())
        Bfield = WindingSurfaceField(cp)
        points = plasma_surface.gamma().reshape(-1, 3)
        Bfield.set_points(points)
        B_ans = Bfield.B()
        self.assertTrue(compare(B_test.reshape(-1, 3), B_ans))

    @unittest.skipIf(not CPF_AVAILABLE, "Skipping Bnormal test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_B_normal(self):
        B_GI_test = Bnormal(qp, jnp.zeros_like(cp.get_dofs()))
        self.assertTrue(compare(B_GI_test.flatten(), cpst.B_GI))

    @unittest.skipIf(not CPF_AVAILABLE, "Skipping f_B, f_K test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_f_B_and_f_K(self):
        phi, f_B_ans, f_K_ans = cpst.solve_tikhonov()
        f_B_val = f_B(qp, phi)
        f_K_val = f_K(qp, phi) 
        self.assertTrue(compare(f_B_val, f_B_ans))
        self.assertTrue(compare(f_K_val, f_K_ans))

if __name__ == "__main__":
    unittest.main()