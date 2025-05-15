import unittest
from quadcoil import QuadcoilParams, SurfaceRZFourierJAX
from quadcoil.quantity import K, K2
import jax.numpy as jnp
from simsopt import load
from load_test_data import load_data, compare
try:
    from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
    CPF_AVAILABLE = True
except ImportError:
    CPF_AVAILABLE = False

winding_surface, plasma_surface, cp, cpst, qp = load_data()
    
class QuadcoilKTest(unittest.TestCase):
    
    @unittest.skipIf(not CPF_AVAILABLE, "Skipping K test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_K(self):
        K_test = K(qp, {'phi': cp.get_dofs()})
        K_ans = cp.K()[:len(cp.winding_surface.quadpoints_phi)//cp.nfp]
        self.assertTrue(compare(K_test, K_ans))
        self.assertTrue(compare(K2(qp, {'phi': cp.get_dofs()}), jnp.linalg.norm(K_ans, axis=-1)**2))
       


if __name__ == "__main__":
    unittest.main()