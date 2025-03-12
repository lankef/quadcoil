import unittest
from quadcoil import QuadcoilParams, SurfaceRZFourierJAX
from quadcoil.objective import K, K2
import jax.numpy as jnp
from simsopt import load
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
from load_test_data import load_data, compare

winding_surface, plasma_surface, cp, cpst, qp = load_data()
    
class QuadcoilKTest(unittest.TestCase):
    def test_K(self):
        K_test = K(qp, cp.get_dofs())
        K_ans = cp.K()[:len(cp.winding_surface.quadpoints_phi)//cp.nfp]
        self.assertTrue(compare(K_test, K_ans))
        self.assertTrue(compare(K2(qp, cp.get_dofs()), jnp.linalg.norm(K_ans, axis=-1)**2))
       


if __name__ == "__main__":
    unittest.main()