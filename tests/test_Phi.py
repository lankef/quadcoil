import unittest
from quadcoil import QuadcoilParams, SurfaceRZFourierJAX
from quadcoil.quantity import Phi
import jax.numpy as jnp
from simsopt import load
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
from load_test_data import load_data, compare
try:
    from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
    CPF_AVAILABLE = True
except ImportError:
    CPF_AVAILABLE = False
winding_surface, plasma_surface, cp, cpst, qp = load_data()
    
class QuadcoilPhiTest(unittest.TestCase):
    # Testing dipole density calculation
    @unittest.skipIf(not CPF_AVAILABLE, "Skipping dipole test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_Phi(self):
        Phi_test = Phi(qp, {'phi': cp.get_dofs()})
        Phi_ans = cp.Phi()[:len(cp.winding_surface.quadpoints_phi)//cp.nfp]
        self.assertTrue(compare(Phi_test, Phi_ans))

if __name__ == "__main__":
    unittest.main()