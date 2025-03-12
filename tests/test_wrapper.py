import unittest
from quadcoil import get_objective
import jax.numpy as jnp
from simsopt import load
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
from load_test_data import load_data, compare

winding_surface, plasma_surface, cp, cpst, qp = load_data()
    
class QuadcoilWrapperTest(unittest.TestCase):

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

    def test_K(self):
        K_func = get_objective('K')
        K_test = K_func(qp, cp.get_dofs())
        K_ans = cp.K()[:len(cp.winding_surface.quadpoints_phi)//cp.nfp]
        self.assertTrue(compare(K_test, K_ans))
       


if __name__ == "__main__":
    unittest.main()