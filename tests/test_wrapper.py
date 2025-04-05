import unittest
from quadcoil import get_objective, merge_callables, parse_constraints, parse_objectives
from quadcoil.objective import f_B, f_K, K_dot_grad_K, f_B_normalized_by_Bnormal_IG
import jax.numpy as jnp
import numpy as np
from simsopt import load
from load_test_data import load_data, compare
try:
    from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
    CPF_AVAILABLE = True
except ImportError:
    CPF_AVAILABLE = False

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
    @unittest.skipIf(not CPF_AVAILABLE, "Skipping objective wrapper test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_get_objective(self):
        K_func = get_objective('K')
        K_test = K_func(qp, cp.get_dofs())
        K_ans = cp.K()[:len(cp.winding_surface.quadpoints_phi)//cp.nfp]
        self.assertTrue(compare(K_test, K_ans))
        # Testing if the desc units is properly attached to at least one function
        self.assertTrue(get_objective('f_max_Bnormal2_desc_unit')({'B': 2})==4)

    def test_merge_callables(self):
        def f1(a, b):
            return 1  # Scalar
        
        def f2(a, b):
            return jnp.zeros((10, 10))  # 2D array
        
        def f3(a, b):
            return jnp.array([2, 3, 4])  # 1D array
        
        big_fn = merge_callables([f1, f2, f3])
        
        result = big_fn(0, 0)  # Call with dummy arguments
        self.assertTrue(result.shape[0]==104)

    @unittest.skipIf(not CPF_AVAILABLE, "Skipping constraint parser test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_parse_constraints(self):
        # We test the constraints using f_K and K_dot_grad_K
        f_K_ans = f_K(qp, cp.get_dofs())
        KK_ans = K_dot_grad_K(qp, cp.get_dofs())
        pert_f_K = (np.random.random()-0.5) * f_K_ans
        pert_KK = (np.random.random(KK_ans.shape)-0.5) * jnp.max(jnp.abs(KK_ans))
        unit_f_K = 114
        unit_KK = 514
        g_ineq_test, h_eq_test = parse_constraints(
            # A list of function names in quadcoil.objective.
            constraint_name = ['f_K', 'K_dot_grad_K', 'f_K', 'K_dot_grad_K'], 
            # A list of strings from ">=", "<=", "==".
            constraint_type =     ['<=', '>=', '==', '=='], 
            # A list of UNTRACED float/ints giving the constraints' order of magnitude.
            constraint_unit =     [unit_f_K, unit_KK, unit_f_K, unit_KK], 
            # An array of TRACED floats giving the constraint targets.
            constraint_value =    [f_K_ans+pert_f_K, KK_ans+pert_KK, f_K_ans+pert_f_K, KK_ans+pert_KK], 
        )
        g_ineq_test_val = g_ineq_test(qp, cp.get_dofs())
        h_eq_test_val = h_eq_test(qp, cp.get_dofs())
        # Test if the constraints are correctly scaled 
        # and centered. The threshold are set at the known 
        # value of these functions + a random offset,
        # so the correct output for the corresponding 
        # component must be - this offset.
        self.assertTrue(compare(g_ineq_test_val[0], -pert_f_K/unit_f_K))
        # Has a sign flip
        self.assertTrue(compare(g_ineq_test_val[1:], (pert_KK/unit_KK).ravel()))
        self.assertTrue(compare(h_eq_test_val[0], -pert_f_K/unit_f_K))
        self.assertTrue(compare(h_eq_test_val[1:], -(pert_KK/unit_KK).ravel()))
    
    @unittest.skipIf(not CPF_AVAILABLE, "Skipping objective parser test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_parse_objectives(self):
        self.assertTrue(compare(parse_objectives('f_B')(qp, cp.get_dofs()), f_B_normalized_by_Bnormal_IG(qp, cp.get_dofs())))
        self.assertTrue(compare(
            parse_objectives(['f_B', 'f_K'], (None, 1.), jnp.array([114, 514]))(qp, cp.get_dofs()), 
            114 * f_B_normalized_by_Bnormal_IG(qp, cp.get_dofs()) + 514 * f_K(qp, cp.get_dofs())
        ))

if __name__ == "__main__":
    unittest.main()