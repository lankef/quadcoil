import unittest
from quadcoil import get_quantity, merge_callables, parse_objectives, parse_constraints
from quadcoil.wrapper import _add_quantity
from quadcoil.quantity import f_B, f_K, K_dot_grad_K, f_B_normalized_by_Bnormal_IG, Phi
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

#     """
#     Testing the operators in f_b_and_k_operators. Thest includes:
#     - f_B_operator_and_current_scale 
#     The integrated normal field error f_B at the surface
#     - K_operator_cylindrical
#     The surface current K in a cylindrical coordinate
#     - K_operator
#     The surface current K in the xyz coordinate
#     - K_theta
#     The surface current K along the theta direction
#     """
#     @unittest.skipIf(not CPF_AVAILABLE, "Skipping objective wrapper test, simsopt.field.CurrentPotentialFourier unavailable.")
#     def test_get_quantity(self):
#         K_func = get_quantity('K')
#         K_test = K_func(qp, {'phi': cp.get_dofs()})
#         K_ans = cp.K()[:len(cp.winding_surface.quadpoints_phi)//cp.nfp]
#         self.assertTrue(compare(K_test, K_ans))
#         # Testing getting a quantity with aux constraints and vars

#     def test_merge_callables(self):
#         def f1(a, b):
#             return 1  # Scalar
        
#         def f2(a, b):
#             return jnp.zeros((10, 10))  # 2D array
        
#         def f3(a, b):
#             return jnp.array([2, 3, 4])  # 1D array
        
#         big_fn = merge_callables([f1, f2, f3, None])
        
#         result = big_fn(0, 0)  # Call with dummy arguments
#         self.assertTrue(result.shape[0]==104)

    @unittest.skipIf(not CPF_AVAILABLE, "Skipping constraint parser test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_parse_constraints(self):
        dofs = {'phi': cp.get_dofs()}
        # We test the constraints using f_K and K_dot_grad_K
        f_K_ans = f_K(qp, dofs)
        KK_ans = K_dot_grad_K(qp, dofs)
        pert_f_K = (np.random.random()-0.5) * f_K_ans
        pert_KK = (np.random.random(KK_ans.shape)-0.5) * jnp.max(jnp.abs(KK_ans))
        unit_f_K = 114
        unit_KK = 514
        g_ineq_test, h_eq_test, aux_dofs = parse_constraints(
            # A list of function names in quadcoil.objective.
            constraint_name = ['f_K', 'K_dot_grad_K', 'K_dot_grad_K'], 
            # A list of strings from ">=", "<=", "==".
            constraint_type =     ['<=', '>=', '=='], 
            # A list of UNTRACED float/ints giving the constraints' order of magnitude.
            constraint_unit =     [unit_f_K, unit_KK, unit_KK], 
            # An array of TRACED floats giving the constraint targets.
            constraint_value =    [f_K_ans+pert_f_K, KK_ans+pert_KK, KK_ans+pert_KK], 
        )
        g_ineq_test_val = merge_callables(g_ineq_test)(qp, dofs)
        h_eq_test_val = merge_callables(h_eq_test)(qp, dofs)
        # Test if the constraints are correctly scaled 
        # and centered. The threshold are set at the known 
        # value of these functions + a random offset,
        # so the correct output for the corresponding 
        # component must be - this offset.
        self.assertTrue(compare(g_ineq_test_val[0], -pert_f_K/unit_f_K))
        # Has a sign flip
        self.assertTrue(compare(g_ineq_test_val[1:], (pert_KK/unit_KK).ravel()))
        self.assertTrue(compare(h_eq_test_val, -(pert_KK/unit_KK).ravel()))

    # @unittest.skipIf(not CPF_AVAILABLE, "Skipping quantity scaling test, simsopt.field.CurrentPotentialFourier unavailable.")
    # def test_add_quantity_and_scaling(self):
    #     # Testing a simple quantity first\
    #     dofs = {'phi': cp.get_dofs(), 'max_Phi': 66.}
    #     (
    #         val_scaled,
    #         g_ineq_list_scaled,
    #         h_eq_list_scaled,
    #         unit_callable,
    #         aux_dofs
    #     ) = _add_quantity('f_B', None, 'f')
    #     self.assertTrue(compare(val_scaled(qp, dofs), f_B_normalized_by_Bnormal_IG(qp, dofs)))
    #     # Testing l-infty norm parsing
    #     test_unit=22.
    #     (
    #         val_scaled,
    #         g_ineq_list_scaled,
    #         h_eq_list_scaled,
    #         unit_callable,
    #         aux_dofs
    #     ) = _add_quantity('f_max_Phi', test_unit, 'f')
    #     print(val_scaled(qp, dofs))
    #     self.assertTrue(compare(val_scaled(qp, dofs), dofs['max_Phi']/test_unit))
        
    
    # @unittest.skipIf(not CPF_AVAILABLE, "Skipping objective parser test, simsopt.field.CurrentPotentialFourier unavailable.")
    # def test_parse_objectives(self):
    #     dofs = {'phi': cp.get_dofs()}
    #     test_weight = 1.
    #     test_unit = 1.

    #     # Testing a simple objective without aux vars and constraints
    #     f_B_test, _, _, _ = parse_objectives('f_B')
    #     self.assertTrue(compare(f_B_test(qp, dofs), f_B_normalized_by_Bnormal_IG(qp, dofs)))

    #     # Testing 2 simple objectives without aux vars and constraints
    #     f_BK_test, g_empty, h_empty, aux_empty = parse_objectives(['f_B', 'f_K'], (None, 1.), jnp.array([114, test_weight]))
    #     self.assertTrue(g_empty==[])
    #     self.assertTrue(h_empty==[])
    #     self.assertTrue(aux_empty=={})
    #     self.assertTrue(compare(
    #         f_BK_test(qp, dofs), 
    #         114 * f_B_normalized_by_Bnormal_IG(qp, dofs) + test_weight * f_K(qp, dofs)
    #     ))

    #     # Test a simple objective and a l-inf one
    #     dof1 = {'phi': cp.get_dofs(), 'max_Phi': 33}
    #     f_1_test, g_1, h_1, aux_1 = parse_objectives(
    #         ['f_B', 'f_max_Phi'], 
    #         (None, test_unit), 
    #         jnp.array([17., test_weight])
    #     )
    #     # The "behind-the-hood" of the L-inf norm function returns 
    #     # the auxiliary var.
    #     self.assertTrue(f_1_test(qp, dof1) == test_weight * 33 + 17 * f_B_normalized_by_Bnormal_IG(qp, dof1))
    #     # It creates one auxiliary constraint: 
    #     # g = [
    #     #     g_+ = ( Phi_grid - p)/unit <= 0
    #     #     g_- = (-Phi_grid - p)/unit <= 0
    #     # ]
    #     g_aux = g_1[0](qp, dof1)
    #     g_plus = g_aux[0]
    #     g_minus = g_aux[1]
    #     Phi_val = Phi(qp, dofs)
    #     self.assertTrue(len(g_1) == 1)
    #     self.assertTrue(g_aux.shape == tuple([2] + list(Phi_val.shape)))
    #     self.assertTrue(h_1 == [])
    #     self.assertTrue(compare(g_plus * test_unit + 33, Phi_val))
    #     self.assertTrue(compare(g_minus * test_unit + 33, -Phi_val))

if __name__ == "__main__":
    unittest.main()