import unittest
import importlib

import quadcoil


class ModuleAliasTest(unittest.TestCase):
    def test_package_aliases_are_identical(self):
        self.assertIs(quadcoil.quantity, quadcoil.quantities)
        self.assertIs(quadcoil.solver, quadcoil.solvers)

    def test_nested_quantity_submodule_aliases(self):
        current_new = importlib.import_module('quadcoil.quantities.current')
        current_old = importlib.import_module('quadcoil.quantity.current')
        self.assertIs(current_old, current_new)

    def test_nested_solver_submodule_aliases(self):
        ipm_new = importlib.import_module('quadcoil.solvers.ipm')
        ipm_old = importlib.import_module('quadcoil.solver.ipm')
        self.assertIs(ipm_old, ipm_new)

    def test_public_symbols_reachable_both_ways(self):
        from quadcoil.quantity import K
        from quadcoil.quantities import K as K_new
        self.assertIs(K, K_new)

        from quadcoil.solver import solve_constrained_ipm
        from quadcoil.solvers import solve_constrained_ipm as solve_new
        self.assertIs(solve_constrained_ipm, solve_new)


if __name__ == '__main__':
    unittest.main()
