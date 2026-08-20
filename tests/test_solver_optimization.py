"""
Compare optimized solvers against legacy versions.

Runs both auglag and IPM solvers (optimized vs legacy) on the same problem
and verifies that objective values, constraint satisfaction, and gradients
are consistent.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

import numpy as np

# Optimized solvers (current)
from quadcoil.solver.ipm import (
    solve_constrained_ipm as solve_constrained_ipm_opt,
    solve_unconstrained_ipm as solve_unconstrained_ipm_opt,
)
from quadcoil.solver.kkt_adjoint import (
    adjoint_kkt as adjoint_auglag_lbfgs_opt,
)

# Legacy solvers
from quadcoil.solver.ipm_legacy import (
    solve_constrained_ipm as solve_constrained_ipm_leg,
    solve_unconstrained_ipm as solve_unconstrained_ipm_leg,
)
from quadcoil.solver.auglag_legacy import (
    adjoint_auglag_lbfgs as adjoint_auglag_lbfgs_leg,
)


def quadratic_f(x):
    """Simple convex quadratic: 0.5 * x^T H x + c^T x."""
    n = x.shape[0]
    H = jnp.eye(n) + 0.1 * jnp.ones((n, n))
    c = jnp.linspace(-1.0, 1.0, n)
    return 0.5 * x @ H @ x + c @ x


def rosenbrock_f(x):
    """Non-convex Rosenbrock in n dimensions."""
    return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)


def linear_ineq(x):
    """g(x) = sum(x) - 1 <= 0."""
    return jnp.array([jnp.sum(x) - 1.0])


class TestIPMOptimization(unittest.TestCase):
    """Compare optimized IPM against legacy."""

    def test_unconstrained_convex(self):
        n = 20
        x0 = jnp.zeros(n)
        res_opt = solve_unconstrained_ipm_opt(x0, quadratic_f, convex=True)
        res_leg = solve_unconstrained_ipm_leg(x0, quadratic_f, convex=False)
        rtol = 1e-4
        np.testing.assert_allclose(
            float(res_opt['fin_f']), float(res_leg['fin_f']), rtol=rtol,
            err_msg="Unconstrained IPM: objective mismatch",
        )
        np.testing.assert_allclose(
            np.array(res_opt['fin_x']), np.array(res_leg['fin_x']), rtol=rtol,
            err_msg="Unconstrained IPM: solution mismatch",
        )
        print(f"  unconstrained convex: f_opt={res_opt['fin_f']:.8f}, "
              f"f_leg={res_leg['fin_f']:.8f}")

    def test_constrained_convex(self):
        n = 20
        x0 = jnp.zeros(n)
        opts = {'max_ipm_iter': 200, 'tol_kkt': 1e-6}
        res_opt = solve_constrained_ipm_opt(
            x0, quadratic_f, g_ineq=linear_ineq, convex=True,
            solver_options=opts, verbose=0,
        )
        res_leg = solve_constrained_ipm_leg(
            x0, quadratic_f, g_ineq=linear_ineq, convex=False,
            solver_options=opts, verbose=0,
        )
        rtol = 1e-3
        np.testing.assert_allclose(
            float(res_opt['fin_f']), float(res_leg['fin_f']), rtol=rtol,
            err_msg="Constrained IPM: objective mismatch",
        )
        # Both should satisfy the constraint
        self.assertLessEqual(float(jnp.max(res_opt['fin_g'])), 1e-4,
                             "Optimized IPM violates constraint")
        self.assertLessEqual(float(jnp.max(res_leg['fin_g'])), 1e-4,
                             "Legacy IPM violates constraint")
        print(f"  constrained convex: f_opt={res_opt['fin_f']:.8f} "
              f"(g={float(jnp.max(res_opt['fin_g'])):.2e}), "
              f"f_leg={res_leg['fin_f']:.8f} "
              f"(g={float(jnp.max(res_leg['fin_g'])):.2e})")

    def test_constrained_nonconvex(self):
        n = 10
        x0 = jnp.zeros(n)
        g_ineq = lambda x: jnp.array([jnp.sum(x) - 2.0])
        opts = {'max_ipm_iter': 200, 'tol_kkt': 1e-5}
        res_opt = solve_constrained_ipm_opt(
            x0, rosenbrock_f, g_ineq=g_ineq, convex=False,
            solver_options=opts, verbose=0,
        )
        res_leg = solve_constrained_ipm_leg(
            x0, rosenbrock_f, g_ineq=g_ineq, convex=False,
            solver_options=opts, verbose=0,
        )
        rtol = 1e-2
        np.testing.assert_allclose(
            float(res_opt['fin_f']), float(res_leg['fin_f']), rtol=rtol,
            err_msg="Nonconvex constrained IPM: objective mismatch",
        )
        print(f"  constrained nonconvex: f_opt={res_opt['fin_f']:.8f}, "
              f"f_leg={res_leg['fin_f']:.8f}")


class TestAuglagOptimization(unittest.TestCase):
    """Compare optimized auglag adjoint against legacy."""

    def _build_stationarity_data(self, stationarity_fn, n=10):
        """Build a minimal stationarity_data dict for the constrained case."""
        from quadcoil.solver.auglag import (
            solve_constrained_auglag_lbfgs, gplus_hard,
        )
        from quadcoil.solver.kkt_adjoint import stationarity_kkt
        x0 = jnp.zeros(n)
        g_ineq = lambda x: jnp.array([jnp.sum(x) - 1.0])
        h_eq = lambda x: jnp.zeros(0)

        result = solve_constrained_auglag_lbfgs(
            x0, quadratic_f, h_eq=h_eq, g_ineq=g_ineq,
            convex=True, verbose=0,
        )
        # For the adjoint test, we need stationarity_data from the same solver
        return result, stationarity_fn

    def test_adjoint_constrained(self):
        """Verify optimized adjoint matches legacy for a constrained problem."""
        from quadcoil.solver.kkt_adjoint import stationarity_kkt
        from quadcoil.solver.auglag import recover_multipliers
        from quadcoil.solver.auglag_legacy import (
            stationarity_auglag_lbfgs as stationarity_leg,
            adjoint_auglag_lbfgs as adjoint_leg,
        )
        import lineax as lx

        n = 10
        x0 = jnp.zeros(n)
        g_ineq = lambda x: jnp.array([jnp.sum(x) - 1.0])
        h_eq = lambda x: jnp.zeros(0)
        f_obj = quadratic_f

        from quadcoil.solver.auglag import solve_constrained_auglag_lbfgs
        solve_res = solve_constrained_auglag_lbfgs(
            x0, f_obj, h_eq=h_eq, g_ineq=g_ineq, convex=True, verbose=0,
        )

        # Build y_flat and helpers for stationarity
        y_flat = jnp.array([1.0])  # dummy parameter

        def f_g_h_from_y(y_dict):
            return f_obj, g_ineq, h_eq, 1, 0, {}

        unravel_y = lambda y: {'dummy': y[0]}
        unravel_unscale_x = lambda x: x

        solver_options = {'svtol': 1e-6}

        x_opt = solve_res['fin_x']
        z_ineq, z_eq = recover_multipliers(
            x_opt, y_flat, f_g_h_from_y, unravel_y, unravel_unscale_x,
        )
        z_opt = jnp.concatenate([z_ineq, z_eq])

        def f_g_combined_from_y(y_dict):
            f_obj_i, g_ineq_i, h_eq_i, n_g_i, n_h_i, aux = f_g_h_from_y(y_dict)
            g_combined = lambda dofs: jnp.concatenate([g_ineq_i(dofs), h_eq_i(dofs)])
            h_empty = lambda dofs: jnp.zeros(0)
            return f_obj_i, g_combined, h_empty, n_g_i + n_h_i, 0, aux

        stat_opt = stationarity_kkt(
            constrained=True,
            x_opt=x_opt, z_opt=z_opt,
            y_flat=y_flat, f_g_ineq_h_eq_from_y=f_g_combined_from_y,
            unravel_y=unravel_y, flat_x_to_dofs=unravel_unscale_x,
            verbose=0,
        )
        stat_leg = stationarity_leg(
            constrained=True, convex=True, solve_results=solve_res,
            y_flat=y_flat, f_g_ineq_h_eq_from_y=f_g_h_from_y,
            unravel_y=unravel_y, unravel_unscale_x=unravel_unscale_x,
            solver_options=solver_options, verbose=0,
        )

        f_metric = lambda x, y: f_obj(x)
        f_metrics_flat = lambda x, y: jnp.atleast_1d(f_obj(x)).ravel()
        solver = lx.AutoLinearSolver(well_posed=False)

        m_opt, dfdy_opt, info_opt = adjoint_auglag_lbfgs_opt(
            f_metrics_flat, stat_opt, y_flat, verbose=1,
        )
        m_leg, dfdy_leg, info_leg = adjoint_leg(
            f_metric, stat_leg, y_flat, solver, verbose=1,
        )

        np.testing.assert_allclose(
            float(m_opt[0]), float(m_leg), rtol=1e-6,
            err_msg="Auglag adjoint: metric value mismatch",
        )
        # Batched adjoint returns (K, n_y); legacy returns (n_y,) for scalars.
        np.testing.assert_allclose(
            np.array(dfdy_opt[0]), np.array(dfdy_leg), rtol=1e-3,
            err_msg="Auglag adjoint: gradient mismatch",
        )
        print(f"  auglag adjoint: m_opt={float(m_opt[0]):.8f}, m_leg={float(m_leg):.8f}, "
              f"dfdy close: {np.allclose(np.array(dfdy_opt[0]), np.array(dfdy_leg), rtol=1e-3)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
