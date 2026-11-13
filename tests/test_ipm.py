"""Unit and end-to-end tests for the generic nonlinear IPM."""
import unittest

import jax
import jax.numpy as jnp

from quadcoil.solvers.ipm import solve_constrained_ipm
from quadcoil import quadcoil
from quadcoil.quadcoil import _quadcoil_pure
from load_test_data import load_data


_, plasma_surface, _, _, _ = load_data()
_NET_G = 11884578.094260072


def _random_convex_qp(key, n=20, m=40):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    L = jax.random.normal(k1, (n, n))
    P = L.T @ L + 0.1 * jnp.eye(n)
    q = jax.random.normal(k2, (n,))
    G = jax.random.normal(k3, (m, n))
    x_feas = jax.random.normal(k4, (n,))
    h = G @ x_feas + 1.0
    f = lambda x, P=P, q=q: 0.5 * x @ P @ x + q @ x
    g = lambda x, G=G, h=h: G @ x - h
    return f, g, P, q, G, h


def _surface_kwargs(**extra):
    kw = dict(
        nfp=plasma_surface.nfp,
        stellsym=plasma_surface.stellsym,
        plasma_mpol=plasma_surface.mpol,
        plasma_ntor=plasma_surface.ntor,
        plasma_dofs=plasma_surface.get_dofs(),
        net_poloidal_current_amperes=_NET_G,
        mpol=4,
        ntor=4,
        plasma_coil_distance=plasma_surface.minor_radius(),
    )
    kw.update(extra)
    return kw


class IpmUnitTest(unittest.TestCase):
    def test_random_convex_qp_kkt(self):
        f, g, P, q, G, h = _random_convex_qp(jax.random.PRNGKey(1))
        # Infeasible start (zeros).
        x0 = jnp.zeros(P.shape[0])
        self.assertGreater(float(jnp.max(g(x0))), 0.0)
        res = solve_constrained_ipm(
            x0, f, g_ineq=g, convex=True, maxiter=80,
        )
        self.assertTrue(bool(res['converged']))
        self.assertEqual(int(res['status_code']), 0)
        x, z, s = res['fin_x'], res['fin_z'], res['fin_s']
        stat = float(jnp.linalg.norm(P @ x + q + G.T @ z))
        prim = float(jnp.linalg.norm(G @ x + s - h))
        comp = float(jnp.linalg.norm(s * z))
        self.assertLess(stat, 1e-4)
        self.assertLess(prim, 1e-4)
        self.assertLess(comp, 1e-4)
        self.assertLessEqual(float(jnp.max(res['fin_g'])), 1e-5)

    def test_infeasible_start_converges(self):
        f, g, P, q, G, h = _random_convex_qp(jax.random.PRNGKey(2), n=15, m=30)
        # Far outside the feasible set.
        x0 = 10.0 * jnp.ones(P.shape[0])
        self.assertGreater(float(jnp.max(g(x0))), 1.0)
        res = solve_constrained_ipm(
            x0, f, g_ineq=g, convex=True, maxiter=100,
        )
        self.assertTrue(bool(res['converged']))
        self.assertLessEqual(float(jnp.max(res['fin_g'])), 1e-5)

    def test_small_nonconvex_nlp(self):
        # min (x^2 - 1)^2  s.t.  x <= 0.5.  Both unconstrained local
        # minima (±1) are feasible; start near −1 so the IPM finds that
        # basin.  The point is to exercise the nonconvex (indefinite-
        # Hessian) path, not to claim global optimality.
        f = lambda x: (x[0] ** 2 - 1.0) ** 2
        g = lambda x: jnp.array([x[0] - 0.5])
        res = solve_constrained_ipm(
            jnp.array([-0.8]), f, g_ineq=g, convex=False, maxiter=80,
        )
        self.assertTrue(
            bool(res['converged']) or float(res['fin_kkt_res']) < 1e-3,
            msg=f"kkt={float(res['fin_kkt_res'])} status={int(res['status_code'])}",
        )
        self.assertLessEqual(float(res['fin_x'][0]), 0.5 + 1e-5)
        self.assertTrue(jnp.isfinite(res['fin_x']).all())

    def test_h_eq_raises(self):
        f = lambda x: jnp.sum(x ** 2)
        g = lambda x: x - 1.0
        h = lambda x: jnp.array([x[0]])
        with self.assertRaises(NotImplementedError):
            solve_constrained_ipm(
                jnp.zeros(3), f, h_eq=h, g_ineq=g,
            )


class IpmQuadcoilTest(unittest.TestCase):
    def test_topology_smoothing_none(self):
        sign = '>=' if _NET_G > 0 else '<='
        out, qp, dofs, res = quadcoil(
            **_surface_kwargs(
                objective_name='f_B',
                constraint_name=('K_theta',),
                constraint_type=(sign,),
                constraint_value=jnp.array([0.]),
                constraint_unit=(None,),
                metric_name=('f_B',),
                value_only=True,
                enable_special=False,
                solver='ipm',
                convex=True,
                smoothing=None,
                maxiter=80,
                phi_init_with_nescoil=False,
            )
        )
        from quadcoil.quantities import K_theta
        self.assertTrue(jnp.isfinite(out['f_B']['value']))
        self.assertGreater(float(jnp.min(K_theta(qp, dofs))), -1e3)
        # Should make progress toward feasibility / optimality.
        self.assertTrue(
            bool(res.get('converged', False))
            or float(res['fin_kkt_res']) < 1e-2
        )

    def test_no_recompile_on_constraint_value(self):
        sign = '>=' if _NET_G > 0 else '<='
        kw = _surface_kwargs(
            objective_name='f_B',
            constraint_name=('K_theta',),
            constraint_type=(sign,),
            constraint_unit=(None,),
            metric_name=('f_B',),
            value_only=True,
            enable_special=False,
            solver='ipm',
            convex=True,
            smoothing=None,
            maxiter=20,
            phi_init_with_nescoil=False,
        )
        quadcoil(**kw, constraint_value=jnp.array([0.0]))
        size0 = _quadcoil_pure._cache_size()
        quadcoil(**kw, constraint_value=jnp.array([1.0]))
        self.assertEqual(_quadcoil_pure._cache_size(), size0)


if __name__ == '__main__':
    unittest.main()
