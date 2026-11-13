"""Tests for the special conic IPM (QP / SOCP) and QP routing."""
import unittest

import jax
import jax.numpy as jnp

from quadcoil.solvers.special.qp_form import (
    ConicProblem, canonicalize_qp, canonicalize_socp, orthant_cones,
)
from quadcoil.solvers.special.ipm_conic import (
    solve_conic_ipm,
    _soc_nt_scaling,
    _soc_apply_W,
    _soc_apply_Winv,
    _soc_step_to_boundary,
    _soc_min_eig,
)
from quadcoil import quadcoil
from quadcoil.quadcoil import _quadcoil_pure
from load_test_data import load_data

_, plasma_surface, _, _, _ = load_data()
_NET_G = 11884578.094260072


def _random_qp(key, n=25, m=60):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    L = jax.random.normal(k1, (n, n))
    P = L.T @ L + 0.1 * jnp.eye(n)
    q = jax.random.normal(k2, (n,))
    G = jax.random.normal(k3, (m, n))
    x_feas = jax.random.normal(k4, (n,))
    h = G @ x_feas + 1.0
    return ConicProblem(
        P=P, q=q, G=G, h=h,
        A=jnp.zeros((0, n)), b=jnp.zeros(0),
    )


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


class CanonicalizeTest(unittest.TestCase):
    def test_exact_at_random_points(self):
        key = jax.random.PRNGKey(3)
        n = 10
        k1, k2, k3 = jax.random.split(key, 3)
        L = jax.random.normal(k1, (n, n))
        P = L.T @ L + jnp.eye(n)
        q = jax.random.normal(k2, (n,))
        G = jax.random.normal(k3, (5, n))
        h = jnp.ones(5)
        f = lambda x: 0.5 * x @ P @ x + q @ x
        g = lambda x: G @ x - h
        x0 = jax.random.normal(k1, (n,))
        prob = canonicalize_qp(f, g, x0)
        self.assertTrue(jnp.allclose(prob.P, P, atol=1e-10))
        self.assertTrue(jnp.allclose(prob.q, q, atol=1e-10))
        self.assertTrue(jnp.allclose(prob.G, G, atol=1e-10))
        self.assertTrue(jnp.allclose(prob.h, h, atol=1e-10))


class QpIpmTest(unittest.TestCase):
    def test_random_dense_qp_kkt(self):
        prob = _random_qp(jax.random.PRNGKey(5), n=30, m=100)
        res = solve_conic_ipm(
            prob, orthant_cones(prob.G.shape[0]),
            opts={'tol': 1e-8, 'maxiter': 50},
        )
        self.assertIn(int(res['status_code']), (0, 1))
        x, z, s = res['fin_x'], res['fin_z'], res['fin_s']
        stat = float(jnp.linalg.norm(prob.P @ x + prob.q + prob.G.T @ z))
        prim = float(jnp.linalg.norm(prob.G @ x + s - prob.h))
        self.assertLess(stat, 1e-5)
        self.assertLess(prim, 1e-5)
        self.assertLessEqual(float(jnp.max(prob.G @ x - prob.h)), 1e-6)

    def test_infeasible_qp(self):
        # 0 ≤ x ≤ 1 and x ≥ 2 — primal infeasible.
        P = jnp.eye(1)
        q = jnp.zeros(1)
        G = jnp.array([[1.0], [-1.0], [-1.0]])
        h = jnp.array([1.0, 0.0, -2.0])  # x<=1, -x<=0 (x>=0), -x<=-2 (x>=2)
        prob = ConicProblem(P=P, q=q, G=G, h=h, A=jnp.zeros((0, 1)), b=jnp.zeros(0))
        res = solve_conic_ipm(prob, orthant_cones(3), opts={'maxiter': 40})
        # May report infeasible (2) or fail to converge (4); must not claim solved.
        self.assertNotIn(int(res['status_code']), (0,))

    def test_degenerate_duplicate_rows(self):
        prob = _random_qp(jax.random.PRNGKey(7), n=15, m=20)
        # Duplicate first constraint row.
        G = jnp.concatenate([prob.G, prob.G[:1]], axis=0)
        h = jnp.concatenate([prob.h, prob.h[:1]])
        prob2 = ConicProblem(P=prob.P, q=prob.q, G=G, h=h, A=prob.A, b=prob.b)
        res = solve_conic_ipm(prob2, orthant_cones(G.shape[0]), opts={'maxiter': 50})
        self.assertIn(int(res['status_code']), (0, 1))

    def test_vs_clarabel_wheel(self):
        try:
            import clarabel  # noqa: F401
        except ImportError:
            self.skipTest('clarabel not installed')
        from quadcoil.solvers.special.qp_cpu import solve_conic_clarabel
        prob = _random_qp(jax.random.PRNGKey(9), n=20, m=50)
        cones = orthant_cones(prob.G.shape[0])
        r1 = solve_conic_ipm(prob, cones, opts={'tol': 1e-10, 'maxiter': 50})
        r2 = solve_conic_clarabel(prob, cones, opts={'tol': 1e-10, 'maxiter': 50})
        self.assertIn(int(r1['status_code']), (0, 1))
        self.assertIn(int(r2['status_code']), (0, 1))
        denom = max(abs(float(r2['pobj'])), 1.0)
        self.assertLess(
            abs(float(r1['pobj']) - float(r2['pobj'])) / denom, 1e-5,
        )


class SocpIpmTest(unittest.TestCase):
    def test_nt_scaling_identity(self):
        key = jax.random.PRNGKey(11)
        count, dim = 5, 4
        # Random interior points of the SOC.
        raw_s = jax.random.normal(key, (count, dim))
        raw_z = jax.random.normal(jax.random.fold_in(key, 1), (count, dim))
        s = raw_s.at[:, 0].set(jnp.linalg.norm(raw_s[:, 1:], axis=1) + 1.0)
        z = raw_z.at[:, 0].set(jnp.linalg.norm(raw_z[:, 1:], axis=1) + 1.0)
        eta, w_bar, lam = _soc_nt_scaling(s, z)
        # W z should equal W^{-1} s (= λ).
        Wz = _soc_apply_W(eta, w_bar, z)
        Winv_s = _soc_apply_Winv(eta, w_bar, s)
        self.assertTrue(jnp.allclose(Wz, lam, atol=1e-10, rtol=1e-8))
        self.assertTrue(jnp.allclose(Winv_s, lam, atol=1e-10, rtol=1e-8))

    def test_step_to_boundary_vs_bisection(self):
        # λ interior, d pointing outward: α_max from the analytic
        # formula must match a brute-force bisection on λ + α d ∈ Q.
        key = jax.random.PRNGKey(13)
        count, dim = 4, 5
        raw = jax.random.normal(key, (count, dim))
        lam = raw.at[:, 0].set(jnp.linalg.norm(raw[:, 1:], axis=1) + 2.0)
        dlam = jax.random.normal(jax.random.fold_in(key, 1), (count, dim))
        # Analytic (tau=1).
        a_an = float(_soc_step_to_boundary(lam, dlam, tau=1.0))
        # Bisection over the batch: largest α with min_eig(λ+α d) ≥ 0.
        lo, hi = 0.0, 10.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            eigs = _soc_min_eig(lam + mid * dlam)
            if float(jnp.min(eigs)) >= 0.0:
                lo = mid
            else:
                hi = mid
        self.assertAlmostEqual(a_an, lo, places=5)

    def test_simple_socp(self):
        # min ½(x²+y²) s.t. sqrt(x²+y²) ≤ 1.
        # As SOC: s ∈ Q³,  G [x;y] + s = [1; 0; 0] with G = [[0,0],[-1,0],[0,-1]].
        P = jnp.eye(2)
        q = jnp.zeros(2)
        G = jnp.array([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])
        h = jnp.array([1.0, 0.0, 0.0])
        prob = ConicProblem(P=P, q=q, G=G, h=h, A=jnp.zeros((0, 2)), b=jnp.zeros(0))
        cones = (('q', 3, 1),)
        res = solve_conic_ipm(prob, cones, opts={'tol': 1e-8, 'maxiter': 40})
        # Optimum is x=y=0.
        self.assertEqual(int(res['status_code']), 0)
        self.assertLess(float(jnp.linalg.norm(res['fin_x'])), 1e-4)

    def test_rotated_qcqp_closed_form(self):
        # min ½‖x‖²  s.t.  ½‖x − e₀‖² ≤ ½
        # Optimum: x* = 0 (constraint: ½‖−e₀‖² = ½ ≤ ½, active at boundary
        # only if we shift).  Use instead:
        #   min ½‖x‖²  s.t.  ‖x‖² ≤ 1   ⇒  x* = 0, inactive.
        # Active case: min −e₀ᵀ x + ½ ε‖x‖²  s.t. ‖x‖² ≤ 1
        # with small ε>0 for strict convexity → x* ≈ e₀.
        n = 3
        eps = 1e-2
        P = eps * jnp.eye(n)
        q = -jnp.array([1.0, 0.0, 0.0])
        # ‖x‖² ≤ 1 ⇔ ½ xᵀ (2I) x − 1 ≤ 0.  Use canonicalize_socp.
        f = lambda x: 0.5 * x @ P @ x + q @ x
        g = lambda x: jnp.array([jnp.dot(x, x) - 1.0])
        x0 = jnp.zeros(n)
        prob, cones, dual_blocks = canonicalize_socp(
            f, g, x0, g_blocks=(('quadratic', 1),),
        )
        res = solve_conic_ipm(prob, cones, opts={'tol': 1e-8, 'maxiter': 50})
        self.assertIn(int(res['status_code']), (0, 1))
        x = res['fin_x']
        # x* = e₀ / max(ε, …) projected onto the ball: for ε≪1, x*≈e₀.
        self.assertAlmostEqual(float(x[0]), 1.0, places=3)
        self.assertLess(float(jnp.linalg.norm(x[1:])), 1e-3)
        self.assertLessEqual(float(jnp.dot(x, x)), 1.0 + 1e-6)
        self.assertLess(
            float(jnp.linalg.norm(prob.P @ x + prob.q + prob.G.T @ res['fin_z'])),
            1e-5,
        )

    def test_mixed_orthant_soc_vs_clarabel(self):
        try:
            import clarabel  # noqa: F401
        except ImportError:
            self.skipTest('clarabel not installed')
        from quadcoil.solvers.special.qp_cpu import solve_conic_clarabel

        # min ½‖x‖² + 1ᵀx
        # s.t. x ≥ −1 (orthant after shift), ‖(x0,x1)‖ ≤ 2.
        n = 4
        P = jnp.eye(n)
        q = jnp.ones(n)
        # Orthant: −x ≤ 1  ⇒  G_l = −I, h_l = 1.
        # SOC: s ∈ Q³, G_q [x0;x1;…] + s = [2;0;0] with
        #      G_q = [[0,0,0,0],[-1,0,0,0],[0,-1,0,0]].
        G_l = -jnp.eye(n)
        h_l = jnp.ones(n)
        G_q = jnp.zeros((3, n)).at[1, 0].set(-1.0).at[2, 1].set(-1.0)
        h_q = jnp.array([2.0, 0.0, 0.0])
        G = jnp.concatenate([G_l, G_q], axis=0)
        h = jnp.concatenate([h_l, h_q])
        prob = ConicProblem(
            P=P, q=q, G=G, h=h,
            A=jnp.zeros((0, n)), b=jnp.zeros(0),
        )
        cones = (('l', n), ('q', 3, 1))
        r1 = solve_conic_ipm(prob, cones, opts={'tol': 1e-10, 'maxiter': 50})
        r2 = solve_conic_clarabel(prob, cones, opts={'tol': 1e-10, 'maxiter': 50})
        self.assertIn(int(r1['status_code']), (0, 1))
        self.assertIn(int(r2['status_code']), (0, 1))
        denom = max(abs(float(r2['pobj'])), 1.0)
        self.assertLess(
            abs(float(r1['pobj']) - float(r2['pobj'])) / denom, 1e-6,
        )
        x, z, s = r1['fin_x'], r1['fin_z'], r1['fin_s']
        self.assertLess(
            float(jnp.linalg.norm(prob.P @ x + prob.q + prob.G.T @ z)), 1e-5,
        )
        self.assertLess(
            float(jnp.linalg.norm(prob.G @ x + s - prob.h)), 1e-5,
        )


class TopologyQpTest(unittest.TestCase):
    def test_topology_e2e(self):
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
                enable_special=True,
                special_backend='ipm_conic',
                phi_init_with_nescoil=False,
            )
        )
        from quadcoil.quantities import K_theta
        self.assertEqual(res.get('problem_class'), 'qp')
        self.assertIn(res.get('status'), ('solved', 'solved_inaccurate'))
        self.assertGreater(float(jnp.min(K_theta(qp, dofs))), -1.0)
        self.assertTrue(jnp.isfinite(out['f_B']['value']))

    def test_no_recompile_on_constraint_value(self):
        sign = '>=' if _NET_G > 0 else '<='
        kw = _surface_kwargs(
            objective_name='f_B',
            constraint_name=('K_theta',),
            constraint_type=(sign,),
            constraint_unit=(None,),
            metric_name=('f_B',),
            value_only=True,
            enable_special=True,
            special_backend='ipm_conic',
            phi_init_with_nescoil=False,
        )
        quadcoil(**kw, constraint_value=jnp.array([0.0]))
        size0 = _quadcoil_pure._cache_size()
        quadcoil(**kw, constraint_value=jnp.array([1.0]))
        self.assertEqual(_quadcoil_pure._cache_size(), size0)


if __name__ == '__main__':
    unittest.main()
