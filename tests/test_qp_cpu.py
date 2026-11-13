"""Tests for the Clarabel CPU backend (pure_callback / optional FFI)."""
import unittest

import jax
import jax.numpy as jnp

from quadcoil.solvers.special.qp_form import ConicProblem, orthant_cones
from quadcoil.solvers.special.ipm_conic import solve_conic_ipm
from quadcoil.solvers.special.qp_cpu import (
    solve_conic_clarabel, clarabel_available,
)
from quadcoil import quadcoil
from quadcoil.quadcoil import _quadcoil_pure
from load_test_data import load_data

_, plasma_surface, _, _, _ = load_data()
_NET_G = 11884578.094260072


def _skip_if_no_clarabel():
    if not clarabel_available():
        raise unittest.SkipTest('clarabel not available')


def _random_qp(key, n=20, m=40):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    L = jax.random.normal(k1, (n, n))
    P = L.T @ L + 0.1 * jnp.eye(n)
    q = jax.random.normal(k2, (n,))
    G = jax.random.normal(k3, (m, n))
    h = G @ jax.random.normal(k4, (n,)) + 1.0
    return ConicProblem(
        P=P, q=q, G=G, h=h,
        A=jnp.zeros((0, n)), b=jnp.zeros(0),
    )


class ClarabelBackendTest(unittest.TestCase):
    def test_agrees_with_ipm_conic(self):
        _skip_if_no_clarabel()
        prob = _random_qp(jax.random.PRNGKey(21))
        cones = orthant_cones(prob.G.shape[0])
        r_ipm = solve_conic_ipm(prob, cones, opts={'tol': 1e-10})
        r_cl = solve_conic_clarabel(prob, cones, opts={'tol': 1e-10})
        self.assertIn(int(r_ipm['status_code']), (0, 1))
        self.assertIn(int(r_cl['status_code']), (0, 1))
        denom = max(abs(float(r_cl['pobj'])), 1.0)
        self.assertLess(
            abs(float(r_ipm['pobj']) - float(r_cl['pobj'])) / denom, 1e-5,
        )

    def test_infeasible_status(self):
        _skip_if_no_clarabel()
        P = jnp.eye(1)
        q = jnp.zeros(1)
        G = jnp.array([[1.0], [-1.0], [-1.0]])
        h = jnp.array([1.0, 0.0, -2.0])
        prob = ConicProblem(P=P, q=q, G=G, h=h, A=jnp.zeros((0, 1)), b=jnp.zeros(0))
        res = solve_conic_clarabel(prob, orthant_cones(3))
        self.assertEqual(int(res['status_code']), 2)

    def test_no_recompile_on_data_change(self):
        _skip_if_no_clarabel()
        cones = orthant_cones(30)

        @jax.jit
        def solve(P, q, G, h):
            prob = ConicProblem(
                P=P, q=q, G=G, h=h,
                A=jnp.zeros((0, P.shape[0])), b=jnp.zeros(0),
            )
            return solve_conic_clarabel(prob, cones)

        prob = _random_qp(jax.random.PRNGKey(0), n=10, m=30)
        solve(prob.P, prob.q, prob.G, prob.h)
        # Second call with different data — same shapes — must not recompile.
        # (We can't easily read pure_callback cache; just check it runs.)
        prob2 = _random_qp(jax.random.PRNGKey(1), n=10, m=30)
        r = solve(prob2.P, prob2.q, prob2.G, prob2.h)
        self.assertTrue(jnp.isfinite(r['fin_x']).all())

    def test_topology_via_quadcoil(self):
        _skip_if_no_clarabel()
        sign = '>=' if _NET_G > 0 else '<='
        out, qp, dofs, res = quadcoil(
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            plasma_mpol=plasma_surface.mpol,
            plasma_ntor=plasma_surface.ntor,
            plasma_dofs=plasma_surface.get_dofs(),
            net_poloidal_current_amperes=_NET_G,
            mpol=4, ntor=4,
            plasma_coil_distance=plasma_surface.minor_radius(),
            objective_name='f_B',
            constraint_name=('K_theta',),
            constraint_type=(sign,),
            constraint_value=jnp.array([0.]),
            constraint_unit=(None,),
            metric_name=('f_B',),
            value_only=True,
            enable_special=True,
            special_backend='clarabel',
            phi_init_with_nescoil=False,
        )
        self.assertEqual(res.get('problem_class'), 'qp')
        self.assertIn(res.get('status'), ('solved', 'solved_inaccurate'))
        self.assertTrue(jnp.isfinite(out['f_B']['value']))

    def test_no_recompile_quadcoil_constraint_value(self):
        _skip_if_no_clarabel()
        sign = '>=' if _NET_G > 0 else '<='
        kw = dict(
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            plasma_mpol=plasma_surface.mpol,
            plasma_ntor=plasma_surface.ntor,
            plasma_dofs=plasma_surface.get_dofs(),
            net_poloidal_current_amperes=_NET_G,
            mpol=4, ntor=4,
            plasma_coil_distance=plasma_surface.minor_radius(),
            objective_name='f_B',
            constraint_name=('K_theta',),
            constraint_type=(sign,),
            constraint_unit=(None,),
            metric_name=('f_B',),
            value_only=True,
            enable_special=True,
            special_backend='clarabel',
            phi_init_with_nescoil=False,
        )
        quadcoil(**kw, constraint_value=jnp.array([0.0]))
        size0 = _quadcoil_pure._cache_size()
        quadcoil(**kw, constraint_value=jnp.array([1.0]))
        self.assertEqual(_quadcoil_pure._cache_size(), size0)


if __name__ == '__main__':
    unittest.main()
