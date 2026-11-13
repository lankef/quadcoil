import unittest

import jax.numpy as jnp

from quadcoil import quadcoil
from quadcoil.solvers.special import classify_problem
from load_test_data import load_data


_, plasma_surface, _, _, _ = load_data()
_NET_G = 11884578.094260072


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


class SpecialClassifyTest(unittest.TestCase):
    def test_qcqp_one_detected(self):
        self.assertEqual(
            classify_problem('f_K', ('f_B',), ('<=',)),
            'qcqp_one',
        )
        self.assertEqual(
            classify_problem('f_K', 'f_B', '<='),
            'qcqp_one',
        )

    def test_multi_objective_with_quadratic_is_socp(self):
        # Multi quadratic/linear objectives + one quadratic constraint → SOCP
        # (qcqp_one requires a single objective).
        self.assertEqual(
            classify_problem(('f_B', 'f_K'), ('f_B',), ('<=',)),
            'socp',
        )

    def test_socp_detected(self):
        # Quadratic objective + linear + quadratic inequalities → SOCP.
        self.assertEqual(
            classify_problem('f_K', ('K_theta', 'f_B'), ('>=', '<=')),
            'socp',
        )
        # Two quadratic inequalities → SOCP (not qcqp_one).
        self.assertEqual(
            classify_problem('f_K', ('f_B', 'f_Phi'), ('<=', '<=')),
            'socp',
        )
        # Multi-objective of quadratic/linear terms still SOCP.
        self.assertEqual(
            classify_problem(('f_B', 'f_K'), ('K_theta', 'f_B'), ('>=', '<=')),
            'socp',
        )

    def test_non_quadratic_is_none(self):
        self.assertIsNone(
            classify_problem('f_max_Bnormal', ('f_B',), ('<=',))
        )

    def test_qp_detected(self):
        # Quadratic objective + linear inequality → QP (not qcqp_one).
        self.assertEqual(
            classify_problem('f_B', ('K_theta',), ('<=',)),
            'qp',
        )
        self.assertEqual(
            classify_problem('f_B', ('K_theta',), ('>=',)),
            'qp',
        )
        # Multi-objective of quadratic/linear terms still QP.
        self.assertEqual(
            classify_problem(('f_B', 'f_K'), ('K_theta',), ('>=',)),
            'qp',
        )

    def test_wrong_constraint_type_is_none(self):
        # Single quadratic >= is non-convex → None (also not qcqp_one).
        self.assertIsNone(
            classify_problem('f_K', ('f_B',), ('>=',))
        )
        self.assertIsNone(
            classify_problem('f_K', ('f_B',), ('==',))
        )
        # Quadratic >= among linear constraints → None.
        self.assertIsNone(
            classify_problem('f_K', ('K_theta', 'f_B'), ('>=', '>='))
        )
        # Slack quantities stay on the generic path.
        self.assertIsNone(
            classify_problem('f_K', ('f_max_Bnormal',), ('<=',))
        )

    def test_no_constraints_is_none(self):
        self.assertIsNone(classify_problem('f_B', (), ()))

    def test_quadcoil_special_path_solves(self):
        out, _qp, dofs, results = quadcoil(
            **_surface_kwargs(
                objective_name='f_K',
                constraint_name=('f_B',),
                constraint_type=('<=',),
                constraint_unit=(None,),
                constraint_value=jnp.array([1e40]),
                metric_name=('f_B', 'f_K'),
                value_only=True,
            )
        )
        self.assertEqual(results['status'], 'unconstrained')
        self.assertTrue(jnp.all(jnp.isfinite(results['fin_x'])))
        self.assertTrue(jnp.isfinite(out['f_K']['value']))
        self.assertTrue(jnp.all(jnp.isfinite(dofs['phi'])))

    def test_enable_special_false_uses_generic_path(self):
        out, _qp, dofs, results = quadcoil(
            **_surface_kwargs(
                objective_name='f_K',
                constraint_name=('f_B',),
                constraint_type=('<=',),
                constraint_unit=(None,),
                constraint_value=jnp.array([1e40]),
                metric_name=('f_B', 'f_K'),
                value_only=True,
                enable_special=False,
            )
        )
        self.assertNotIn('status', results)
        self.assertTrue(jnp.all(jnp.isfinite(results['fin_x'])))
        self.assertTrue(jnp.isfinite(out['f_K']['value']))
        self.assertTrue(jnp.all(jnp.isfinite(dofs['phi'])))


if __name__ == '__main__':
    unittest.main()
