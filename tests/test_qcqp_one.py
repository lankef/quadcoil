"""End-to-end tests for the auto-detected qcqp_one special path."""
import unittest

import jax.numpy as jnp

from quadcoil import quadcoil
from quadcoil.quadcoil import _quadcoil_pure
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


class QcqpOneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nescoil_out, _, _, _ = quadcoil(
            **_surface_kwargs(
                objective_name='f_B',
                metric_name=('f_B', 'f_K'),
                value_only=True,
            )
        )
        cls.f_B_min = float(nescoil_out['f_B']['value'])

        loose_out, _, _, loose_res = quadcoil(
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
        cls.f_B_at_min_f_K = float(loose_out['f_B']['value'])
        cls.loose_status = loose_res['status']

    def test_unconstrained_feasible(self):
        self.assertEqual(self.loose_status, 'unconstrained')
        out, _, _, results = quadcoil(
            **_surface_kwargs(
                objective_name='f_K',
                constraint_name=('f_B',),
                constraint_type=('<=',),
                constraint_unit=(None,),
                constraint_value=jnp.array([max(self.f_B_at_min_f_K * 2, 1.0)]),
                metric_name=('f_B', 'f_K'),
                value_only=True,
            )
        )
        self.assertEqual(results['status'], 'unconstrained')
        self.assertEqual(results['fin_z'].shape, (1,))
        self.assertAlmostEqual(float(results['fin_z'][0]), 0.0, places=12)
        self.assertLessEqual(float(results['fin_g'][0]), 1e-8)
        self.assertTrue(jnp.isfinite(out['f_K']['value']))

    def test_active(self):
        # Bound between min f_B and f_B(min f_K) forces an active constraint.
        bound = 0.5 * (self.f_B_min + self.f_B_at_min_f_K)
        self.assertGreater(bound, self.f_B_min)
        self.assertLess(bound, self.f_B_at_min_f_K)
        out, _, _, results = quadcoil(
            **_surface_kwargs(
                objective_name='f_K',
                constraint_name=('f_B',),
                constraint_type=('<=',),
                constraint_unit=(None,),
                constraint_value=jnp.array([bound]),
                metric_name=('f_B', 'f_K'),
                value_only=True,
            )
        )
        self.assertEqual(results['status'], 'active')
        self.assertEqual(results['fin_z'].shape, (1,))
        self.assertGreater(float(results['fin_z'][0]), 0.0)
        self.assertLess(abs(float(results['fin_g'][0])), 1e-5 * max(bound, 1.0))
        self.assertAlmostEqual(
            float(out['f_B']['value']), bound, delta=1e-4 * max(bound, 1.0),
        )

    def test_infeasible(self):
        out, _, _, results = quadcoil(
            **_surface_kwargs(
                objective_name='f_K',
                constraint_name=('f_B',),
                constraint_type=('<=',),
                constraint_unit=(None,),
                # f_B >= 0 typically, so a negative bound is impossible.
                constraint_value=jnp.array([-1.0]),
                metric_name=('f_B', 'f_K'),
                value_only=True,
            )
        )
        self.assertEqual(results['status'], 'infeasible')
        self.assertEqual(results['fin_z'].shape, (1,))
        self.assertAlmostEqual(float(results['fin_z'][0]), 0.0, places=12)
        self.assertGreater(float(results['fin_g'][0]), 0.0)
        # Solution should be near argmin g ≡ argmin f_B.
        self.assertAlmostEqual(
            float(out['f_B']['value']), self.f_B_min, delta=1e-3 * max(self.f_B_min, 1.0),
        )

    def test_gradient_smoke_unconstrained_and_active(self):
        for bound, expected in (
            (max(self.f_B_at_min_f_K * 2, 1.0), 'unconstrained'),
            (0.5 * (self.f_B_min + self.f_B_at_min_f_K), 'active'),
        ):
            out, _, _, results = quadcoil(
                **_surface_kwargs(
                    objective_name='f_K',
                    constraint_name=('f_B',),
                    constraint_type=('<=',),
                    constraint_unit=(None,),
                    constraint_value=jnp.array([bound]),
                    metric_name=('f_B', 'f_K'),
                    value_only=False,
                )
            )
            self.assertEqual(results['status'], expected)
            for name in ('f_B', 'f_K'):
                g = out[name]['grad']['df_dplasma_dofs']
                self.assertTrue(jnp.all(jnp.isfinite(g)), msg=f'{name} {expected}')

    def test_unconstrained_gradient_matches_generic(self):
        bound = max(self.f_B_at_min_f_K * 2, 1.0)
        kw = _surface_kwargs(
            objective_name='f_K',
            constraint_name=('f_B',),
            constraint_type=('<=',),
            constraint_unit=(None,),
            constraint_value=jnp.array([bound]),
            metric_name=('f_B', 'f_K'),
            value_only=False,
        )
        out_s, _, _, res_s = quadcoil(**kw, enable_special=True)
        out_g, _, _, res_g = quadcoil(**kw, enable_special=False)
        self.assertEqual(res_s['status'], 'unconstrained')
        self.assertNotIn('status', res_g)
        for name in ('f_B', 'f_K'):
            gs = out_s[name]['grad']['df_dplasma_dofs']
            gg = out_g[name]['grad']['df_dplasma_dofs']
            # Relative tolerance: KKT forms differ (z=0 constrained vs unconstrained).
            denom = jnp.maximum(jnp.max(jnp.abs(gg)), 1.0)
            err = jnp.max(jnp.abs(gs - gg)) / denom
            self.assertLess(
                float(err), 1e-4,
                msg=f'{name} grad relative err {float(err)}',
            )

    def test_no_recompile_on_constraint_value(self):
        bound0 = max(self.f_B_at_min_f_K * 2, 1.0)
        # Warm-up compile.
        quadcoil(
            **_surface_kwargs(
                objective_name='f_K',
                constraint_name=('f_B',),
                constraint_type=('<=',),
                constraint_unit=(None,),
                constraint_value=jnp.array([bound0]),
                metric_name=('f_B', 'f_K'),
                value_only=True,
            )
        )
        size0 = _quadcoil_pure._cache_size()
        # Different traced constraint_value must not grow the jit cache.
        quadcoil(
            **_surface_kwargs(
                objective_name='f_K',
                constraint_name=('f_B',),
                constraint_type=('<=',),
                constraint_unit=(None,),
                constraint_value=jnp.array([bound0 * 1.01]),
                metric_name=('f_B', 'f_K'),
                value_only=True,
            )
        )
        self.assertEqual(_quadcoil_pure._cache_size(), size0)


if __name__ == '__main__':
    unittest.main()
