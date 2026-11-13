"""End-to-end tests for the auto-detected SOCP special path (REGCOIL)."""
import unittest

import jax.numpy as jnp

from quadcoil import quadcoil
from quadcoil.quadcoil import _quadcoil_pure
from quadcoil.quantities import K_theta
from quadcoil.solvers.special import classify_problem
from load_test_data import load_data


_, plasma_surface, _, _, _ = load_data()
_NET_G = 11884578.094260072
_K_SIGN = '>=' if _NET_G > 0 else '<='


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


class SocpRegcoilTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Topology QP (filament §2.2) → f_B_target = 2 f_B_topo, as in the notebook.
        out_topo, _, _, _ = quadcoil(
            **_surface_kwargs(
                objective_name='f_B',
                constraint_name=('K_theta',),
                constraint_type=(_K_SIGN,),
                constraint_value=jnp.array([0.0]),
                constraint_unit=(None,),
                metric_name=('f_B', 'f_K'),
                value_only=True,
                enable_special=True,
                special_backend='ipm_conic',
            )
        )
        cls.f_B_target = 2.0 * float(out_topo['f_B']['value'])
        cls.f_K_ref = float(out_topo['f_K']['value'])

        cls.regcoil_kw = _surface_kwargs(
            objective_name='f_K',
            objective_unit=cls.f_K_ref,
            constraint_name=('K_theta', 'f_B'),
            constraint_type=(_K_SIGN, '<='),
            constraint_value=jnp.array([0.0, cls.f_B_target]),
            constraint_unit=(None, cls.f_B_target),
            metric_name=('f_B', 'f_K'),
        )

    def test_classifies_as_socp(self):
        self.assertEqual(
            classify_problem('f_K', ('K_theta', 'f_B'), (_K_SIGN, '<=')),
            'socp',
        )

    def test_solves_and_feasible(self):
        out, qp, dofs, res = quadcoil(
            **self.regcoil_kw,
            value_only=True,
            enable_special=True,
            special_backend='ipm_conic',
        )
        self.assertEqual(res.get('problem_class'), 'socp')
        self.assertIn(res.get('status'), ('solved', 'solved_inaccurate'))
        self.assertTrue(jnp.isfinite(out['f_K']['value']))
        self.assertLessEqual(
            float(out['f_B']['value']),
            self.f_B_target * (1.0 + 1e-5),
        )
        self.assertGreaterEqual(float(jnp.min(K_theta(qp, dofs))), -1e-3)

    def test_matches_generic_ipm(self):
        out_s, _, _, res_s = quadcoil(
            **self.regcoil_kw,
            value_only=True,
            enable_special=True,
            special_backend='ipm_conic',
        )
        out_g, _, _, res_g = quadcoil(
            **self.regcoil_kw,
            value_only=True,
            enable_special=False,
            solver='ipm',
            convex=False,
            smoothing=None,
            maxiter=100,
        )
        f_K_s = float(out_s['f_K']['value'])
        f_K_g = float(out_g['f_K']['value'])
        denom = max(abs(f_K_g), 1.0)
        self.assertLess(abs(f_K_s - f_K_g) / denom, 1e-4)
        self.assertIn(res_s.get('status'), ('solved', 'solved_inaccurate'))

    def test_dfdy_agrees_with_generic_ipm(self):
        out_s, _, _, _ = quadcoil(
            **self.regcoil_kw,
            value_only=False,
            enable_special=True,
            special_backend='ipm_conic',
        )
        out_g, _, _, _ = quadcoil(
            **self.regcoil_kw,
            value_only=False,
            enable_special=False,
            solver='ipm',
            convex=False,
            smoothing=None,
            maxiter=100,
        )
        for name in ('f_B', 'f_K'):
            gs = out_s[name]['grad']['df_dplasma_dofs']
            gg = out_g[name]['grad']['df_dplasma_dofs']
            denom = jnp.maximum(jnp.max(jnp.abs(gg)), 1.0)
            err = jnp.max(jnp.abs(gs - gg)) / denom
            self.assertLess(
                float(err), 5e-3,
                msg=f'{name} grad relative err {float(err)}',
            )

    def test_no_recompile_on_constraint_value(self):
        kw = dict(self.regcoil_kw)
        kw.update(
            value_only=True,
            enable_special=True,
            special_backend='ipm_conic',
            constraint_unit=(None, None),
        )
        kw.pop('constraint_value')
        cv0 = jnp.array([0.0, self.f_B_target])
        cv1 = jnp.array([0.0, self.f_B_target * 1.1])
        quadcoil(**kw, constraint_value=cv0)
        size0 = _quadcoil_pure._cache_size()
        quadcoil(**kw, constraint_value=cv1)
        self.assertEqual(_quadcoil_pure._cache_size(), size0)


if __name__ == '__main__':
    unittest.main()
