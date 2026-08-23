"""Checks of custom JVP/grads on a curvature-constrained problem.

Uses the same formulation as examples/curvature.ipynb
(f_max_K_dot_grad_K_cyl objective with K_theta and f_B constraints),
at the test-sized harmonics mpol=4, ntor=4 used by the other JVP tests.

Random-tangent check_jvp/check_grads are not used here: f_B sits on an
active bound (true plasma derivative ~0, FD is solver noise), and many
near-zero plasma Fourier modes are non-monotone at typical FD steps.
"""
import unittest

import jax
import jax.numpy as jnp

from quadcoil import quadcoil
from quadcoil.io import gen_quadcoil_for_diff
from load_test_data import load_data


_, plasma_surface, _, _, _ = load_data()
net_poloidal_current_amperes = 11884578.094260072
MPOL = 4
NTOR = 4
METRIC_NAMES = ('f_B', 'f_K', 'f_max_K_dot_grad_K_cyl')


def _base_surface_kwargs():
    return dict(
        nfp=plasma_surface.nfp,
        stellsym=plasma_surface.stellsym,
        mpol=MPOL,
        ntor=NTOR,
        plasma_dofs=plasma_surface.get_dofs(),
        plasma_mpol=plasma_surface.mpol,
        plasma_ntor=plasma_surface.ntor,
        net_poloidal_current_amperes=net_poloidal_current_amperes,
        net_toroidal_current_amperes=0.,
        plasma_coil_distance=plasma_surface.minor_radius(),
    )


class CurvatureGradTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nescoil_out, _, _, _ = quadcoil(
            **_base_surface_kwargs(),
            objective_name='f_B',
            constraint_name=('K_theta',),
            constraint_type=('>=',),
            constraint_value=jnp.array([0.]),
            constraint_unit=(None,),
            metric_name=METRIC_NAMES,
        )
        cls.f_B_ref = nescoil_out['f_B']['value']
        f_K_ref = nescoil_out['f_K']['value']

        regcoil_out, _, _, _ = quadcoil(
            **_base_surface_kwargs(),
            objective_name='f_K',
            objective_weight=None,
            objective_unit=f_K_ref,
            constraint_name=('f_B',),
            constraint_type=('<=',),
            constraint_unit=jnp.array([cls.f_B_ref * 2]),
            constraint_value=(cls.f_B_ref * 2,),
            metric_name=METRIC_NAMES,
        )
        cls.f_max_K_dot_grad_K_cyl_ref = (
            regcoil_out['f_max_K_dot_grad_K_cyl']['value']
        )
        cls.constraint_value = jnp.array([0., cls.f_B_ref * 2])
        cls.plasma_coil_distance = plasma_surface.minor_radius()

        _, qc_diff = gen_quadcoil_for_diff(
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            mpol=MPOL,
            ntor=NTOR,
            plasma_mpol=plasma_surface.mpol,
            plasma_ntor=plasma_surface.ntor,
            objective_name='f_max_K_dot_grad_K_cyl',
            objective_weight=None,
            objective_unit=cls.f_max_K_dot_grad_K_cyl_ref,
            constraint_name=('K_theta', 'f_B'),
            constraint_type=('>=', '<='),
            constraint_unit=(None, cls.f_B_ref * 2),
            metric_name=METRIC_NAMES,
        )
        cls.qc_diff = qc_diff
        cls.G = jnp.asarray(net_poloidal_current_amperes)
        cls.I = jnp.asarray(0.)
        cls.dist = jnp.asarray(cls.plasma_coil_distance)
        cls.plasma_dofs = jnp.asarray(plasma_surface.get_dofs())

    def _metrics(self, plasma_dofs):
        out = self.qc_diff(
            plasma_dofs,
            self.G,
            self.I,
            None,
            self.dist,
            None,
            None,
            self.constraint_value,
        )
        return tuple(out[name] for name in METRIC_NAMES)

    def test_curvature_f_B_active_bound(self):
        """Active f_B constraint: custom JVP of f_B vs plasma_dofs is ~0."""
        def f_B(dofs):
            return self._metrics(dofs)[0]
        g = jax.grad(f_B)(self.plasma_dofs)
        self.assertLess(float(jnp.max(jnp.abs(g))), 1e-4)

    def test_curvature_dof0_f_K_matches_fd(self):
        """AD vs central FD of f_K on plasma_dofs[0] (well-scaled R00)."""
        def f_K(dofs):
            return self._metrics(dofs)[1]
        g_ad = float(jax.grad(f_K)(self.plasma_dofs)[0])
        e0 = jnp.zeros_like(self.plasma_dofs).at[0].set(1.0)
        step = 1e-3
        g_fd = float(
            (f_K(self.plasma_dofs + step * e0) - f_K(self.plasma_dofs - step * e0))
            / (2.0 * step)
        )
        scale = max(abs(g_ad), abs(g_fd), 1e-30)
        self.assertLess(abs(g_ad - g_fd) / scale, 0.15)


if __name__ == '__main__':
    unittest.main()
