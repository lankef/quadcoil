"""Finite-difference check of the custom JVP on a curvature-constrained problem.

Uses the same formulation as examples/curvature.ipynb (f_max_K_dot_grad_K_cyl
objective with K_theta and f_B constraints), at the test-sized harmonics
mpol=4, ntor=4 used by the other JVP tests.
"""
import functools
import unittest

import jax
import jax.numpy as jnp
from jax.test_util import check_jvp

from quadcoil import quadcoil
from quadcoil.io import gen_quadcoil_for_diff
from load_test_data import load_data


_, plasma_surface, _, _, _ = load_data()
net_poloidal_current_amperes = 11884578.094260072
MPOL = 4
NTOR = 4


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


class CurvatureJVPTest(unittest.TestCase):
    def setUp(self):
        # NESCOIL: minimize f_B subject to K_theta >= 0.
        nescoil_out, _, _, _ = quadcoil(
            **_base_surface_kwargs(),
            objective_name='f_B',
            constraint_name=('K_theta',),
            constraint_type=('>=',),
            constraint_value=jnp.array([0.]),
            constraint_unit=(None,),
            metric_name=('f_B', 'f_K', 'f_max_K_dot_grad_K_cyl'),
        )
        self.f_B_ref = nescoil_out['f_B']['value']
        f_K_ref = nescoil_out['f_K']['value']

        # REGCOIL: minimize f_K subject to f_B <= 2 * f_B_ref.
        regcoil_out, _, _, _ = quadcoil(
            **_base_surface_kwargs(),
            objective_name='f_K',
            objective_weight=None,
            objective_unit=f_K_ref,
            constraint_name=('f_B',),
            constraint_type=('<=',),
            constraint_unit=jnp.array([self.f_B_ref * 2]),
            constraint_value=(self.f_B_ref * 2,),
            metric_name=('f_B', 'f_K', 'f_max_K_dot_grad_K_cyl'),
        )
        self.f_max_K_dot_grad_K_cyl_ref = (
            regcoil_out['f_max_K_dot_grad_K_cyl']['value']
        )
        self.constraint_value = jnp.array([0., self.f_B_ref * 2])
        self.plasma_coil_distance = plasma_surface.minor_radius()

    def test_curvature_check_jvp(self):
        _, qc_diff = gen_quadcoil_for_diff(
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            mpol=MPOL,
            ntor=NTOR,
            plasma_mpol=plasma_surface.mpol,
            plasma_ntor=plasma_surface.ntor,
            objective_name='f_max_K_dot_grad_K_cyl',
            objective_weight=None,
            objective_unit=self.f_max_K_dot_grad_K_cyl_ref,
            constraint_name=('K_theta', 'f_B'),
            constraint_type=('>=', '<='),
            constraint_unit=(None, self.f_B_ref * 2),
            metric_name=('f_B', 'f_K', 'f_max_K_dot_grad_K_cyl'),
        )

        G = jnp.asarray(net_poloidal_current_amperes)
        I = jnp.asarray(0.)
        dist = jnp.asarray(self.plasma_coil_distance)
        constraint_value = self.constraint_value

        def f(plasma_dofs):
            out = qc_diff(
                plasma_dofs,
                G,
                I,
                None,  # Bnormal_plasma
                dist,
                None,  # winding_dofs
                None,  # objective_weight
                constraint_value,
            )
            return (
                out['f_B'],
                out['f_K'],
                out['f_max_K_dot_grad_K_cyl'],
            )

        plasma_dofs = jnp.asarray(plasma_surface.get_dofs())
        # Implicit auglag solve: float64 default rtol=1e-5 is too tight.
        check_jvp(
            f,
            functools.partial(jax.jvp, f),
            (plasma_dofs,),
            atol=1e-6,
            rtol=1e-2,
            eps=1e-4,
        )


if __name__ == '__main__':
    unittest.main()
