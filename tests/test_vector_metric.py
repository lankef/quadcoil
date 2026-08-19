import unittest
from quadcoil import quadcoil
from quadcoil.io import gen_quadcoil_for_diff
import jax
import jax.numpy as jnp
from load_test_data import load_data

_, plasma_surface, _, _, _ = load_data()
net_poloidal_current_amperes = 11884578.094260072


def _base_kwargs(**overrides):
    kwargs = dict(
        nfp=plasma_surface.nfp,
        stellsym=plasma_surface.stellsym,
        mpol=4,
        ntor=4,
        plasma_dofs=plasma_surface.get_dofs(),
        plasma_mpol=plasma_surface.mpol,
        plasma_ntor=plasma_surface.ntor,
        net_poloidal_current_amperes=net_poloidal_current_amperes,
        net_toroidal_current_amperes=0.,
        plasma_coil_distance=plasma_surface.minor_radius(),
        objective_name='f_B',
        metric_name=('f_B', 'f_K'),
        maxiter=200,
    )
    kwargs.update(overrides)
    return kwargs


class VectorMetricTest(unittest.TestCase):
    def test_empty_metric_name(self):
        for value_only in (True, False):
            out_dict, _, _, _ = quadcoil(
                **_base_kwargs(metric_name=(), value_only=value_only)
            )
            self.assertEqual(out_dict, {})

        out_dict, _, _, _ = quadcoil(
            **_base_kwargs(metric_name=None, value_only=True)
        )
        self.assertEqual(out_dict, {})

    def test_phi_dofs_value_and_shape(self):
        out_dict, _, dofs_opt, _ = quadcoil(
            **_base_kwargs(metric_name=('phi_dofs',))
        )
        self.assertIn('phi_dofs', out_dict)
        phi = dofs_opt['phi']
        self.assertTrue(
            jnp.allclose(out_dict['phi_dofs']['value'], phi),
            'phi_dofs value must equal dofs_opt["phi"]',
        )
        grad_plasma = out_dict['phi_dofs']['grad']['df_dplasma_dofs']
        self.assertEqual(
            grad_plasma.shape,
            (len(phi), len(plasma_surface.get_dofs())),
        )

    def test_phi_dofs_fd_net_poloidal_current(self):
        out_dict, _, dofs_opt, _ = quadcoil(
            **_base_kwargs(metric_name=('phi_dofs',))
        )
        dphi_dG = out_dict['phi_dofs']['grad'][
            'df_dnet_poloidal_current_amperes'
        ]
        # Finite difference of phi w.r.t. net poloidal current.
        eps = 1e-4 * abs(net_poloidal_current_amperes)
        _, _, dofs_plus, _ = quadcoil(
            **_base_kwargs(
                metric_name=(),
                value_only=True,
                net_poloidal_current_amperes=net_poloidal_current_amperes + eps,
            )
        )
        _, _, dofs_minus, _ = quadcoil(
            **_base_kwargs(
                metric_name=(),
                value_only=True,
                net_poloidal_current_amperes=net_poloidal_current_amperes - eps,
            )
        )
        dphi_dG_fd = (dofs_plus['phi'] - dofs_minus['phi']) / (2 * eps)
        # Relative error against the larger of FD / adjoint scale.
        scale = jnp.maximum(jnp.max(jnp.abs(dphi_dG_fd)), jnp.max(jnp.abs(dphi_dG)))
        rel_err = jnp.max(jnp.abs(dphi_dG - dphi_dG_fd)) / scale
        self.assertLess(
            float(rel_err), 5e-2,
            f'd(phi)/dG adjoint vs FD relative error too large: {rel_err}',
        )

    def test_rank2_metric_gradient_shape(self):
        """Bnormal is (nphi, ntheta); grad leaves must be (*metric, *param)."""
        nfp = plasma_surface.nfp
        # Tiny grid so jacrev of the 2-D field stays in memory.
        qp_phi = jnp.linspace(0., 1. / nfp, 3, endpoint=False)
        qp_theta = jnp.linspace(0., 1., 3, endpoint=False)
        out_dict, _, _, _ = quadcoil(
            **_base_kwargs(
                metric_name=('Bnormal',),
                mpol=2,
                ntor=2,
                plasma_quadpoints_phi=qp_phi,
                plasma_quadpoints_theta=qp_theta,
                winding_quadpoints_phi=qp_phi,
                winding_quadpoints_theta=qp_theta,
                maxiter=50,
            )
        )
        self.assertIn('Bnormal', out_dict)
        val = out_dict['Bnormal']['value']
        self.assertEqual(val.ndim, 2)
        grad_plasma = out_dict['Bnormal']['grad']['df_dplasma_dofs']
        plasma_dofs = plasma_surface.get_dofs()
        self.assertEqual(
            grad_plasma.shape,
            val.shape + (len(plasma_dofs),),
        )
        grad_G = out_dict['Bnormal']['grad'][
            'df_dnet_poloidal_current_amperes'
        ]
        self.assertEqual(grad_G.shape, val.shape)

    def test_scalar_metric_regression_shape(self):
        """Scalar metrics keep scalar values and param-shaped grads."""
        out_dict, _, _, _ = quadcoil(
            **_base_kwargs(metric_name=('f_B',))
        )
        self.assertEqual(jnp.ndim(out_dict['f_B']['value']), 0)
        grad_plasma = out_dict['f_B']['grad']['df_dplasma_dofs']
        self.assertEqual(
            grad_plasma.shape,
            (len(plasma_surface.get_dofs()),),
        )

    def test_gen_quadcoil_for_diff_jvp_scalar(self):
        """custom_jvp contracts scalar metric grads correctly."""
        _, qc_diff = gen_quadcoil_for_diff(
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            mpol=4,
            ntor=4,
            plasma_mpol=plasma_surface.mpol,
            plasma_ntor=plasma_surface.ntor,
            objective_name='f_B',
            metric_name=('f_B',),
            maxiter=200,
        )
        plasma_dofs = jnp.array(plasma_surface.get_dofs())
        G = jnp.asarray(net_poloidal_current_amperes)
        I = jnp.asarray(0.)
        Bnormal_plasma = None
        dist = jnp.asarray(plasma_surface.minor_radius())
        winding_dofs = None
        objective_weight = jnp.asarray(1.)
        constraint_value = jnp.array([])

        primals = (
            plasma_dofs, G, I, Bnormal_plasma, dist,
            winding_dofs, objective_weight, constraint_value,
        )
        # Tangents must match the primal pytree; perturb only G.
        tangents = (
            jnp.zeros_like(plasma_dofs),
            jnp.asarray(1.),
            jnp.zeros_like(I),
            None,
            jnp.zeros_like(dist),
            None,
            jnp.zeros_like(objective_weight),
            jnp.zeros_like(constraint_value),
        )
        out_primal, out_dot = jax.jvp(qc_diff, primals, tangents)
        self.assertIn('f_B', out_primal)
        self.assertEqual(jnp.ndim(out_primal['f_B']), 0)
        self.assertEqual(jnp.ndim(out_dot['f_B']), 0)

        # Cross-check against adjoint grad * tangent.
        out_full, _, _, _ = quadcoil(
            **_base_kwargs(metric_name=('f_B',))
        )
        expected = out_full['f_B']['grad']['df_dnet_poloidal_current_amperes']
        self.assertTrue(
            jnp.allclose(out_dot['f_B'], expected, rtol=1e-5, atol=1e-8),
            f'JVP {out_dot["f_B"]} != adjoint grad {expected}',
        )


if __name__ == '__main__':
    unittest.main()
