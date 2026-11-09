'''Cached phi-independent operators must match on-the-fly evaluation.

The cache changes where the operator is built, not the value. Values, first
and second derivatives with respect to the current potential, and first
derivatives with respect to the winding and plasma surfaces must agree,
with and without evaluation-point chunking.
'''
import unittest

import jax
import jax.numpy as jnp
from jax import jacrev, tree_util

jax.config.update('jax_enable_x64', True)

from quadcoil import QuadcoilParams
from quadcoil.quantities import f_B, f_max_force_cyl
from quadcoil.quantities.force import _force_cyl
from quadcoil.quantities.magnetic_field import _Bnormal, _winding_surface_B
from quadcoil.quantities.operators import build_cached_ops
from quadcoil.quantities.self_field import _B_self
from load_test_data import load_data

_, _, _, _, QP = load_data()
PHI = jnp.asarray(jax.random.normal(jax.random.PRNGKey(0), (QP.ndofs,)))

# Array-valued kernels and the two scalar quantities used as objectives.
QUANTITIES = {
    'winding_surface_B': _winding_surface_B,
    'Bnormal': _Bnormal,
    'f_B': f_B,
    'B_self': _B_self,
    'force_cyl': _force_cyl,
    'f_max_force_cyl': f_max_force_cyl,
}


def _rel(a, b):
    a = jnp.ravel(jnp.asarray(a))
    b = jnp.ravel(jnp.asarray(b))
    scale = jnp.maximum(jnp.linalg.norm(a), 1e-30)
    return jnp.linalg.norm(a - b) / scale


def _replace_surface(surf, dofs):
    return type(surf)(
        nfp=surf.nfp,
        stellsym=surf.stellsym,
        mpol=surf.mpol,
        ntor=surf.ntor,
        quadpoints_phi=surf.quadpoints_phi,
        quadpoints_theta=surf.quadpoints_theta,
        dofs=dofs,
    )


def _qp_like(src, bs_chunk_size, winding_surface=None, plasma_surface=None):
    return QuadcoilParams(
        plasma_surface=src.plasma_surface if plasma_surface is None else plasma_surface,
        winding_surface=src.winding_surface if winding_surface is None else winding_surface,
        net_poloidal_current_amperes=src.net_poloidal_current_amperes,
        net_toroidal_current_amperes=src.net_toroidal_current_amperes,
        Bnormal_plasma=src.Bnormal_plasma,
        mpol=src.mpol,
        ntor=src.ntor,
        quadpoints_phi=src.quadpoints_phi,
        quadpoints_theta=src.quadpoints_theta,
        stellsym=src.stellsym,
        bs_chunk_size=bs_chunk_size,
    )


class CachedOperatorTest(unittest.TestCase):
    def test_matches_uncached(self):
        for chunk in (None, 64):
            base = _qp_like(QP, chunk)
            cached = build_cached_ops(base)
            self.assertIsNotNone(cached.cached_ops)
            for name, fn in QUANTITIES.items():
                self._check_phi(name, chunk, fn, base, cached)
                self._check_surface(name, chunk, fn, base, 'winding_surface')
                self._check_surface(name, chunk, fn, base, 'plasma_surface')

    def _check_phi(self, name, chunk, fn, base, cached):
        def packed(phi, qp):
            return jnp.ravel(fn(qp, {'phi': phi}))

        # Sum of squares, not the sum: the sum of a signed field cancels and
        # its derivatives are roundoff. Squaring keeps every component.
        def sq(phi, qp):
            v = packed(phi, qp)
            return jnp.dot(v, v)

        value_err = _rel(packed(PHI, base), packed(PHI, cached))
        grad_err = _rel(jacrev(packed)(PHI, base), jacrev(packed)(PHI, cached))
        hess_err = _rel(
            jacrev(jacrev(sq))(PHI, base),
            jacrev(jacrev(sq))(PHI, cached),
        )
        print(f'  phi {name} chunk={chunk}: value {value_err:.2e} '
              f'grad {grad_err:.2e} hess {hess_err:.2e}')
        self.assertLess(float(value_err), 1e-9)
        self.assertLess(float(grad_err), 1e-9)
        self.assertLess(float(hess_err), 1e-9)

    def _check_surface(self, name, chunk, fn, base, which):
        surf = getattr(base, which)
        dofs0 = surf.dofs

        def scalar(dofs, use_cache):
            repl = _replace_surface(surf, dofs)
            qp = _qp_like(
                base, chunk,
                winding_surface=repl if which == 'winding_surface' else None,
                plasma_surface=repl if which == 'plasma_surface' else None,
            )
            if use_cache:
                qp = build_cached_ops(qp)
            v = jnp.ravel(fn(qp, {'phi': PHI}))
            return jnp.dot(v, v)

        err = _rel(jacrev(scalar)(dofs0, False), jacrev(scalar)(dofs0, True))
        print(f'  {which} {name} chunk={chunk}: grad {err:.2e}')
        self.assertLess(float(err), 1e-9)

    def test_pytree_roundtrip_and_strip(self):
        cached = build_cached_ops(QP)
        children, aux = tree_util.tree_flatten(cached)
        restored = tree_util.tree_unflatten(aux, children)
        A, B0 = restored.cached_ops['winding_surface_B']
        A0, B0_0 = cached.cached_ops['winding_surface_B']
        self.assertLess(float(_rel(A, A0)), 1e-15)
        self.assertLess(float(_rel(B0, B0_0)), 1e-15)

        stripped = cached.without_cached_ops()
        self.assertIsNone(stripped.cached_ops)
        qp_new, _ = cached.change_phi_resolution({'phi': PHI}, QP.mpol, QP.ntor)
        self.assertIsNone(qp_new.cached_ops)


if __name__ == '__main__':
    unittest.main()
