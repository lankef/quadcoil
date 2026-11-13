"""Unit tests for smooth max / abs / L-inf approximations in math_utils."""
import math
import unittest

import jax.numpy as jnp
import numpy as np

from quadcoil.math_utils import abs_lse, linf_lse, max_lse


class MathUtilsLseTest(unittest.TestCase):
    def test_abs_lse_overshoot_independent_of_shape(self):
        """abs_lse overshoot is <= eps*log(2) and does not depend on x.shape."""
        eps = 1e-3
        for shape in ((5,), (4, 3), (33, 33, 3)):
            rng = np.random.default_rng(0)
            x = jnp.asarray(rng.standard_normal(shape))
            approx = abs_lse(x, eps)
            err = approx - jnp.abs(x)
            self.assertTrue(jnp.all(err >= -1e-12))
            self.assertTrue(jnp.all(err <= eps * math.log(2) + 1e-12))
            # Closed form; must not use a log(2N) sharpened scale.
            expected = eps * jnp.logaddexp(x / eps, -x / eps)
            np.testing.assert_allclose(approx, expected, rtol=0, atol=0)

    def test_max_lse_overshoot(self):
        eps = 1e-3
        for n in (2, 10, 100, 1000):
            rng = np.random.default_rng(n)
            x = jnp.asarray(rng.standard_normal(n))
            approx = max_lse(x, eps)
            err = float(approx - jnp.max(x))
            self.assertGreaterEqual(err, -1e-12)
            self.assertLessEqual(err, eps + 1e-12)

    def test_linf_lse_overshoot(self):
        eps = 1e-3
        for shape in ((5,), (8, 4), (20, 10, 3)):
            rng = np.random.default_rng(hash(shape) % (2**31))
            x = jnp.asarray(rng.standard_normal(shape))
            approx = linf_lse(x, eps)
            exact = float(jnp.max(jnp.abs(x)))
            err = float(approx - exact)
            self.assertGreaterEqual(err, -1e-12)
            # abs_lse overshoots by <= eps*log(2); outer max_lse by <= eps.
            self.assertLessEqual(err, 2.0 * eps + 1e-12)


if __name__ == '__main__':
    unittest.main()
