import os
os.environ["JAX_PLATFORMS"] = "cpu"
import unittest
import numpy as np

try:
    from simsopt.geo import CurvePlanarFourier
    SIMSOPT_AVAILABLE = True
except ImportError:
    SIMSOPT_AVAILABLE = False

from quadcoil.io.coil_cutting import planar_curve_from_gamma


class PlanarCurveFromGammaTest(unittest.TestCase):
    @unittest.skipIf(not SIMSOPT_AVAILABLE, "Skipping planar curve fit test, simsopt unavailable.")
    def test_reconstruct_tilted_planar_fourier(self):
        n = 80
        order_true = 2
        source = CurvePlanarFourier(n, order_true)
        dofs = np.zeros(2 * order_true + 1 + 4 + 3)
        # Even harmonics only so the polar origin coincides with the centroid.
        dofs[0] = 1.0   # rc(0)
        dofs[2] = 0.2   # rc(2)
        dofs[4] = 0.1   # rs(2)
        axis = np.array([1.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        theta = np.deg2rad(50.0)
        dofs[5:9] = np.concatenate((
            [np.cos(theta / 2.0)],
            axis * np.sin(theta / 2.0),
        ))
        dofs[9:12] = [1.5, -0.4, 0.8]
        source.set_dofs(dofs)
        gamma = source.gamma()

        order = 4
        fitted = planar_curve_from_gamma(gamma, order=order, ppp=20)
        check = CurvePlanarFourier(n, order)
        check.set_dofs(fitted.get_dofs())

        err = np.max(np.linalg.norm(check.gamma() - gamma, axis=1))
        self.assertLess(err, 1e-10)


if __name__ == '__main__':
    unittest.main()
