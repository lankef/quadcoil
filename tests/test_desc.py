import unittest
import jax.numpy as jnp
import quadcoil
from simsopt.mhd import Vmec
from load_test_data import compare
try:
    import desc
    DESC_AVAILABLE = True
except ImportError:
    DESC_AVAILABLE = False
    
class QuadcoilDESCTest(unittest.TestCase):
    # DESC has the opposite t
    @unittest.skipIf(not DESC_AVAILABLE, "Skipping DESC surface loading test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_surface(self):
        muse_desc = desc.vmec.VMECIO.load("wout_muse++.nc")
        # Loading surface into simsopt
        equil_pp = Vmec('wout_muse++.nc', keep_all_files=True)
        # equil = Vmec(filename, mpi=mpi)
        surf_pp = equil_pp.boundary
        test_surf_pp = quadcoil.SurfaceRZFourierJAX.from_desc(
            muse_desc.surface, 
            surf_pp.quadpoints_phi,
            surf_pp.quadpoints_theta
        ).to_simsopt()
        print('Testing DESC surface loading')
        print('Because DESC surface seem to have '
              'the opposite poloidal angle sign '
              'we test by finding the maximum surface-surface '
              'distance instad.')       
        gamma1 = surf_pp.gamma().reshape(-1, 3)
        gamma2 = test_surf_pp.gamma().reshape(-1, 3)
        dists = jnp.linalg.norm(gamma1[:, None, :] - gamma2[None, :, :], axis=-1)
        min_dists = jnp.min(dists, axis=1)
        print('Maximum surface-to-surface dist', jnp.max(min_dists))
        print('Average surface-to-surface dist', jnp.average(min_dists))
        self.assertTrue(jnp.max(min_dists)<1e-10)
if __name__ == "__main__":
    unittest.main()