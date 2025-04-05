import os
os.environ["JAX_PLATFORMS"] = "cpu"
import unittest
import jax.numpy as jnp
import quadcoil
import quadcoil.io
from load_test_data import compare
try:
    from simsopt.mhd import Vmec
    import desc
    DESC_AVAILABLE = True
except ImportError:
    DESC_AVAILABLE = False
    
class QuadcoilDESCTest(unittest.TestCase):
    # DESC has the opposite t
    @unittest.skipIf(not DESC_AVAILABLE, "Skipping DESC surface loading test, DESC and/or Simsopt unavailable.")
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
        
    def test_simple_wrapper(self):
        # Loading equilibrium into DESC
        wout_file = "wout_muse++.nc"
        desc_eq = desc.vmec.VMECIO.load(wout_file)
        # Loading equilibrium into quadcoil directly through Simsopt
        equil_qs = Vmec(wout_file, keep_all_files=True)
        plasma_surface = equil_qs.boundary
        net_poloidal_current_amperes = equil_qs.external_current()
        # Test arguments
        test_args = {
            'mpol':4, # 4 poloidal harmonics for the current potential
            'ntor':4, # 4 toroidal harmonics for the current potential
            'plasma_coil_distance': 0.1,
            'objective_name':'f_B',
            'objective_weight':None,
            'objective_unit':None,
            'metric_name':('f_B', 'f_K'),
        }
        # DESC interface
        outdict_desc, qp_desc, phi_mn_desc, status_desc = quadcoil.io.quadcoil_desc(
            desc_eq=desc_eq,
            vacuum=True,
            plasma_M_theta=20,
            plasma_N_phi=21,
            desc_scaling=False,
            **test_args
        )
        # DESC interface
        outdict_desc2, qp_desc2, phi_mn_desc2, status_desc2 = quadcoil.io.quadcoil_desc(
            desc_eq=desc_eq,
            vacuum=True,
            plasma_M_theta=20,
            plasma_N_phi=21,
            desc_scaling=True,
            **test_args
        )
        outdict_ans, qp_ans, phi_mn_ans, _ = quadcoil.quadcoil(
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            plasma_dofs=plasma_surface.get_dofs(),
            plasma_mpol=plasma_surface.mpol,
            plasma_ntor=plasma_surface.ntor,
            plasma_quadpoints_phi=qp_desc.plasma_surface.quadpoints_phi,
            plasma_quadpoints_theta=qp_desc.plasma_surface.quadpoints_theta,
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            **test_args
        )
        print(
            'Objective values: f_B =',
            outdict_desc['f_B']['value'],
            outdict_desc2['f_B']['value'],
            outdict_ans['f_B']['value']
        )
        print(
            'Objective values: f_K =',
            outdict_desc['f_K']['value'],
            outdict_desc2['f_K']['value'],
            outdict_ans['f_K']['value']
        )
        # The negative sign come from handedness different
        print('Comparing wout -> desc -> quadcoil vs wout -> quadcoil.')
        print('The wout -> desc relies on fitting and is not exact. ')
        print('Therefore this test will only require <0.5% deviation.')
        print('between the two optimu')
        print('Testing vacuum.')
        self.assertTrue(compare(phi_mn_ans, -phi_mn_desc, err=5e-3))
        print('Testing vacuum, scaled using DESC quantities.')
        self.assertTrue(compare(phi_mn_ans, -phi_mn_desc2, err=1e-2))

if __name__ == "__main__":
    unittest.main()