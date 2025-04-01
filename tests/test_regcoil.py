
from load_test_data import load_data, compare
try:
    from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
    CPF_AVAILABLE = True
except ImportError:
    CPF_AVAILABLE = False
from quadcoil import quadcoil
import jax.numpy as jnp
import numpy as np
import unittest
from jax import block_until_ready
import time
winding_surface, plasma_surface, cp, cpst, qp = load_data()

class QuadcoilKTest(unittest.TestCase):
    @unittest.skipIf(not CPF_AVAILABLE, "Skipping sln vs regcoil test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_regcoil(self):
        # First, test with the NESCOIL problem, auto-generating WS
        print('Testing NESCOIL, with auto-generated winding surface')
        nescoil_out_dict, nescoil_qp, nescoil_phi_mn, _ = quadcoil(
            nfp=cp.nfp,
            stellsym=cp.stellsym,
            mpol=cp.mpol,
            ntor=cp.ntor,
            plasma_dofs=cpst.plasma_surface.get_dofs(),
            plasma_mpol=cpst.plasma_surface.mpol,
            plasma_ntor=cpst.plasma_surface.ntor,
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            plasma_coil_distance=plasma_surface.minor_radius(),
            metric_name=('f_B', 'f_K')
        )
        ws0 = nescoil_qp.winding_surface.to_simsopt()
        cp0 = CurrentPotentialFourier(
            ws0, mpol=cp.mpol, ntor=cp.ntor,
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            stellsym=True)
        cpst0 = CurrentPotentialSolve(cp0, plasma_surface, 0)
        nescoil_phi_mn_ans, nescoil_f_B_ans, nescoil_f_K_ans = cpst0.solve_tikhonov()
        print('Phi:')
        compare(nescoil_phi_mn_ans, nescoil_phi_mn, 1e-3)
        print('f_B:')
        compare(nescoil_out_dict['f_B']['value'], nescoil_f_B_ans, 1e-3)
        print('f_K:')
        compare(nescoil_out_dict['f_K']['value'], nescoil_f_K_ans, 1e-3)

        
        # First, test with the REGCOIL problem, auto-generating WS
        print('Testing REGCOIL, with auto-generated winding surface')
        regcoil1_out_dict, regcoil1_qp, regcoil1_phi_mn, _ = quadcoil(
            nfp=cp.nfp,
            stellsym=cp.stellsym,
            mpol=cp.mpol,
            ntor=cp.ntor,
            plasma_dofs=cpst.plasma_surface.get_dofs(),
            plasma_mpol=cpst.plasma_surface.mpol,
            plasma_ntor=cpst.plasma_surface.ntor,
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            plasma_coil_distance=plasma_surface.minor_radius(),
            objective_name=('f_B', 'f_K'),
            objective_weight=(1., 0.01),
            # Usually we recommend normalizing f_B and f_K to 
            # ~1, but in this case, for testing prupose, not normalizing 
            # is also okay
            objective_unit=(1., 1.), 
            metric_name=('f_B', 'f_K')
        )
        ws1 = regcoil1_qp.winding_surface.to_simsopt()
        cp1 = CurrentPotentialFourier(
            ws1, mpol=cp.mpol, ntor=cp.ntor,
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            stellsym=True)
        cpst1 = CurrentPotentialSolve(cp1, plasma_surface, 0)
        regcoil1_phi_mn_ans, regcoil1_f_B_ans, regcoil1_f_K_ans = cpst1.solve_tikhonov(lam=0.01)
        print('Phi:')
        compare(regcoil1_phi_mn_ans, regcoil1_phi_mn, 1e-3)
        print('f_B:')
        compare(regcoil1_out_dict['f_B']['value'], regcoil1_f_B_ans, 1e-3)
        print('f_K:')
        compare(regcoil1_out_dict['f_K']['value'], regcoil1_f_K_ans, 1e-3)

        # Then, test with REGCOIL, given winding surface.
        print('Testing REGCOIL, with known winding surface')
        regcoil2_out_dict, regcoil2_qp, regcoil2_phi_mn, _ = quadcoil(
            nfp=cp.nfp,
            stellsym=cp.stellsym,
            mpol=cp.mpol,
            ntor=cp.ntor,
            plasma_dofs=cpst.plasma_surface.get_dofs(),
            plasma_mpol=cpst.plasma_surface.mpol,
            plasma_ntor=cpst.plasma_surface.ntor,
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            winding_dofs=cpst.winding_surface.get_dofs(),
            winding_mpol=cpst.winding_surface.mpol,
            winding_ntor=cpst.winding_surface.ntor,
            winding_quadpoints_phi=cpst.winding_surface.quadpoints_phi,
            winding_quadpoints_theta=cpst.winding_surface.quadpoints_theta,
            objective_name=('f_B', 'f_K'),
            objective_weight=(1., 1e-14),
            # Usually we recommend normalizing f_B and f_K to 
            # ~1, but in this case, for testing prupose, not normalizing 
            # is also okay
            objective_unit=(1., 1.), 
            metric_name=('f_B', 'f_K')
        )
        regcoil2_phi_mn_ans, regcoil2_f_B_ans, regcoil2_f_K_ans = cpst.solve_tikhonov(lam=1e-14)
        print('Phi:')
        compare(regcoil2_phi_mn, regcoil2_phi_mn_ans, 1e-3)
        print('f_B:')
        compare(regcoil2_out_dict['f_B']['value'], regcoil2_f_B_ans, 1e-3)
        print('f_K:')
        compare(regcoil2_out_dict['f_K']['value'], regcoil2_f_K_ans, 1e-3)

    @unittest.skipIf(not CPF_AVAILABLE, "Skipping gradient vs regcoil test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_taylor(self):
        print('Taylor test, w.r.t. plasma_dof[0]')
        plasma_dof_orig = plasma_surface.get_dofs()
        plasma_dof_0_list = []
        phi_test_list = []
        f_B_test_list = []
        f_K_test_list = []
        phi_ans_list = []
        f_B_ans_list = []
        f_K_ans_list = []
        f_B_diff_test_list = []
        f_K_diff_test_list = []
        time_quadcoil_list = []
        time_regcoil_list = []
        for i in jnp.linspace(-0.1, 0.1, 200):
            plasma_dof_i = plasma_dof_orig.copy()
            plasma_dof_i[0] *= (1 + i)
            # Run QUADCOIL first
            time1 = time.time()
            regcoili_out_dict, regcoili_qp, regcoili_phi_mn, _ = quadcoil(
                nfp=cp.nfp,
                stellsym=cp.stellsym,
                mpol=cp.mpol,
                ntor=cp.ntor,
                plasma_dofs=plasma_dof_i,
                plasma_mpol=cpst.plasma_surface.mpol,
                plasma_ntor=cpst.plasma_surface.ntor,
                net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
                net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
                plasma_coil_distance=plasma_surface.minor_radius(),
                objective_name=('f_B', 'f_K'),
                objective_weight=(1., 0.01),
                # Usually we recommend normalizing f_B and f_K to 
                # ~1, but in this case, for testing prupose, not normalizing 
                # is also okay
                objective_unit=(1., 1.), 
                metric_name=('f_B', 'f_K')
            )
            block_until_ready(regcoili_phi_mn)
            block_until_ready(regcoili_out_dict)
            block_until_ready(regcoili_qp)
            time2 = time.time()
            time_quadcoil_list.append(time2-time1)
            # REGCOIL ---------------------------------------------------
            time1 = time.time()
            plasma_dof_0_list.append(plasma_dof_i[0])
            phi_test_list.append(regcoili_phi_mn)
            f_B_test_list.append(regcoili_out_dict['f_B']['value'])
            f_K_test_list.append(regcoili_out_dict['f_K']['value'])
            f_B_diff_test_list.append(regcoili_out_dict['f_B']['grad']['df_dplasma_dofs'][0])
            f_K_diff_test_list.append(regcoili_out_dict['f_K']['grad']['df_dplasma_dofs'][0])
            wsi = regcoili_qp.winding_surface.to_simsopt()
            psi = regcoili_qp.plasma_surface.to_simsopt()
            cpi = CurrentPotentialFourier(
                wsi, mpol=cp.mpol, ntor=cp.ntor,
                net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
                net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
                stellsym=True)
            cpsti = CurrentPotentialSolve(cpi, psi, 0)
            regcoili_phi_mn_ans, regcoili_f_B_ans, regcoili_f_K_ans = cpsti.solve_tikhonov(0.01)
            time2 = time.time()
            time_regcoil_list.append(time2-time1)
            phi_ans_list.append(regcoili_phi_mn_ans)
            f_B_ans_list.append(regcoili_f_B_ans)
            f_K_ans_list.append(regcoili_f_K_ans)
        print('QUADCOIL avg time:', np.average(time_quadcoil_list))
        print('QUADCOIL max time:', np.max(time_quadcoil_list))
        print('REGCOIL  avg time:', np.average(time_regcoil_list))
        print('REGCOIL  max time:', np.max(time_regcoil_list))
        print('f_K:')
        compare(jnp.array(f_K_test_list), jnp.array(f_K_ans_list), 1e-3)
        print('f_B:')
        compare(jnp.array(f_B_test_list), jnp.array(f_B_ans_list), 1e-3)
        print('f_K gradient:')
        grad_f_K = jnp.gradient(jnp.array(f_K_test_list).flatten(), plasma_dof_0_list[1]-plasma_dof_0_list[0])
        compare(grad_f_K[1:-1], jnp.array(f_K_diff_test_list)[1:-1], 1e-3)
        print('f_B gradient:')
        grad_f_B = jnp.gradient(jnp.array(f_B_test_list).flatten(), plasma_dof_0_list[1]-plasma_dof_0_list[0])
        compare(grad_f_B[1:-1], jnp.array(f_B_diff_test_list)[1:-1], 1e-3)
    
if __name__ == "__main__":
    unittest.main()