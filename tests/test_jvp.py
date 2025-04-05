import unittest
from quadcoil import quadcoil
from quadcoil.io import quadcoil_for_diff
import jax.numpy as jnp
from jax import grad
from load_test_data import load_data, compare


_, plasma_surface, _, _, _ = load_data()
net_poloidal_current_amperes = 11884578.094260072
    
class QuadcoilDESCTest(unittest.TestCase):
    def test_jvp(self):
        nescoil_out_dict, nescoil_qp, nescoil_phi_mn, _ = quadcoil(
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            mpol=4, # 4 poloidal harmonics for the current potential
            ntor=4, # 4 toroidal harmonics for the current potential
            plasma_dofs=plasma_surface.get_dofs(),
            plasma_mpol=plasma_surface.mpol,
            plasma_ntor=plasma_surface.ntor,
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            net_toroidal_current_amperes=0.,
            plasma_coil_distance=plasma_surface.minor_radius(),
            # Set the objective to 
            # f_B
            objective_name='f_B',
            objective_weight=None,
            objective_unit=None,
            # Set the output metrics to f_B and f_K
            metric_name=('f_B', 'f_K')
        )
        nescoil_out_dict2 = quadcoil_for_diff(
            plasma_dofs=plasma_surface.get_dofs(),
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            net_toroidal_current_amperes=0.,
            Bnormal_plasma=None,
            plasma_coil_distance=plasma_surface.minor_radius(),
            winding_dofs=None,
            objective_weight=None,
            constraint_value=(),
            nondiff_args={
                'nfp':plasma_surface.nfp,
                'stellsym':plasma_surface.stellsym,
                'mpol':4, # 4 poloidal harmonics for the current potential
                'ntor':4, # 4 toroidal harmonics for the current potential
                'plasma_mpol':plasma_surface.mpol,
                'plasma_ntor':plasma_surface.ntor,
                'objective_name':'f_B',
                'objective_weight':None,
                'objective_unit':None,
                'metric_name':('f_B', 'f_K')
            }
        )
    
        # For testing if the custom_jvp rule
        # is working properly
        def test_f_B(plasma_dofs, G):
            nescoil_out_dict = quadcoil_for_diff(
                plasma_dofs=plasma_dofs,
                net_poloidal_current_amperes=G,
                net_toroidal_current_amperes=0.,
                Bnormal_plasma=None,
                plasma_coil_distance=plasma_surface.minor_radius(),
                winding_dofs=None,
                objective_weight=None,
                constraint_value=(),
                nondiff_args={
                    'nfp':plasma_surface.nfp,
                    'stellsym':plasma_surface.stellsym,
                    'mpol':4, # 4 poloidal harmonics for the current potential
                    'ntor':4, # 4 toroidal harmonics for the current potential
                    'plasma_mpol':plasma_surface.mpol,
                    'plasma_ntor':plasma_surface.ntor,
                    'objective_name':'f_B',
                    'objective_weight':None,
                    'objective_unit':None,
                    'metric_name':('f_B', 'f_K')
                }
            )
            return(nescoil_out_dict['f_B'])
            
        grad_plasma_dofs = grad(test_f_B)(plasma_surface.get_dofs(), net_poloidal_current_amperes)
        grad_plasma_dofs_ans = nescoil_out_dict['f_B']['grad']['df_dplasma_dofs']
        self.assertTrue(compare(grad_plasma_dofs, grad_plasma_dofs_ans))
        self.assertTrue(compare(nescoil_out_dict['f_B']['value'], test_f_B(plasma_surface.get_dofs(), net_poloidal_current_amperes)))

if __name__ == "__main__":
    unittest.main()