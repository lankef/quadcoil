import unittest
from quadcoil import QuadcoilParams, SurfaceRZFourierJAX, project_arr_cylindrical
from quadcoil.objective import K_dot_grad_K, K_dot_grad_K_cyl
import jax.numpy as jnp
import numpy as np
from simsopt.geo import SurfaceRZFourier
from load_test_data import load_data, compare
try:
    from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
    CPF_AVAILABLE = True
except ImportError:
    CPF_AVAILABLE = False

winding_surface, plasma_surface, cp, _, _ = load_data()

class QuadcoilKKTest(unittest.TestCase):

    @unittest.skipIf(not CPF_AVAILABLE, "Skipping K dot grad K test, simsopt.field.CurrentPotentialFourier unavailable.")
    def test_KK(self):
        for i in range(5):
            # We compare the operator and a finite difference value of K dot grad K
            # in 10 small, random patches of the winding surface. This is because
            # np.gradient is only 2nd order accurate and needs very small grid size.
            loc1 = np.random.random()
            loc2 = np.random.random()

            winding_surface_hi_res = SurfaceRZFourier(
                    nfp=winding_surface.nfp, 
                    stellsym=winding_surface.stellsym, 
                    mpol=winding_surface.mpol, 
                    ntor=winding_surface.ntor, 
                    quadpoints_phi=np.linspace(loc1, loc1+0.0001, 200, endpoint=False), 
                    quadpoints_theta=np.linspace(loc2, loc2+0.0001, 200, endpoint=False), 
                )
            winding_surface_hi_res.set_dofs(winding_surface.get_dofs())
            cp_hi_res = CurrentPotentialFourier(
                winding_surface_hi_res, mpol=cp.mpol, ntor=cp.ntor,
                net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
                net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
                stellsym=True)
            cp_hi_res.set_dofs(cp.get_dofs())
            cpst = CurrentPotentialSolve(cp_hi_res, plasma_surface, np.zeros(1024))
            theta_study1d, phi_study1d = cp_hi_res.quadpoints_theta, cp_hi_res.quadpoints_phi
            G = cp_hi_res.net_poloidal_current_amperes
            I = cp_hi_res.net_toroidal_current_amperes
            normal_vec = cp_hi_res.winding_surface.normal()
            normN_prime_2d = np.sqrt(np.sum(normal_vec**2, axis=-1)) # |N|
            dK_dphi, dK_dtheta = np.gradient(
                cp_hi_res.K(), 
                phi_study1d,
                theta_study1d,
                axis=(0,1)
            )
            KK = (
                (cp_hi_res.Phidash1()[:, :, None]+G)*dK_dtheta
                -(cp_hi_res.Phidash2()[:, :, None]+I)*dK_dphi
            )/normN_prime_2d[:,:,None]
            KK_cyl = project_arr_cylindrical(gamma=cp_hi_res.winding_surface.gamma(), operator=KK)

            ''' Quadcoil implementation '''
            
            plasma_surface_jax = SurfaceRZFourierJAX.from_simsopt(plasma_surface)
            winding_surface_hi_res_jax = SurfaceRZFourierJAX.from_simsopt(winding_surface_hi_res)
            qp = QuadcoilParams(
                plasma_surface=plasma_surface_jax,
                winding_surface=winding_surface_hi_res_jax,
                net_poloidal_current_amperes=cp_hi_res.net_poloidal_current_amperes, 
                net_toroidal_current_amperes=cp_hi_res.net_toroidal_current_amperes,
                quadpoints_phi=cp_hi_res.quadpoints_phi,
                quadpoints_theta=cp_hi_res.quadpoints_theta,
                mpol=cp_hi_res.mpol, 
                ntor=cp_hi_res.ntor, 
            )
            KK_test = K_dot_grad_K(qp, cp.get_dofs())
            KK_test_cyl = K_dot_grad_K_cyl(qp, cp.get_dofs())
            # Remove the edge of both results because np.gradient
            # is inaccurate at the edges.
            KK = KK[1:-1,1:-1,:]
            KK_test = KK_test[1:-1,1:-1,:]
            KK_cyl = KK_cyl[1:-1,1:-1,:]
            KK_test_cyl = KK_test_cyl[1:-1,1:-1,:]
            print(
                'Test #', i, 
                'max error/max amplitude =', 
                np.max(KK-KK_test)/np.max(KK)
            )
            print(
                'Test #', i, 
                'max error/max amplitude (cylindrical) =', 
                np.max(KK_cyl-KK_test_cyl)/np.max(KK_cyl)
            )
            self.assertTrue(compare(KK_cyl, KK_test_cyl))
            self.assertTrue(compare(KK, KK_test))

if __name__ == "__main__":
    unittest.main()