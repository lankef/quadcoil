import jax.numpy as jnp
from .surfacerzfourierjax import SurfaceRZFourierJAX
from .objectives.field_error import winding_surface_field_Bn, winding_surface_field_Bn_GI
from jax import jit, tree_util
from functools import lru_cache

'''
This class is somewhat analogous to CurrentPotential in simsopt. 
It stores all QUADCOIL parameters. These includes:
- The plasma surface
- The winding surface
- The evaluation surface
- The net currents
- Numerical parameters
'''
@tree_util.register_pytree_node_class
class QuadcoilParams:
    def __init__(
        plasma_surface: SurfaceRZFourierJAX, 
        winding_surface: SurfaceRZFourierJAX, 
        net_poloidal_current: float, 
        net_toroidal_current: float,
        Bnormal_plasma: jnp.array,
        mpol: int, ntor: int, 
        quadpoints_phi = None,
        quadpoints_theta = None, 
        ):
        ''' Writing peroperties '''
        self.plasma_surface = plasma_surface
        self.winding_surface = winding_surface
        self.net_poloidal_current = net_poloidal_current
        self.net_toroidal_current = net_toroidal_current
        self.Bnormal_plasma = Bnormal_plasma
        self.nfp = winding_surface.nfp
        self.stellsym = winding_surface.stellsym
        self.mpol = mpol
        self.ntor = ntor
        ndof = 2 * mpol * ntor + mpol + ntor
        if self.stellsym:
            self.ndof = ndof
            self.ndofs_half = ndof
        else:
            self.ndof = 2 * ndof
            self.ndofs_half = ndof//2

        ''' Calculating phi, theta integration step sizes '''

        dphi_plasma = (plasma_surface.quadpoints_phi[1] - plasma_surface.quadpoints_phi[0])
        dtheta_plasma = (plasma_surface.quadpoints_theta[1] - plasma_surface.quadpoints_theta[0])
        dphi_winding = (winding_surface.quadpoints_phi[1] - winding_surface.quadpoints_phi[0])
        dtheta_winding = (winding_surface.quadpoints_theta[1] - winding_surface.quadpoints_theta[0])

        ''' Generating evaluation surface '''
        if quadpoints_phi is None:
            len_phi = len(winding_surface.quadpoints_phi)//winding_surface.nfp
            self.quadpoints_phi = winding_surface.quadpoints_phi[:len_phi, :]
        else:
            self.quadpoints_phi = quadpoints_phi
        if quadpoints_theta is None:
            self.quadpoints_theta = winding_surface.quadpoints_theta
        else:
            self.quadpoints_theta = quadpoints_theta
        # Evaluation surface. Winding surface is used for integration, 
        # and eval_surface is used for evaluating quantities on a grid
        self.eval_surface = SurfaceRZFourierJAX(
            nfp=self.winding_surface.nfp, 
            stellsym=self.winding_surface.stellsym, 
            mpol=self.winding_surface.mpol, 
            ntor=self.winding_surface.ntor, 
            quadpoints_phi=self.quadpoints_phi, 
            quadpoints_theta=self.quadpoints_theta, 
            dofs=self.winding_surface.dofs
        )
    
    ''' Cached quantites '''
    @lru_cache()
    @jit
    def make_mn(self):
        """
        Make the list of m and n values. Equivalent to CurrentPotential.make_mn.
        """
        mpol = self.mpol
        ntor = self.ntor
        stellsym = self.stellsym
        m1d = jnp.arange(mpol + 1)
        n1d = jnp.arange(-ntor, ntor + 1)
        n2d, m2d = jnp.meshgrid(n1d, m1d)
        m0 = m2d.flatten()[ntor:]
        n0 = n2d.flatten()[ntor:]
        m = m0[1::]
        n = n0[1::]

        if not stellsym:
            m = jnp.append(m, m)
            n = jnp.append(n, n)
        return(m, n)

    @lru_cache()
    @jit
    def biot_savart(self, integral_mode=True):
        # When integral_mode is True, returns the same 
        # gj and b_e from simsopt.CurrentPotential.
        ''' Calculating gj '''
        normal_plasma = self.plasma_surface.normal()
        gamma_plasma = self.plasma_surface.gamma()

        normal_winding = self.winding_surface.normal()
        gamma_winding = self.winding_surface.gamma()
        gammadash1_winding = self.winding_surface.gammadash1()
        gammadash2_winding = self.winding_surface.gammadash2()


        normal_plasma_flat = normal_plasma.reshape(-1, 3)
        normal_winding_flat = normal_winding.reshape(-1, 3)
        points_plasma_flat = gamma_plasma.reshape(-1, 3)
        points_coil_flat = gamma_winding.reshape(-1, 3)
        gammadash1_winding_flat = gammadash1_winding.reshape(-1, 3)
        gammadash2_winding_flat = gammadash2_winding.reshape(-1, 3)

        Bnormal_plasma_flat = self.Bnormal_plasma.flatten()

        m, n = self.make_mn()

        # The biot-savart linear operator, gj, and 
        # the field due to plasma B_norm and net currents b_e
        gj_unscaled, _ = winding_surface_field_Bn(
            points_plasma_flat, # contig(points_plasma), 
            points_coil_flat, # contig(points_coil), 
            normal_plasma_flat, # contig(normal_plasma), 
            normal_winding_flat, # contig(normal), 
            self.stellsym, # self.winding_surface.stellsym, 
            jnp.ravel(self.winding_surface.phi_mesh), # contig(zeta_coil), 
            jnp.ravel(self.winding_surface.theta_mesh), # contig(theta_coil), 
            self.ndof, # dummy, but used to keep the function signature the same
            m[:self.ndofs_half], # contig(self.current_potential.m[:ndofs_half]), 
            n[:self.ndofs_half], # contig(self.current_potential.n[:ndofs_half]), 
            self.nfp# self.winding_surface.nfp
        )
        ''' Calculating b_e '''
        normN_plasma = jnp.linalg.norm(normal_plasma, axis=-1)
        normN_plasma_flat = normN_plasma.flatten()

        B_GI = winding_surface_field_Bn_GI(
            points_plasma_flat, # points_plasma, 
            points_coil_flat, # points_coil, 
            normal_plasma_flat, # normal_plasma,
            jnp.ravel(self.winding_surface.phi_mesh), # zeta_coil, 
            jnp.ravel(self.winding_surface.theta_mesh), # theta_coil, 
            self.net_poloidal_current, # G
            self.net_toroidal_current, # I
            gammadash1_winding_flat, # gammadash1, 
            gammadash2_winding_flat, # gammadash2
        ) * self.dphi_winding * self.dtheta_winding

        if integral_mode:
            b_e = - jnp.sqrt(normN_plasma_flat * self.dphi_plasma * self.dtheta_plasma) * (B_GI + Bnormal_plasma_flat)
            B_normal = gj_unscaled * jnp.sqrt(
                self.dphi_plasma 
                * self.dtheta_plasma 
                * self.dphi_winding ** 2 
                * self.dtheta_winding ** 2
                / normN_plasma_flat[:, None]
            )
            return B_normal, b_e
        else:
            raise NotImplementedError('Local field error is not implemented yet.')

    ''' Jax prerequisites '''
    def tree_flatten(self):
        children = (
            self.plasma_surface,
            self.winding_surface,
            self.eval_surface,
            self.net_poloidal_current,
            self.net_toroidal_current,
            self.Bnormal_plasma,
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.dphi_plasma,
            self.dtheta_plasma,
            self.dphi_winding,
            self.dtheta_winding,
        )
        aux_data = {
            'nfp': self.nfp,
            'stellsym': self.stellsym,
            'mpol': self.mpol,
            'ntor': self.ntor,
            'ndof': self.ndof,
            'ndofs_half': self.ndofs_half,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            plasma_surface=children[0],
            winding_surface=children[1],
            net_poloidal_current=children[3],
            net_toroidal_current=children[4],
            Bnormal_plasma=children[5],
            mpol=aux_data['mpol'],
            ntor=aux_data['ntor'],
            quadpoints_phi=children[-2],
            quadpoints_theta=children[-1],
        )