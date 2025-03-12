import jax.numpy as jnp
from .surfacerzfourier_jax import SurfaceRZFourierJAX
from .math_utils import sin_or_cos, norm_helper
from jax import jit, tree_util
from functools import lru_cache, partial

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
        self,
        plasma_surface: SurfaceRZFourierJAX, 
        winding_surface: SurfaceRZFourierJAX, 
        net_poloidal_current_amperes: float, 
        net_toroidal_current_amperes: float,
        Bnormal_plasma=None,
        mpol=4, 
        ntor=4, 
        quadpoints_phi = None,
        quadpoints_theta = None, 
        ):
        ''' Writing peroperties '''
        self.plasma_surface = plasma_surface
        self.winding_surface = winding_surface
        self.net_poloidal_current_amperes = net_poloidal_current_amperes
        self.net_toroidal_current_amperes = net_toroidal_current_amperes
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

        ''' Generating evaluation surface '''
        if quadpoints_phi is None:
            len_phi = len(winding_surface.quadpoints_phi)//winding_surface.nfp
            self.quadpoints_phi = winding_surface.quadpoints_phi[:len_phi]
        else:
            self.quadpoints_phi = quadpoints_phi
        if quadpoints_theta is None:
            self.quadpoints_theta = winding_surface.quadpoints_theta
        else:
            self.quadpoints_theta = quadpoints_theta
        if Bnormal_plasma is None:
            self.Bnormal_plasma = jnp.zeros((len(plasma_surface.quadpoints_phi), len(plasma_surface.quadpoints_theta)))
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
    def Kdash_helper(self):
        '''
        Calculates the following quantity
        - Kdash1_sv_op, Kdash2_sv_op: 
        Partial derivatives of K in term of Phi (current potential) harmonics.
        Shape: (n_phi, n_theta, 3(xyz), n_dof)
        - Kdash1_const, Kdash2_const: 
        Partial derivatives of K due to secular terms (net poloidal/toroidal 
        currents). 
        Shape: (n_phi, n_theta, 3(xyz))
        '''
        normal = self.eval_surface.normal()
        gammadash1 = self.eval_surface.gammadash1()
        gammadash2 = self.eval_surface.gammadash2()
        gammadash1dash1 = self.eval_surface.gammadash1dash1()
        gammadash1dash2 = self.eval_surface.gammadash1dash2()
        gammadash2dash2 = self.eval_surface.gammadash2dash2()
        net_poloidal_current_amperes = self.net_poloidal_current_amperes
        net_toroidal_current_amperes = self.net_toroidal_current_amperes
        quadpoints_phi = self.quadpoints_phi
        quadpoints_theta = self.quadpoints_theta
        nfp = self.nfp
        stellsym = self.stellsym
        cp_m, cp_n = self.make_mn()
        normN_prime_2d, _ = norm_helper(normal)
        (
            trig_m_i_n_i,
            trig_diff_m_i_n_i,
            partial_phi,
            partial_theta,
            partial_phi_phi,
            partial_phi_theta,
            partial_theta_theta,
        ) = self.diff_helper()
        # Some quantities
        (
            dg1_inv_n_dash1,
            dg1_inv_n_dash2,
            dg2_inv_n_dash1,
            dg2_inv_n_dash2 
        ) = self.eval_surface.dga_inv_n_dashb()
        # Operators that generates the derivative of K
        # Note the use of trig_diff_m_i_n_i for inverse
        # FT following odd-order derivatives.
        # Shape: (n_phi, n_theta, 3(xyz), n_dof)
        Kdash2_sv_op = (
            dg2_inv_n_dash2[:, :, None, :]
            *(trig_diff_m_i_n_i@partial_phi)[:, :, :, None]
            
            +gammadash2[:, :, None, :]
            *(trig_m_i_n_i@partial_phi_theta)[:, :, :, None]
            /normN_prime_2d[:, :, None, None]
            
            -dg1_inv_n_dash2[:, :, None, :]
            *(trig_diff_m_i_n_i@partial_theta)[:, :, :, None]
            
            -gammadash1[:, :, None, :]
            *(trig_m_i_n_i@partial_theta_theta)[:, :, :, None]
            /normN_prime_2d[:, :, None, None]
        )
        Kdash2_sv_op = jnp.swapaxes(Kdash2_sv_op, 2, 3)
        Kdash1_sv_op = (
            dg2_inv_n_dash1[:, :, None, :]
            *(trig_diff_m_i_n_i@partial_phi)[:, :, :, None]
            
            +gammadash2[:, :, None, :]
            *(trig_m_i_n_i@partial_phi_phi)[:, :, :, None]
            /normN_prime_2d[:, :, None, None]
            
            -dg1_inv_n_dash1[:, :, None, :]
            *(trig_diff_m_i_n_i@partial_theta)[:, :, :, None]
            
            -gammadash1[:, :, None, :]
            *(trig_m_i_n_i@partial_phi_theta)[:, :, :, None]
            /normN_prime_2d[:, :, None, None]
        )
        Kdash1_sv_op = jnp.swapaxes(Kdash1_sv_op, 2, 3)
        G = net_poloidal_current_amperes 
        I = net_toroidal_current_amperes 
        # Constant components of K's partial derivative.
        # Shape: (n_phi, n_theta, 3(xyz))
        Kdash1_const = \
            dg2_inv_n_dash1*G \
            -dg1_inv_n_dash1*I
        Kdash2_const = \
            dg2_inv_n_dash2*G \
            -dg1_inv_n_dash2*I
        return(
            Kdash1_sv_op, 
            Kdash2_sv_op, 
            Kdash1_const,
            Kdash2_const
        )
    
    @lru_cache()
    @partial(jit, static_argnames=[
        'winding_surface_mode',
    ])
    def diff_helper(self, winding_surface_mode=False):
        '''
        Calculates the following quantity:
        - trig_m_i_n_i, trig_diff_m_i_n_i: 
        IFT operator that transforms even/odd derivatives of Phi harmonics
        produced by partial_* (see below). 
        Shape: (n_phi, n_theta, n_dof)
        - partial_theta, partial_phi, ... ,partial_phi_theta,
        A partial derivative operators that works by multiplying the harmonic
        coefficients of Phi by its harmonic number and a sign, depending whether
        the coefficient is sin or cos. DOES NOT RE-ORDER the coefficients
        into the simsopt conventions. Therefore, IFT for such derivatives 
        must be performed with trig_m_i_n_i and trig_diff_m_i_n_i (see above).

        When winding_surface_mode is set to True, uses the winding surface's quadpoints 
        instead. This will be used in K(winding_surface_mode=True), which is used in B calculations.
        '''
        
        nfp = self.nfp
        cp_m, cp_n = self.make_mn()
        if winding_surface_mode:
            quadpoints_phi = self.winding_surface.quadpoints_phi
            quadpoints_theta = self.winding_surface.quadpoints_theta
        else:    
            quadpoints_phi = self.quadpoints_phi
            quadpoints_theta = self.quadpoints_theta
        stellsym = self.stellsym
        # The uniform index for phi contains first sin Fourier 
        # coefficients, then optionally cos is stellsym=False.
        n_harmonic = len(cp_m)
        iden = jnp.identity(n_harmonic)
        # Shape: (n_phi, n_theta)
        phi_grid = jnp.pi*2*quadpoints_phi[:, None]
        theta_grid = jnp.pi*2*quadpoints_theta[None, :]
        # When stellsym is enabled, Phi is a sin fourier series.
        # After a derivative, it becomes a cos fourier series.
        if stellsym:
            trig_choice = 1
        # Otherwise, it's a sin-cos series. After a derivative,
        # it becomes a cos-sin series.
        else:
            trig_choice = jnp.repeat([1,-1], n_harmonic//2)
        # Inverse Fourier transform that transforms a dof 
        # array to grid values. trig_diff_m_i_n_i acts on 
        # odd-order derivatives of dof, where the sin coeffs 
        # become cos coefficients, and cos coeffs become
        # sin coeffs.
        # sin or sin-cos coeffs -> grid vals
        # Shape: (n_phi, n_theta, dof)
        trig_m_i_n_i = sin_or_cos(
            (cp_m)[None, None, :]*theta_grid[:, :, None]
            -(cp_n*nfp)[None, None, :]*phi_grid[:, :, None],
            trig_choice
        )    
        # cos or cos-sin coeffs -> grid vals
        # Shape: (n_phi, n_theta, dof)
        trig_diff_m_i_n_i = sin_or_cos(
            (cp_m)[None, None, :]*theta_grid[:, :, None]
            -(cp_n*nfp)[None, None, :]*phi_grid[:, :, None],
            -trig_choice
        )
    
        # Fourier derivatives
        partial_theta = cp_m*trig_choice*iden*2*jnp.pi
        partial_phi = -cp_n*trig_choice*iden*nfp*2*jnp.pi
        partial_theta_theta = -cp_m**2*iden*(2*jnp.pi)**2
        partial_phi_phi = -(cp_n*nfp)**2*iden*(2*jnp.pi)**2
        partial_phi_theta = cp_n*nfp*cp_m*iden*(2*jnp.pi)**2
        return(
            trig_m_i_n_i,
            trig_diff_m_i_n_i,
            partial_phi,
            partial_theta,
            partial_phi_phi,
            partial_phi_theta,
            partial_theta_theta,
        )
        
    ''' JAX prereqs '''
    def tree_flatten(self):
        children = (
            self.plasma_surface,
            self.winding_surface,
            self.eval_surface,
            self.net_poloidal_current_amperes,
            self.net_toroidal_current_amperes,
            self.Bnormal_plasma,
            self.quadpoints_phi,
            self.quadpoints_theta,
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
            net_poloidal_current_amperes=children[3],
            net_toroidal_current_amperes=children[4],
            Bnormal_plasma=children[5],
            mpol=aux_data['mpol'],
            ntor=aux_data['ntor'],
            quadpoints_phi=children[6],
            quadpoints_theta=children[7],
        )