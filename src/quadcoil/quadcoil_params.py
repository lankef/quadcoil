import jax.numpy as jnp
import numpy as np
from .surfacerzfourier_jax import SurfaceRZFourierJAX
from .math_utils import sin_or_cos, norm_helper
from jax import jit, tree_util
from functools import lru_cache, partial


# Contains
# - The plasma surface
# - The winding surface
# - The evaluation surface
# - The net currents
# - Numerical parameters
class _Params:
    r'''
    An abstract class for different types of QuadcoilParams. 
    (The default one uses fourier basis. A new one that uses 
    will be added.)
    '''
    def __init__(
        self,
        plasma_surface: SurfaceRZFourierJAX, 
        winding_surface: SurfaceRZFourierJAX, 
        net_poloidal_current_amperes: float, 
        net_toroidal_current_amperes: float,
        Bnormal_plasma=None,
        quadpoints_phi=None,
        quadpoints_theta=None, 
        stellsym=None
        ):
        
        # Writing peroperties 
        self.plasma_surface = plasma_surface
        self.winding_surface = winding_surface
        self.net_poloidal_current_amperes = net_poloidal_current_amperes
        self.net_toroidal_current_amperes = net_toroidal_current_amperes
        self.Bnormal_plasma = Bnormal_plasma
        self.nfp = winding_surface.nfp
        if stellsym is None:
            stellsym = winding_surface.stellsym and plasma_surface.stellsym
        self.stellsym = stellsym

        # Generating evaluation surface 
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
        self.eval_surface = self.winding_surface.copy_and_set_quadpoints(
            quadpoints_phi=self.quadpoints_phi, 
            quadpoints_theta=self.quadpoints_theta, 
        )

@tree_util.register_pytree_node_class
class QuadcoilParams(_Params):
    r'''
    A class storing all informations required to solve a quadcoil 
    problem. These includes plasma info, winding surface info, but 
    does not include problem-specific info such as objectives, constraints
    or solutions. This class is primarily intended as a concise way to 
    pass information into objective functions. It allows functions
    in ``quadcoil.objective`` to have the same signature, despite requiring
    different info to calculate. 

    Parameters
    ----------
    plasma_surface : SurfaceRZFourierJAX
        The plasma surface.
    winding_surface : SurfaceRZFourierJAX
        The winding surface. Must have all field periods.
    net_poloidal_current_amperes : float
        The net poloidal current.
    net_toroidal_current_amperes : float
        The net toroidal current.
    Bnormal_plasma : ndarray, shape (nphi, ntheta), optional, default=None
        The magnetic field distribution on the plasma 
        surface. 
    mpol : int, optional, default=4
        The number of poloidal Fourier harmonics in the current potential :math:`\Phi_{sv}`. 
    ntor : int, optional, default=4
        The number of toroidal Fourier harmonics in :math:`\Phi_{sv}`. 
    quadpoints_phi : ndarray, shape (nphi,), optional, default=None
        The toroidal quadrature points to evaluate quantities at. 
        Takes one field period from the winding surface by default. 
    quadpoints_theta : ndarray, shape (ntheta,), optional, default=None
        The poloidal quadrature points to evaluate quantities at. 
        Takes the winding surface's quadrature points by default. 

    Attributes
    ----------
    plasma_surface : SurfaceRZFourierJAX
        (Traced) The plasma surface.
    winding_surface : SurfaceRZFourierJAX
        (Traced) The winding surface. Must have all field periods.
    eval_surface : SurfaceRZFourierJAX
        (Traced) The evaluation surface. Has the same dofs as the winding surface, but uses the
        quadrature points given by ``self.quadpoints_phi`` and ``self.quadpoints_phi``.
    net_poloidal_current_amperes : float
        (Traced) The net poloidal current.
    net_toroidal_current_amperes : float
        (Traced) The net toroidal current.
    Bnormal_plasma : ndarray, shape (nphi, ntheta)
        (Traced) The magnetic field distribution on the plasma 
        surface. will be filled with zeros by default. 
    quadpoints_phi : ndarray, shape (nphi,)
        (Traced) The toroidal quadrature points to evaluate quantities at. 
    quadpoints_theta : ndarray, shape (ntheta,)
        (Traced) The poloidal quadrature points to evaluate quantities at. 
    nfp : int
        (Static) The number of field periods.
    stellsym : bool
        (Static) Stellarator symmetry.
    mpol : int
        (Static) The number of poloidal Fourier harmonics in :math:`\Phi_{sv}`.
    ntor : int
        (Static) The number of toroidal Fourier harmonics in :math:`\Phi_{sv}`.
    ndofs : int
        (Static) The number of degrees of freedom in :math:`\Phi_{sv}`.
    ndofs_half : int
        (Static) ``ndof`` if ``stellsym==True``, ``ndof//2`` otherwise. 
    '''
    def __init__(
        self,
        plasma_surface, 
        winding_surface, 
        net_poloidal_current_amperes: float, 
        net_toroidal_current_amperes: float,
        Bnormal_plasma=None,
        mpol=4, 
        ntor=4, 
        quadpoints_phi=None,
        quadpoints_theta=None, 
        stellsym=None
        ):
        
        # Writing peroperties 
        super().__init__(
            plasma_surface=plasma_surface,
            winding_surface=winding_surface,
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            net_toroidal_current_amperes=net_toroidal_current_amperes,
            Bnormal_plasma=Bnormal_plasma,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            stellsym=stellsym,
        )
        self.mpol = mpol
        self.ntor = ntor
        self.ndofs, self.ndofs_half = cp_ndofs(self.stellsym, self.mpol, self.ntor)
        

    # -- JAX prereqs --
    # Update this if you ever change the constructor!
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
            stellsym=aux_data['stellsym']
        )

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
            'ndofs': self.ndofs,
            'ndofs_half': self.ndofs_half,
            # 'm': self.m,
            # 'n': self.n,
        }
        return children, aux_data
    
    # -- Cached quantites -- 
    # == Helpers == 
    @lru_cache()
    @jit
    def make_mn(self):
        r'''
        Generates 2 ``array(int)`` of Fourier mode numbers, :math:`m` and :math:`n`, that
        gives the :math:`m` and :math:`n` of the corresponding element in 
        ``self.dofs``. Equivalent to CurrentPotential.make_mn. Caches. 

        Returns
        -------
        m : ndarray
            An array of ints storing the poloidal Fourier mode numbers. Shape: (ndofs)
        n : ndarray
            An array of ints storing the toroidal Fourier mode numbers. Shape: (ndofs)
        '''
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
            m_first = jnp.append(m, 0)
            n_first = jnp.append(n, 0)
            m = jnp.append(m_first, m)
            n = jnp.append(n_first, n)
        return m, n
    
    @lru_cache()
    @jit
    def Kdash_helper(self):
        r'''
        Calculates the following quantities. Caches.

        Returns
        -------
        Kdash1_sv_op: ndarray, shape (n_phi, n_theta, 3(xyz), n_dof)
        Kdash2_sv_op: ndarray, shape (n_phi, n_theta, 3(xyz), n_dof)
            Partial derivatives of K in term of Phi (current potential) harmonics.
            Shape: (n_phi, n_theta, 3(xyz), n_dof)
        Kdash1_const: ndarray, shape (n_phi, n_theta, 3(xyz))
        Kdash2_const: ndarray, shape (n_phi, n_theta, 3(xyz))
            Partial derivatives of the part of K due produced by the 
            uniform current from the net poloidal/toroidal currents. 
        '''
        normal = self.eval_surface.normal()
        gammadash1 = self.eval_surface.gammadash1()
        gammadash2 = self.eval_surface.gammadash2()
        net_poloidal_current_amperes = self.net_poloidal_current_amperes
        net_toroidal_current_amperes = self.net_toroidal_current_amperes
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
        r'''
        Calculates the following quantities. Caches. 

        Returns
        -------
        trig_m_i_n_i: ndarray, shape (n_phi, n_theta, n_dof)
        trig_diff_m_i_n_i: ndarray, shape (n_phi, n_theta, n_dof)
            IFT operator that performs IFT on an array of Fourier harmonics of
            :math:`Phi_{sv}` or its derivatives (see below).
        partial_phi: ndarray, shape (n_dof, n_dof)
        partial_theta: ndarray, shape (n_dof, n_dof)
        partial_phi_phi: ndarray, shape (n_dof, n_dof)
        partial_phi_theta: ndarray, shape (n_dof, n_dof)
        partial_theta_theta: ndarray, shape (n_dof, n_dof)
            Partial derivative operators that works by multiplying the harmonic
            coefficients of :math:`Phi_{sv}` by its harmonic number and a sign, depending whether
            the coefficient is sin or cos. DOES NOT RE-ORDER the coefficients
            into the simsopt conventions. Therefore, IFT for such derivatives 
            must be performed with trig_m_i_n_i and trig_diff_m_i_n_i (see above).
        '''
        nfp = self.nfp
        cp_m, cp_n = self.make_mn()
        if winding_surface_mode:
            quadpoints_phi = self.winding_surface.quadpoints_phi
            quadpoints_theta = self.winding_surface.quadpoints_theta
        else:    
            quadpoints_phi = self.quadpoints_phi
            quadpoints_theta = self.quadpoints_theta
        # The uniform index for phi contains first sin Fourier 
        # coefficients, then optionally cos is stellsym=False.
        n_harmonic = len(cp_m)
        iden = jnp.identity(n_harmonic)
        # Shape: (n_phi, n_theta)
        phi_grid = jnp.pi*2*quadpoints_phi[:, None]
        theta_grid = jnp.pi*2*quadpoints_theta[None, :]
        # When stellsym is enabled, Phi is a sin fourier series.
        # After a derivative, it becomes a cos fourier series.
        if self.stellsym:
            trig_choice = 1
        # Otherwise, it's a sin-cos series. After a derivative,
        # it becomes a cos-sin series.
        else:
            trig_choice = jnp.append(jnp.repeat(jnp.array([1,-1]), n_harmonic//2), -1)
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
    
def cp_ndofs(stellsym, mpol, ntor):
    ndofs = 2 * mpol * ntor + mpol + ntor
    if stellsym:
        ndofs = ndofs
        ndofs_half = ndofs
    else:
        ndofs_half = ndofs
        ndofs = 2 * ndofs + 1
    return(ndofs, ndofs_half)

@tree_util.register_pytree_node_class
class QuadcoilParamsFiniteElement(_Params):
    def __init__(
        self,
        plasma_surface, 
        winding_surface, 
        net_poloidal_current_amperes: float, 
        net_toroidal_current_amperes: float,
        Bnormal_plasma=None,
        quadpoints_phi=None,
        quadpoints_theta=None, 
        stellsym=None
        ):
        
        # Writing peroperties 
        super().__init__(
            plasma_surface=plasma_surface,
            winding_surface=winding_surface,
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            net_toroidal_current_amperes=net_toroidal_current_amperes,
            Bnormal_plasma=Bnormal_plasma,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            stellsym=stellsym,
        )
        

    # -- JAX prereqs --
    # Update this if you ever change the constructor!
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            plasma_surface=children[0],
            winding_surface=children[1],
            net_poloidal_current_amperes=children[3],
            net_toroidal_current_amperes=children[4],
            Bnormal_plasma=children[5],
            quadpoints_phi=children[6],
            quadpoints_theta=children[7],
            stellsym=aux_data['stellsym']
        )

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
            'ndofs': self.ndofs,
            'ndofs_half': self.ndofs_half,
            # 'm': self.m,
            # 'n': self.n,
        }
        return children, aux_data
