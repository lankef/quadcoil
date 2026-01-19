import jax.numpy as jnp
from functools import partial, lru_cache
from jax import jit, tree_util
from jax.scipy.special import factorial
from .math_utils import norm_helper, is_ndarray


@tree_util.register_pytree_node_class
class SurfaceRZFourierJAX:
    def __init__(self, nfp: int, stellsym: bool, mpol: int, ntor: int, 
                 quadpoints_phi: jnp.ndarray, quadpoints_theta: jnp.ndarray, dofs: jnp.ndarray):
        if not is_ndarray(quadpoints_phi, 1):
            raise TypeError('quadpoints_phi has incorrect type or shape: ' + str(type(quadpoints_phi)))    
        if not is_ndarray(quadpoints_theta, 1):
            raise TypeError('quadpoints_phi has incorrect type or shape: ' + str(type(quadpoints_theta)))    
        if not is_ndarray(dofs, 1):
            raise TypeError('quadpoints_phi has incorrect type or shape: ' + str(type(dofs)))    
        self.nfp = nfp
        self.stellsym = stellsym
        self.mpol = mpol
        self.ntor = ntor
        self.dofs = dofs
        self.quadpoints_phi = quadpoints_phi
        self.quadpoints_theta = quadpoints_theta
        self.theta_mesh, self.phi_mesh = jnp.meshgrid(quadpoints_theta, quadpoints_phi)
        self.dphi = (quadpoints_phi[1] - quadpoints_phi[0])
        self.dtheta = (quadpoints_theta[1] - quadpoints_theta[0])
    
    def copy_and_set_quadpoints(self, quadpoints_phi, quadpoints_theta):
        return SurfaceRZFourierJAX(
            nfp=self.nfp, 
            stellsym=self.stellsym, 
            mpol=self.mpol, 
            ntor=self.ntor, 
            quadpoints_phi=quadpoints_phi, 
            quadpoints_theta=quadpoints_theta, 
            dofs=self.dofs
        )

    def from_simsopt(simsopt_surf):
        return SurfaceRZFourierJAX(
            nfp=simsopt_surf.nfp,
            stellsym=simsopt_surf.stellsym,
            mpol=simsopt_surf.mpol,
            ntor=simsopt_surf.ntor,
            quadpoints_phi=simsopt_surf.quadpoints_phi,
            quadpoints_theta=simsopt_surf.quadpoints_theta,
            dofs=simsopt_surf.get_dofs(),
        )

    def to_simsopt(self):
        try:
            from simsopt.geo import SurfaceRZFourier
        except:
            raise ModuleNotFoundError('Simsopt must be installed to export surface to simsopt with to_simsopt().')
        surf = SurfaceRZFourier(
            nfp=self.nfp,
            stellsym=self.stellsym,
            mpol=self.mpol,
            ntor=self.ntor,
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
        )
        surf.set_dofs(self.dofs)
        return surf
    
    def plot(self, **kwargs):
        try:
            self.to_simsopt().plot(**kwargs)
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Simsopt must be installed to use plot().')

    # May not be jittable
    def from_desc(desc_surf, quadpoints_phi, quadpoints_theta):
        try:
            from desc.vmec_utils import ptolemy_identity_rev
        except:
            raise ModuleNotFoundError('DESC must be installed to load surface from DESC.')
        # mm, nn is the same as mc, nc from make_rzfourier_mc_ms_nc_ns
        # ms, ns are just mc, nc with the first zero removed.
        # R modes
        # R_basis_modes[:,1] is desc_surf.R_basis.modes[:,1]
        # ptolemy_identity_rev has mutable shape, but there are only 3 possibilities:
        # all sin, all cos, and sin + cos.
        # These possibilities are known before-hand from stellarator symmetry.
        mm, nn, rs_raw, rc_raw = ptolemy_identity_rev(desc_surf.R_basis.modes[:,1], desc_surf.R_basis.modes[:,2], desc_surf.R_lmn)
        # Z modes
        mm, nn, zs_raw, zc_raw = ptolemy_identity_rev(desc_surf.Z_basis.modes[:,1], desc_surf.Z_basis.modes[:,2], desc_surf.Z_lmn)
        # Loading basic properties
        mpol = desc_surf.M
        ntor = desc_surf.N
        stellsym = desc_surf.sym
        nfp = desc_surf.NFP
        # Stellsym SurfaceRZFourier's dofs consists of 
        # [rc, zs]
        # Non-stellsym SurfaceRZFourier's dofs consists of 
        # [rc, rs, zc, zs]
        # Because rs, zs from ptolemy_identity_rev shares the same m, n 
        # arrays as rc, zc, they both have a zero as the first element 
        # that need to be removed.
        rc = rc_raw.flatten()
        rs = rs_raw.flatten()[1:]
        zc = zc_raw.flatten()
        zs = zs_raw.flatten()[1:]
        if stellsym:
            dofs = jnp.concatenate([rc, zs])
        else:
            dofs = jnp.concatenate([rc, rs, zc, zs])

        return SurfaceRZFourierJAX(
            nfp=nfp,
            stellsym=stellsym,
            mpol=mpol,
            ntor=ntor,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            dofs=dofs,
        )
    
    def to_desc(self):
        try:
            from desc.vmec_utils import ptolemy_identity_fwd
            from desc.geometry import FourierRZToroidalSurface
        except:
            raise ModuleNotFoundError('DESC must be installed to export surface to DESC.')
        if self.stellsym:
            # The dofs contain rc, zs, and rc has one more element than zs.
            len_sin = len(self.dofs)//2
            rc = self.dofs[:-len_sin]
            zs = self.dofs[-len_sin:]
            zs = jnp.insert(zs, 0, 0.)
            zc = jnp.zeros_like(rc)
            rs = jnp.zeros_like(rc)
        else:
            # The dofs contain rc, rs, zc, zs, and rc has one more element than zs.
            half_len = len(self.dofs)//2
            len_sin = half_len//2
            rcrs = self.dofs[:half_len//2]
            zczs = self.dofs[half_len//2:]
            rc = rcrs[:-len_sin]
            rs = rcrs[-len_sin:]
            rs = jnp.insert(rs, 0, 0.)
            zc = zczs[:-len_sin]
            zs = zczs[-len_sin:]
            zs = jnp.insert(zs, 0, 0.)
        mc, _, nc, _ = make_rzfourier_mc_ms_nc_ns(self.mpol, self.ntor)
        Rm, Rn, R_lmn = ptolemy_identity_fwd(mc, nc, rs, rc)
        Zm, Zn, Z_lmn = ptolemy_identity_fwd(mc, nc, zs, zc)
        R_lmn = R_lmn.flatten()
        Z_lmn = Z_lmn.flatten()
        modes_R = jnp.vstack([Rm, Rn]).T
        modes_Z = jnp.vstack([Zm, Zn]).T
        return FourierRZToroidalSurface(
            R_lmn, Z_lmn, 
            modes_R.astype(int), modes_Z.astype(int), 
            NFP=self.nfp, sym=self.stellsym, 
            M=self.mpol, N=self.ntor, rho=1
        )

    def get_dofs(self):
        return(self.dofs.copy())

    # ''' Gamma '''
    
    # Attention, gamma() evaluations are NOT cached 
    # like in simsopt to make the class look 
    # similar to simsopt. The caching occurs in 
    # QuadcoilState.
    # Note: LRU cache must be applied outside of jit.
    # @lru_cache()
    @partial(jit, static_argnames=['a', 'b'])
    def gammadash(self, a:int, b:int):
        return dof_to_gamma(
            dofs=self.dofs,
            phi_grid=self.phi_mesh, 
            theta_grid=self.theta_mesh, 
            nfp=self.nfp, 
            stellsym=self.stellsym, 
            dash1_order=a, 
            dash2_order=b,
            mpol=self.mpol, 
            ntor=self.ntor
        )
        
    gamma = lambda self: self.gammadash(0, 0)
    gammadash1 = lambda self: self.gammadash(1, 0)
    gammadash2 = lambda self: self.gammadash(0, 1)
    gammadash1dash1 = lambda self: self.gammadash(2, 0)
    gammadash1dash2 = lambda self: self.gammadash(1, 1)
    gammadash2dash2 = lambda self: self.gammadash(0, 2)

    # @lru_cache()
    @jit
    def normal(self):
        return jnp.cross(self.gammadash1(), self.gammadash2(), axis=-1)

    # @lru_cache()
    @jit
    def unitnormal(self):
        normal = self.normal()
        return normal/jnp.linalg.norm(normal, axis=-1)[:, :, None]

    # ''' Other cached quantities '''
    # @lru_cache()
    @jit
    def da(self):
        # For surface integrating a quantity sampled over this surface.
        dphi = self.dphi
        dtheta = self.dtheta
        normN = jnp.linalg.norm(self.normal(), axis=-1)
        return dphi * dtheta * normN

    @jit
    def integrate(self, scalar_field):
        # For integrating a scalar field over this surface.
        return jnp.sum(scalar_field * self.da())
    
    # @lru_cache()
    @jit
    def area(self):
        # For integrating a scalar field over this surface.
        return jnp.sum(self.da())
    
    # @lru_cache()
    @jit
    def grad_helper(self):
        # '''
        # This is a helper method that calculates the contravariant 
        # vectors, grad phi and grad theta, using the curvilinear coordinate 
        # identities:
        # - grad1: grad phi = (dg2 x (dg1 x dg2))/|dg1 x dg2|^2
        # - grad2: grad theta = -(dg1 x (dg2 x dg1))/|dg1 x dg2|^2
        # Shape: (n_phi, n_theta, 3(xyz))
        # '''
        dg2 = self.gammadash2()
        dg1 = self.gammadash1()
        dg1xdg2 = jnp.cross(dg1, dg2, axis=-1)
        denom = jnp.sum(dg1xdg2**2, axis=-1)
        # grad phi
        grad1 = jnp.cross(dg2, dg1xdg2, axis=-1)/denom[:,:,None]
        # grad theta
        grad2 = jnp.cross(dg1, -dg1xdg2, axis=-1)/denom[:,:,None]
        return (grad1, grad2)
    
    # @lru_cache()
    @jit
    def unitnormaldash(self):
        # ''' 
        # This is a helper method that calculates the following quantities:
        # - unitnormaldash1: d unitnormal/dphi
        # - unitnormaldash2: d unitnormal/dtheta
        # Shape: (n_phi, n_theta, 3(xyz))
        # '''
        normal = self.normal()
        gammadash1 = self.gammadash1()
        gammadash2 = self.gammadash2()
        gammadash1dash2 = self.gammadash1dash2()
        gammadash2dash2 = self.gammadash2dash2()
        _, inv_normN_prime_2d = norm_helper(normal)
        (
            dg1_inv_n_dash1, dg1_inv_n_dash2,
            _, _ # dg2_inv_n_dash1, dg2_inv_n_dash2
        ) = self.dga_inv_n_dashb()
    
        dg2 = gammadash2
        dg1_inv_n = gammadash1 * inv_normN_prime_2d[:, :, None]
        dg22 = gammadash2dash2
        dg12 = gammadash1dash2
        unitnormaldash1 = (
            jnp.cross(dg1_inv_n_dash1, dg2, axis=-1)
            + jnp.cross(dg1_inv_n, dg12, axis=-1)
        )
        unitnormaldash2 = (
            jnp.cross(dg1_inv_n_dash2, dg2, axis=-1)
            + jnp.cross(dg1_inv_n, dg22, axis=-1)
        )
        return(unitnormaldash1, unitnormaldash2)

    # @lru_cache()
    @jit
    def dga_inv_n_dashb(self):
        # ''' 
        # This is a helper method that calculates the following quantities:
        # - dg1_inv_n_dash1: d[(1/|n|)(dgamma/dphi)]/dphi
        # - dg1_inv_n_dash2: d[(1/|n|)(dgamma/dphi)]/dtheta
        # - dg2_inv_n_dash1: d[(1/|n|)(dgamma/dtheta)]/dphi
        # - dg2_inv_n_dash2: d[(1/|n|)(dgamma/dtheta)]/dtheta
        # Shape: (n_phi, n_theta, 3(xyz))
        # '''
        normal = self.normal()
        # gammadash1() calculates partial r/partial phi. Keep in mind that the angles
        # in simsopt go from 0 to 1.
        # Shape: (n_phi, n_theta, 3(xyz))
        dg1 = self.gammadash1()
        dg2 = self.gammadash2()
        dg11 = self.gammadash1dash1()
        dg12 = self.gammadash1dash2()
        dg22 = self.gammadash2dash2()
    
        # Because Phi is defined around the unit normal, rather 
        # than N, we need to calculate the derivative and double derivative 
        # of (dr/dtheta)/|N| and (dr/dphi)/|N|.
        # phi (phi) derivative of the normal's length
        normaldash1 = (
            jnp.cross(dg11, dg2)
            + jnp.cross(dg1, dg12)
        )
    
        # Theta derivative of the normal's length
        normaldash2 = (
            jnp.cross(dg12, dg2)
            + jnp.cross(dg1, dg22)
        )
        normal_vec = normal
        _, inv_normN_prime_2d = norm_helper(normal_vec)
    
        # Derivatives of 1/|N|:
        # d/dx(1/sqrt(f(x)^2 + g(x)^2 + h(x)^2)) 
        # = (-f(x)f'(x) - g(x)g'(x) - h(x)h'(x))
        # /(f(x)^2 + g(x)^2 + h(x)^2)^(3/2)
        denominator = jnp.sum(normal_vec**2, axis=-1)**1.5
        nominator_inv_normN_prime_2d_dash1 = -jnp.sum(normal_vec*normaldash1, axis=-1)
        nominator_inv_normN_prime_2d_dash2 = -jnp.sum(normal_vec*normaldash2, axis=-1)
        inv_normN_prime_2d_dash1 = nominator_inv_normN_prime_2d_dash1/denominator
        inv_normN_prime_2d_dash2 = nominator_inv_normN_prime_2d_dash2/denominator
        
        # d[(1/|n|)(dgamma/dphi)]/dphi
        dg1_inv_n_dash1 = dg11*inv_normN_prime_2d[:,:,None] + dg1*inv_normN_prime_2d_dash1[:,:,None] 
        # d[(1/|n|)(dgamma/dphi)]/dtheta
        dg1_inv_n_dash2 = dg12*inv_normN_prime_2d[:,:,None] + dg1*inv_normN_prime_2d_dash2[:,:,None] 
        # d[(1/|n|)(dgamma/dtheta)]/dphi
        dg2_inv_n_dash1 = dg12*inv_normN_prime_2d[:,:,None] + dg2*inv_normN_prime_2d_dash1[:,:,None] 
        # d[(1/|n|)(dgamma/dtheta)]/dtheta
        dg2_inv_n_dash2 = dg22*inv_normN_prime_2d[:,:,None] + dg2*inv_normN_prime_2d_dash2[:,:,None] 
        return(
            dg1_inv_n_dash1,
            dg1_inv_n_dash2,
            dg2_inv_n_dash1,
            dg2_inv_n_dash2 
        )
    
    # ''' JAX prereqs '''
    
    def tree_flatten(self):
        children = (
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.dofs,
            self.theta_mesh,
            self.phi_mesh,
            self.dphi,
            self.dtheta,
        )
        aux_data = {
            'nfp': self.nfp,
            'stellsym': self.stellsym,
            'mpol': self.mpol,
            'ntor': self.ntor,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            nfp=aux_data['nfp'],
            stellsym=aux_data['stellsym'],
            mpol=aux_data['mpol'],
            ntor=aux_data['ntor'],
            quadpoints_phi=children[0],
            quadpoints_theta=children[1],
            dofs=children[2],
        )


@partial(jit, static_argnames=['mpol', 'ntor'])
def make_rzfourier_mc_ms_nc_ns(mpol:int, ntor:int):
    # Make a list of mc, ms, nc, ns
    ms = jnp.concatenate([
        jnp.zeros(ntor),
        jnp.repeat(jnp.arange(1, mpol+1), ntor*2+1)
    ])
    ns = jnp.concatenate([
        jnp.arange(1, ntor+1),
        jnp.tile(jnp.arange(-ntor, ntor+1), mpol)
    ])
    mc = jnp.concatenate([jnp.zeros(1), ms])
    nc = jnp.concatenate([jnp.zeros(1), ns])
    return mc, ms, nc, ns

# ''' Backends '''
@partial(jit, static_argnames=['nfp', 'stellsym', 'dash1_order', 'dash2_order', 'mpol', 'ntor'])
def dof_to_rz_op(
        phi_grid, theta_grid, 
        nfp:int, stellsym:bool,
        dash1_order=0, dash2_order=0,
        mpol:int=10, ntor:int=10):
    # maps a [ndof] array
    # to a [nphi, ntheta, 2(r, z)] array.
    mc, ms, nc, ns = make_rzfourier_mc_ms_nc_ns(mpol, ntor)
    total_neg = (dash1_order + dash2_order)//2
    derivative_factor_c = (
        (- nc[:, None, None] * jnp.pi * 2 * nfp) ** dash1_order 
        * (mc[:, None, None] * jnp.pi * 2      ) ** dash2_order
    ) * (-1) ** total_neg
    derivative_factor_s = (
        (- ns[:, None, None] * jnp.pi * 2 * nfp) ** dash1_order 
        * (ms[:, None, None] * jnp.pi * 2      ) ** dash2_order
    ) * (-1) ** total_neg
    if (dash1_order + dash2_order)%2 == 0:
        # 2 arrays of shape [*stack of mn, nphi, ntheta]
        # consisting of 
        # [
        #     cos(m theta - nfp * n * phi)
        #     ...
        # ]
        # [
        #     sin(m theta - nfp * n * phi)
        #     ...
        # ]
        cmn = derivative_factor_c * jnp.cos(
            mc[:, None, None] * jnp.pi * 2 * theta_grid[None, :, :]
            - nc[:, None, None] * jnp.pi * 2 * nfp * phi_grid[None, :, :]
        )
        smn = derivative_factor_s * jnp.sin(
            ms[:, None, None] * jnp.pi * 2 * theta_grid[None, :, :]
            - ns[:, None, None] * jnp.pi * 2 * nfp * phi_grid[None, :, :]
        )
    else:
        # For odd-order derivatives, 
        # consisting of 
        # [
        #     sin(m theta - nfp * n * phi)
        #     ...
        # ]
        # [
        #     cos(m theta - nfp * n * phi)
        #     ...
        # ]
        cmn = -derivative_factor_c * jnp.sin(
            mc[:, None, None] * theta_grid[None, :, :] * jnp.pi * 2
            - nc[:, None, None] * phi_grid[None, :, :] * jnp.pi * 2 * nfp
        )
        smn = derivative_factor_s * jnp.cos(
            ms[:, None, None] * theta_grid[None, :, :] * jnp.pi * 2
            - ns[:, None, None] * phi_grid[None, :, :] * jnp.pi * 2 * nfp
        )
    m_2_n_2 = jnp.concatenate([mc, ms]) ** 2 + jnp.concatenate([nc, ns]) ** 2
    if not stellsym:
        m_2_n_2 = jnp.tile(m_2_n_2, 2)
    # Stellsym SurfaceRZFourier's dofs consists of 
    # [rc, zs]
    # Non-stellsym SurfaceRZFourier's dofs consists of 
    # [rc, rs, zc, zs]
    # Here we construct operators that maps 
    # dofs -> r and z of known, valid, expanded quadpoints.
    if stellsym:
        r_operator = cmn
        z_operator = smn
    else:
        r_operator = jnp.concatenate([cmn, smn], axis=0)
        z_operator = jnp.concatenate([cmn, smn], axis=0)
    r_operator_padded = jnp.concatenate([r_operator, jnp.zeros_like(z_operator)], axis=0)
    z_operator_padded = jnp.concatenate([jnp.zeros_like(r_operator), z_operator], axis=0)

    # overall operator
    # has shape 
    # [nphi, ntheta, 2(r, z), ndof]
    # maps a [ndof] array
    # to a [nphi, ntheta, 2(r, z)] array.
    A_lstsq = jnp.concatenate([r_operator_padded[:, :, :, None], z_operator_padded[:, :, :, None]], axis=3)
    A_lstsq = jnp.moveaxis(A_lstsq, 0, -1)
    return A_lstsq, m_2_n_2

def dof_to_gamma_op(
    phi_grid, theta_grid, 
    nfp, stellsym,
    dash1_order=0, dash2_order=0,
    mpol:int=10, ntor:int=10):
    # '''
    # Generates an operator with shape [ndof, nphi, ntheta, 3(x, y, z)]
    # that calculates gamma from a surface's dofs.
    # '''
    dof_to_x = 0.
    dof_to_y = 0.
    for dash1_order_rz in range(dash1_order + 1):
        # Applying chain rule to 
        # dof_to_r * jnp.cos(phi_grid * jnp.pi * 2)[:, :, None].
        dash1_order_trig = dash1_order - dash1_order_rz
        # Shape: [nphi, ntheta, 2(r, z), ndof]
        dof_to_rz_dash, _ = dof_to_rz_op(
            phi_grid=phi_grid, 
            theta_grid=theta_grid, 
            nfp=nfp, 
            stellsym=stellsym,
            dash1_order=dash1_order_rz,
            dash2_order=dash2_order,
            mpol=mpol, 
            ntor=ntor
        )
        # Shape: [ndof, nphi, ntheta]
        dof_to_r_dash = dof_to_rz_dash[:, :, 0, :]
        if dash1_order_rz == dash1_order:
            dof_to_z = dof_to_rz_dash[:, :, 1, :]
        total_neg = dash1_order_trig//2
        # Calculating the binomial coefficient (n k)
        # n: total order + 1
        # k: dash1_order_rz + 1
        binomial_coef = factorial(dash1_order) / factorial(dash1_order_rz) / factorial(dash1_order_trig)
        derivative_factor = binomial_coef * (-1)**total_neg * (jnp.pi * 2)**dash1_order_trig
        if dash1_order_trig%2 == 0:
            dof_to_x += derivative_factor * dof_to_r_dash * jnp.cos(phi_grid * jnp.pi * 2)[:, :, None]
            dof_to_y += derivative_factor * dof_to_r_dash * jnp.sin(phi_grid * jnp.pi * 2)[:, :, None]
        else:
            dof_to_x += -derivative_factor * dof_to_r_dash * jnp.sin(phi_grid * jnp.pi * 2)[:, :, None]
            dof_to_y += derivative_factor * dof_to_r_dash * jnp.cos(phi_grid * jnp.pi * 2)[:, :, None]
    dof_to_gamma = jnp.concatenate([dof_to_x[:, :, None, :], dof_to_y[:, :, None, :], dof_to_z[:, :, None, :]], axis=2)
    return dof_to_gamma

def dof_to_gamma(
    dofs, phi_grid, theta_grid, 
    nfp, stellsym,
    dash1_order=0, dash2_order=0,
    mpol:int=10, ntor:int=10):
    return(
        dof_to_gamma_op(
            phi_grid=phi_grid, 
            theta_grid=theta_grid, 
            nfp=nfp, 
            stellsym=stellsym,
            dash1_order=dash1_order, 
            dash2_order=dash2_order,
            mpol=mpol, 
            ntor=ntor
        ) @ dofs
    )

