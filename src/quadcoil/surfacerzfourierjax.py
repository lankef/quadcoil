import jax.numpy as jnp
from functools import partial, lru_cache
from jax import jit, tree_util
from jax.scipy.special import factorial

@tree_util.register_pytree_node_class
class SurfaceRZFourierJAX:
    def __init__(self, nfp: int, stellsym: bool, mpol: int, ntor: int, 
                 quadpoints_phi: jnp.ndarray, quadpoints_theta: jnp.ndarray, dofs: jnp.ndarray):
        if quadpoints_phi.ndim != 1 or quadpoints_theta.ndim != 1 or dofs.ndim != 1:
            raise ValueError('quadpoints_phi, quadpoints_theta, dofs must all be 1D arrays.')
        
        self.nfp = nfp
        self.stellsym = stellsym
        self.mpol = mpol
        self.ntor = ntor
        self.dofs = dofs
        self.quadpoints_phi = quadpoints_phi
        self.quadpoints_theta = quadpoints_theta
        self.theta_mesh, self.phi_mesh = jnp.meshgrid(quadpoints_theta, quadpoints_phi)

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

    def get_dofs(self):
        return(self.dofs.copy())

    ''' Gamma '''
    # Attention, gamma() evaluations are NOT cached 
    # like in simsopt to make the class look 
    # similar to simsopt. The caching occurs in 
    # QuadcoilState.
    # Note: LRU cache must be applied outside of jit.
    @lru_cache()
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

    @lru_cache()
    @jit
    def normal(self):
        return jnp.cross(self.gammadash1(), self.gammadash2(), axis=-1)

    @lru_cache()
    @jit
    def unitnormal(self):
        normal = self.normal()
        return normal/jnp.linalg.norm(normal, axis=-1)[:, :, None]
    
    ''' JAX prereqs '''
    
    def tree_flatten(self):
        children = (
            self.quadpoints_phi,
            self.quadpoints_theta,
            self.dofs,
            self.theta_mesh,
            self.phi_mesh,
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


''' Backends '''
def dof_to_rz_op(
        phi_grid, theta_grid, 
        nfp, stellsym,
        dash1_order=0, dash2_order=0,
        mpol:int=10, ntor:int=10):
    # maps a [ndof] array
    # to a [nphi, ntheta, 2(r, z)] array.
    m_c = jnp.concatenate([
        jnp.zeros(ntor+1),
        jnp.repeat(jnp.arange(1, mpol+1), ntor*2+1)
    ])
    m_s = jnp.concatenate([
        jnp.zeros(ntor),
        jnp.repeat(jnp.arange(1, mpol+1), ntor*2+1)
    ])
    n_c = jnp.concatenate([
        jnp.arange(0, ntor+1),
        jnp.tile(jnp.arange(-ntor, ntor+1), mpol)
    ])
    n_s = jnp.concatenate([
        jnp.arange(1, ntor+1),
        jnp.tile(jnp.arange(-ntor, ntor+1), mpol)
    ])
    
    total_neg = (dash1_order + dash2_order)//2
    derivative_factor_c = (
        (- n_c[:, None, None] * jnp.pi * 2 * nfp) ** dash1_order 
        * (m_c[:, None, None] * jnp.pi * 2      ) ** dash2_order
    ) * (-1) ** total_neg
    derivative_factor_s = (
        (- n_s[:, None, None] * jnp.pi * 2 * nfp) ** dash1_order 
        * (m_s[:, None, None] * jnp.pi * 2      ) ** dash2_order
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
            m_c[:, None, None] * jnp.pi * 2 * theta_grid[None, :, :]
            - n_c[:, None, None] * jnp.pi * 2 * nfp * phi_grid[None, :, :]
        )
        smn = derivative_factor_s * jnp.sin(
            m_s[:, None, None] * jnp.pi * 2 * theta_grid[None, :, :]
            - n_s[:, None, None] * jnp.pi * 2 * nfp * phi_grid[None, :, :]
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
            m_c[:, None, None] * theta_grid[None, :, :] * jnp.pi * 2
            - n_c[:, None, None] * phi_grid[None, :, :] * jnp.pi * 2 * nfp
        )
        smn = derivative_factor_s * jnp.cos(
            m_s[:, None, None] * theta_grid[None, :, :] * jnp.pi * 2
            - n_s[:, None, None] * phi_grid[None, :, :] * jnp.pi * 2 * nfp
        )
    m_2_n_2 = jnp.concatenate([m_c, m_s]) ** 2 + jnp.concatenate([n_c, n_s]) ** 2
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
    '''
    Generates an operator with shape [ndof, nphi, ntheta, 3(x, y, z)]
    that calculates gamma from a surface's dofs.
    '''
    dof_to_x = 0
    dof_to_y = 0
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
