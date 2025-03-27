import sys
sys.path.insert(1,'..')
import jax.numpy as jnp
from proto_utils import sin_or_cos, avg_order_of_magnitude
from jax import jit
from functools import partial
'''
This file includes some surface quantities that can 
potentially be reused by many objective functions,
such as K dot grad K and surface self force.
The inputs are arrays so that the code is easy to port into 
c++.
'''
@jit
def grad_helper(gammadash1, gammadash2):
    '''
    This is a helper method that calculates the contravariant 
    vectors, grad phi and grad theta, using the curvilinear coordinate 
    identities:
    - grad1: grad phi = (dg2 x (dg1 x dg2))/|dg1 x dg2|^2
    - grad2: grad theta = -(dg1 x (dg2 x dg1))/|dg1 x dg2|^2
    Shape: (n_phi, n_theta, 3(xyz))
    '''
    dg2 = gammadash2
    dg1 = gammadash1
    dg1xdg2 = jnp.cross(dg1, dg2, axis=-1)
    denom = jnp.sum(dg1xdg2**2, axis=-1)
    # grad phi
    grad1 = jnp.cross(dg2, dg1xdg2, axis=-1)/denom[:,:,None]
    # grad theta
    grad2 = jnp.cross(dg1, -dg1xdg2, axis=-1)/denom[:,:,None]
    return(grad1, grad2)

@jit
def norm_helper(vec):
    '''
    This is a helper method that calculates the following quantities:
    - normN_prime_2d: The normal vector's length, |N|
    - inv_normN_prime_2d: 1/|N|
    Shape: (n_phi, n_theta, 3(xyz)); (n_phi, n_theta); (n_phi, n_theta)
    '''
    # Length of the non-unit WS normal vector |N|,
    # its inverse (1/|N|) and its inverse's derivatives
    # w.r.t. phi(phi) and theta
    # Not to be confused with the normN (plasma surface Jacobian)
    # in Regcoil.
    norm = jnp.linalg.norm(vec, axis=-1) # |N|
    inv_norm = 1/norm # 1/|N|
    return(
        norm,
        inv_norm
    )

@jit
def dga_inv_n_dashb(
    normal,
    gammadash1,
    gammadash2,
    gammadash1dash1,
    gammadash1dash2,
    gammadash2dash2,
):
    ''' 
    This is a helper method that calculates the following quantities:
    - dg1_inv_n_dash1: d[(1/|n|)(dgamma/dphi)]/dphi
    - dg1_inv_n_dash2: d[(1/|n|)(dgamma/dphi)]/dtheta
    - dg2_inv_n_dash1: d[(1/|n|)(dgamma/dtheta)]/dphi
    - dg2_inv_n_dash2: d[(1/|n|)(dgamma/dtheta)]/dtheta
    Shape: (n_phi, n_theta, 3(xyz))
    '''
    # gammadash1() calculates partial r/partial phi. Keep in mind that the angles
    # in simsopt go from 0 to 1.
    # Shape: (n_phi, n_theta, 3(xyz))
    dg1 = gammadash1
    dg2 = gammadash2
    dg11 = gammadash1dash1
    dg12 = gammadash1dash2
    dg22 = gammadash2dash2

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

@jit
def unitnormaldash(
        normal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
    ):
    ''' 
    This is a helper method that calculates the following quantities:
    - unitnormaldash1: d unitnormal/dphi
    - unitnormaldash2: d unitnormal/dtheta
    Shape: (n_phi, n_theta, 3(xyz))
    '''
    _, inv_normN_prime_2d = norm_helper(normal)
    (
        dg1_inv_n_dash1, dg1_inv_n_dash2,
        _, _ # dg2_inv_n_dash1, dg2_inv_n_dash2
    ) = dga_inv_n_dashb(
        normal=normal,
        gammadash1=gammadash1,
        gammadash2=gammadash2,
        gammadash1dash1=gammadash1dash1,
        gammadash1dash2=gammadash1dash2,
        gammadash2dash2=gammadash2dash2,
    )

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

@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
])
def diff_helper(
        nfp, cp_m, cp_n, 
        quadpoints_phi,
        quadpoints_theta,
        stellsym,
    ):
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
    '''
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

@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
])
def Kdash_helper(
        normal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        nfp, cp_m, cp_n,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        stellsym):
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
    normN_prime_2d, _ = norm_helper(normal)
    (
        trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        partial_phi_phi,
        partial_phi_theta,
        partial_theta_theta,
    ) = diff_helper(
        nfp, cp_m, cp_n,
        quadpoints_phi,
        quadpoints_theta,
        stellsym,
    )
    # Some quantities
    (
        dg1_inv_n_dash1,
        dg1_inv_n_dash2,
        dg2_inv_n_dash1,
        dg2_inv_n_dash2 
    ) = dga_inv_n_dashb(
        normal=normal,
        gammadash1=gammadash1,
        gammadash2=gammadash2,
        gammadash1dash1=gammadash1dash1,
        gammadash1dash2=gammadash1dash2,
        gammadash2dash2=gammadash2dash2,
    )
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

@partial(jit, static_argnames=[
    'normalize',
])
def A_b_c_to_block_operator(A, b, c, current_scale, normalize):
    '''
    Converts a set of A, b, c that gives 
    f(p) = pTAp + bTp +c

    into a matrix consisting of 
    O = [
        [(AT+A)/2 / (current_scale**2), b/2 / current_scale],
        [bT/2     /  current_scale,     c  ]
    ]
    That satisfy 
    f(p) = tr(OX) 
    for X = (p, 1)(p, 1)^T
    '''
    O = jnp.block([
        [(A+A.swapaxes(-1, -2))    /2/(current_scale**2), jnp.expand_dims(b, axis=-1)       /2/current_scale],
        [jnp.expand_dims(b, axis=-2)/2/ current_scale,     jnp.expand_dims(c, axis=(-1, -2))                 ]
    ])
    if normalize:
        out_scale = avg_order_of_magnitude(O)
        O /= out_scale
    else: 
        out_scale = 1
    return(O, out_scale)
    
@partial(jit, static_argnames=[
    'nfp',
])
def integrate_operators(A, b, c, nfp, normal, quadpoints_phi, quadpoints_theta):
    if (
        normal.shape[:2] != A.shape[:2] 
        or normal.shape[:2] != b.shape[:2] 
        or normal.shape[:2] != c.shape[:2]
        or normal.shape[0] != len(quadpoints_phi)
        or normal.shape[1] != len(quadpoints_theta)
    ):
        raise ValueError(
            'Shapes of A, b, c, normal and quadpoints does not match.'
            + 'A.shape: ' + str(A.shape) + ', '
            + 'b.shape: ' + str(b.shape) + ', '
            + 'c.shape: ' + str(c.shape) + ', '
            + 'normal.shape: ' + str(normal.shape) + ', '
            + 'quadpoints_phi.shape: ' + str(quadpoints_phi.shape) + ', '
            + 'quadpoints_theta.shape: ' + str(quadpoints_theta.shape)
        )
    normN = jnp.linalg.norm(normal, axis=-1)
    dzeta = (quadpoints_phi[1] - quadpoints_phi[0])
    dtheta = (quadpoints_theta[1] - quadpoints_theta[0])
    A_int = jnp.sum(A * dzeta * dtheta * normN[:, :, None, None], axis=(0, 1)) * nfp
    b_int = jnp.sum(b * dzeta * dtheta * normN[:, :, None], axis=(0, 1)) * nfp
    c_int = jnp.sum(c * dzeta * dtheta * normN, axis=(0, 1)) * nfp
    return(A_int, b_int, c_int)



