import sys
sys.path.insert(1,'..')
import jax.numpy as jnp
from quadcoil import sin_or_cos
from jax import jit
from functools import partial
'''
This file includes some surface quantities that can 
potentially be reused by many objective functions,
such as K dot grad K and surface self force.
The inputs are arrays so that the code is easy to port into 
c++.
'''
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




