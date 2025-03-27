import cvxpy
from proto_utils import sdp_helper_get_elem
import numpy as np
from proto_f_b_and_k_operators import K_theta
from proto_operator_helper import A_b_c_to_block_operator


def cvxpy_create_X(n_dof,):
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    # One additional row and col each are added
    # they have blank elements except for 
    # the diagonal element.
    # X contains:
    # Phi  , Phi y, 
    # Phi y,     y, 
    X = cvxpy.Variable((n_dof+1,n_dof+1), symmetric=True)

    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]

    # This constraint sets the last diagonal item of 
    # x\otimes x to 1. When a rank-1 solution is feasible,
    # the SDP will produce a rank-1 solution. This allows
    # us to exactly control y despite having no way
    # to constrain the Phi_i y terms.
    constraints += [
        cvxpy.trace(sdp_helper_get_elem(-1, -1, n_dof+1) @ X) == 1
    ]
    return(X, constraints)

def cvxpy_no_windowpane(cp, current_scale, X):
    (A_K_theta, b_K_theta, c_K_theta) = K_theta(
        cp.net_poloidal_current_amperes,
        cp.quadpoints_phi,
        cp.quadpoints_theta,
        cp.nfp, cp.m, cp.n,
        cp.stellsym,
    )
    A_K_theta = A_K_theta.reshape((-1, A_K_theta.shape[-2], A_K_theta.shape[-1]))
    b_K_theta = b_K_theta.reshape((-1, b_K_theta.shape[-1]))
    c_K_theta = c_K_theta.flatten()
    K_theta_operator, K_theta_scale = A_b_c_to_block_operator(
        A=A_K_theta, b=b_K_theta, c=c_K_theta, 
        current_scale=current_scale,
        normalize=True
    )
    constraints = []
    if False:
    # if cp.stellsym:
        loop_size = K_theta_operator.shape[0]
    else:
        loop_size = K_theta_operator.shape[0]
    # This if statement distinguishes the sign of 
    # tot. pol. current. The total current should never
    # change sign.
    print('Testing net current sign')
    if cp.net_poloidal_current_amperes > 0:
        for i in range(loop_size):
            constraints.append(
                cvxpy.trace(
                    K_theta_operator[i, :, :] @ X
                )>=0
            )
    else:
        print('Net current is negative')
        for i in range(loop_size):
            constraints.append(
                cvxpy.trace(
                    K_theta_operator[i, :, :] @ X
                )<=0
            )
    return(constraints, K_theta_operator, K_theta_scale)

def cvxpy_create_integrated_L1_from_array(cpst, grid_3d_operator, X, stellsym):
    '''
    Constructing cvxpy constraints and variables necessary for an
    L1 norm term.
    
    -- Inputs:
    grid_3d_operator: Array, has shape (n_grid, 3, ndof+1, ndof+1)
    X: cvxpy Variable
    stellsym: Whether the grid the operator lives on has stellarator 
    symmetry.

    -- Outputs:
    constraints: A list of cvxpy constraints. 
    L1_comps_to_sum: Adding a lam*cvxpy.sum(L1_comps_to_sum) term in the 
    objective adds an L1 norm term.
    '''

    normN_prime = np.linalg.norm(cpst.winding_surface.normal(), axis=-1)
    normN_prime = normN_prime.flatten()
    # if stellsym:
    if False:
        loop_size = grid_3d_operator.shape[0]//2
        jacobian_prime = normN_prime[:normN_prime.shape[0]//cpst.winding_surface.nfp//2]
    else:
        loop_size = grid_3d_operator.shape[0]
        jacobian_prime = normN_prime[:normN_prime.shape[0]//cpst.winding_surface.nfp]
    # q is used for L1 norm.
    L1_comps_to_sum = cvxpy.Variable(loop_size*3, nonneg=True)

    constraints = []
    for i in range(loop_size):
        for j in range(3):
            # K dot nabla K L1
            if np.all(grid_3d_operator[i, j, :, :]==0):
                continue
            constraints.append(
                cvxpy.trace(
                    jacobian_prime[i] * grid_3d_operator[i, j, :, :] @ X
                )<=L1_comps_to_sum[3*i+j]
            )
            constraints.append(
                cvxpy.trace(
                    jacobian_prime[i] * grid_3d_operator[i, j, :, :] @ X
                )>=-L1_comps_to_sum[3*i+j]
            )
    # The L1 norm is given by cvxpy.sum(L1_comps_to_sum)
    return(constraints, L1_comps_to_sum)

def cvxpy_create_Linf_from_array(grid_3d_operator, X, stellsym):
    '''
    Constructing cvxpy constraints and variables necessary for an
    L-inf norm term.
    
    -- Inputs:
    grid_3d_operator: Array, has shape (n_grid, 3, ndof+1, ndof+1)
    X: cvxpy Variable
    stellsym: Whether the grid the operator lives on has stellarator 
    symmetry.

    -- Outputs:
    constraints: A list of cvxpy constraints. 
    Linf: Adding a lam*Linf term in the 
    objective adds an L1 norm term.
    '''
    # if stellsym:
    if False:
        loop_size = grid_3d_operator.shape[0]//2
    else:
        loop_size = grid_3d_operator.shape[0]

    Linf = cvxpy.Variable(nonneg=True)
    constraints = []
    for i in range(loop_size):
        for j in range(3):
            # K dot nabla K L1
            if np.all(grid_3d_operator[i, j, :, :]==0):
                continue
            constraints.append(
                cvxpy.trace(
                    grid_3d_operator[i, j, :, :] @ X
                )<=Linf
            )
            constraints.append(
                cvxpy.trace(
                    grid_3d_operator[i, j, :, :] @ X
                )>=-Linf
            )
    return(constraints, Linf)

def cvxpy_create_Linf_leq_from_array(grid_3d_operator, grid_3d_operator_scale, grid_1d_operator, grid_1d_operator_scale, k_param, X, stellsym):
    '''
    Constructing cvxpy constraints and variables necessary for the following constraint:

    -kg(x) <= ||f(x)||_\infty <= kg(x)
    
    -- Inputs:
    grid_3d_operator: Array, has shape (n_grid, 3, ndof+1, ndof+1)
    X: cvxpy Variable
    stellsym: Whether the grid the operator lives on has stellarator 
    symmetry.

    -- Outputs:
    constraints: A list of cvxpy constraints. 
    Linf: Adding a lam*Linf term in the 
    objective adds an L1 norm term.
    '''
    # if stellsym:
    if False:
        loop_size = grid_3d_operator.shape[0]//2
    else:
        loop_size = grid_3d_operator.shape[0]

    #k_param = cvxpy.Variable()
    k_param_eff = k_param * grid_1d_operator_scale / grid_3d_operator_scale 
    constraints = []
    for i in range(loop_size):
        for j in range(3):
            # K dot nabla K L1
            if np.all(grid_3d_operator[i, j, :, :]==0):
                continue
            # constraints.append(
            #     cvxpy.trace(grid_1d_operator[i, :, :] @ X)
            #     >=0
            # )
            constraints.append(
                cvxpy.trace(
                    (
                        grid_3d_operator[i, j, :, :] 
                        - k_param_eff * grid_1d_operator[i, :, :]
                    ) @ X
                )<=0
            )
            constraints.append(
                cvxpy.trace(
                    (
                        -grid_3d_operator[i, j, :, :] 
                        - k_param_eff * grid_1d_operator[i, :, :]
                    ) @ X
                )<=0
            )
    return(constraints)