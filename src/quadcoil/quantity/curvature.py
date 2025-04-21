from quadcoil import norm_helper, project_arr_cylindrical
from jax import jit
# For calculating normalization constant
from .current import _K, _K_desc_unit
from .quantity import _Quantity

# ----- Implementations -----
@jit
def _K_dot_grad_K(qp, dofs):
    phi_mn = dofs['phi']
    normal = qp.eval_surface.normal()
    gammadash1 = qp.eval_surface.gammadash1()
    gammadash2 = qp.eval_surface.gammadash2()
    gammadash1dash1 = qp.eval_surface.gammadash1dash1()
    gammadash1dash2 = qp.eval_surface.gammadash1dash2()
    gammadash2dash2 = qp.eval_surface.gammadash2dash2()
    net_poloidal_current_amperes = qp.net_poloidal_current_amperes
    net_toroidal_current_amperes = qp.net_toroidal_current_amperes
    quadpoints_phi = qp.quadpoints_phi
    quadpoints_theta = qp.quadpoints_theta
    nfp = qp.nfp
    stellsym = qp.stellsym
    cp_m, cp_n = qp.make_mn()
    # Partial derivatives of K
    (
        Kdash1_sv_op, 
        Kdash2_sv_op, 
        Kdash1_const,
        Kdash2_const
    ) = qp.Kdash_helper()
    # Differentiating the current potential
    (
        _, # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = qp.diff_helper()
    normN_prime_2d, inv_normN_prime_2d = norm_helper(normal)
    G = net_poloidal_current_amperes
    I = net_toroidal_current_amperes
    ''' Pointwise product with partial r/partial phi or theta'''
    # Shape: (n phi, n theta, 3)
    Kdash1 = Kdash1_sv_op @ phi_mn + Kdash1_const
    Kdash2 = Kdash2_sv_op @ phi_mn + Kdash2_const
    term_a = ((trig_diff_m_i_n_i @ partial_phi) @ phi_mn + G)[:, :, None] * Kdash2
    term_b = ((trig_diff_m_i_n_i @ partial_theta) @ phi_mn + I)[:, :, None] * Kdash1
    K_dot_grad_K = (term_a - term_b) * inv_normN_prime_2d[:, :, None]
    return(K_dot_grad_K)
_K_dot_grad_K_desc_unit = lambda scales: _K_desc_unit(scales)**2 / scales["a"]

@jit
def _K_dot_grad_K_cyl(qp, dofs):
    KK_ans = _K_dot_grad_K(qp, dofs)
    return project_arr_cylindrical(qp.eval_surface.gamma(), KK_ans)

# ----- Wrappers -----
# This is the xyz component of the 
# K dot grad K. It's a non-convex, quadratic function of the 
# current potential Phi. Compatible with all types of copnstraints.
# '==' constraint is somewhat trivial, but we need one "==" constraint
# somewhere for testing so we pick this uncommonly used vector field for it.
K_dot_grad_K = _Quantity(
    val_func=_K_dot_grad_K, 
    eff_val_func=_K_dot_grad_K, 
    aux_g_ineq_func=None,
    aux_g_ineq_unit_conv=None,
    aux_h_eq_func=None,
    aux_h_eq_unit_conv=None,
    aux_dofs_init=None, 
    compatibility=['<=', '>=', '=='], 
    desc_unit=_K_dot_grad_K_desc_unit,
)

# This is the r, phi, z component of the 
# K dot grad K. It's a non-convex, quadratic function of the 
# current potential Phi. Compatible with only inequality constraints.
K_dot_grad_K_cyl = _Quantity(
    val_func=_K_dot_grad_K_cyl, 
    eff_val_func=_K_dot_grad_K_cyl, 
    aux_g_ineq_func=None,
    aux_g_ineq_unit_conv=None,
    aux_h_eq_func=None,
    aux_h_eq_unit_conv=None,
    aux_dofs_init=None, 
    compatibility=['<=', '>='], 
    desc_unit=_K_dot_grad_K_desc_unit,
)

# This is an l-inf norm. We have implemented a template
# in _Quantity. It's non-convex but Shor-relaxable into SDP.
f_max_K_dot_grad_K_cyl = _Quantity.generate_linf_norm(
    func=_K_dot_grad_K_cyl, 
    aux_argname='max_KK_cyl', 
    desc_unit=_K_dot_grad_K_desc_unit
)