from quadcoil import norm_helper, project_arr_cylindrical
from jax import jit

@jit
def K_dot_grad_K(qp, cp_mn):
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
    Kdash1 = Kdash1_sv_op@cp_mn + Kdash1_const
    Kdash2 = Kdash2_sv_op@cp_mn + Kdash2_const
    term_a = ((trig_diff_m_i_n_i@partial_phi) @ cp_mn + G)[:, :, None] * Kdash2
    term_b = ((trig_diff_m_i_n_i@partial_theta) @ cp_mn + I)[:, :, None] * Kdash1
    K_dot_grad_K = (term_a-term_b) * inv_normN_prime_2d[:, :, None]
    return(K_dot_grad_K)

@jit
def K_dot_grad_K_cyl(qp, cp_mn):
    KK_ans = K_dot_grad_K(qp, cp_mn)
    return project_arr_cylindrical(qp.eval_surface.gamma(), KK_ans)

@jit
def f_max_K_dot_grad_K_cyl(qp, cp_mn):
    return jnp.max(jnp.abs(K_dot_grad_K_cyl(qp, cp_mn)))