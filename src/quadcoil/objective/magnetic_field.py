import jax.numpy as jnp
import jax.numpy as jnp
from jax import jit, vmap
from quadcoil.objective import K

@jit
def winding_surface_B(qp, cp_mn):
    # A port of simsoptpp/winding_surface.cpp
    # Array WindingSurfaceB(Array& points, Array& ws_points, Array& ws_normal, Array& K)
    # NOTE: Bnormal_plasma is not necessarily stellarator 
    # symmetric even for stellarator symmetric equilibria.
    gamma = qp.plasma_surface.gamma() # The shapes will be used later
    points = gamma.reshape((-1, 3))
    num_points = points.shape[0]
    ws_gamma = qp.winding_surface.gamma()
    ws_points = ws_gamma.reshape((-1, 3))
    ws_normal = qp.winding_surface.normal().reshape((-1, 3))
    num_ws_points = ws_points.shape[0]
    nphi = ws_gamma.shape[0]
    ntheta = ws_gamma.shape[1]
    fak = 1e-7  # mu0 divided by 4 * pi factor
    K_val = K(qp, cp_mn, winding_surface_mode=True).reshape((-1, 3))
    def compute_B(point):
        point = point.reshape(1, -1)
        r = point - ws_points
        rmag_2 = jnp.sum(r**2, axis=1)
        rmag_inv = 1.0 / jnp.sqrt(rmag_2)
        rmag_inv_3 = rmag_inv**3
        nmag = jnp.sqrt(jnp.sum(ws_normal**2, axis=1))
        Kcrossr = jnp.cross(K_val, r)
        B_i = jnp.sum(nmag[:, None] * Kcrossr * rmag_inv_3[:, None], axis=0)
        return B_i
    B = vmap(compute_B)(points) * fak
    return B.reshape(gamma.shape) / nphi / ntheta
winding_surface_B_desc_unit = lambda scales: scales["B"]

@jit 
def Bnormal(qp, cp_mn):
    # Calculates the field at the plasma due to both 
    # the winding surface and the plasma
    unitnormal_plasma = qp.plasma_surface.unitnormal()
    Bnormal_winding_surface = jnp.sum(unitnormal_plasma * winding_surface_B(qp, cp_mn) , axis=-1)
    return Bnormal_winding_surface + qp.Bnormal_plasma
Bnormal_desc_unit = lambda scales: scales["B"]

@jit 
def f_B(qp, cp_mn):
    # The nescoil objective.
    Bnormal_val = Bnormal(qp, cp_mn)
    return qp.plasma_surface.integrate(Bnormal_val**2/2) * qp.nfp
f_B_desc_unit = lambda scales: scales["B"]**2 * scales["R0"] * scales["a"]

@jit
def f_B_normalized_by_Bnormal_IG(qp, cp_mn):
    # f_B normalized by f_B produced by B_normal_IG.
    # This normalization method does not change the functional
    # form of f_B's dependence on the current potential cp_mn,
    # and the optimum to this objective is the same as f_B.
    f_B_with_unit = f_B(qp, cp_mn)
    f_B_IG = f_B(qp, jnp.zeros_like(cp_mn))
    return(f_B_with_unit / f_B_IG)
f_B_normalized_by_Bnormal_IG_desc_unit = lambda scales: 1.

@jit 
def f_max_Bnormal_abs(qp, cp_mn):
    return jnp.max(jnp.abs(Bnormal(qp, cp_mn)))
f_max_Bnormal_abs_desc_unit = lambda scales: scales["B"]

@jit 
def f_max_Bnormal2(qp, cp_mn):
    return jnp.max(Bnormal(qp, cp_mn)**2)
f_max_Bnormal2_desc_unit = lambda scales: scales["B"]**2
    