from quadcoil import quadcoil
from quadcoil.quantity import K_theta, Phi_with_net_current, K2, K, f_B, f_max_Phi, f_max_force_cyl
from quadcoil.io import simsopt_coil_from_qp
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import time
import jax
jax.config.update('jax_enable_x64', True)
from simsopt import load
from simsopt.geo import curves_to_vtk
from simsopt.field import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux
# The example is li383.
_, plasma_surface = load('surfaces.json')
net_poloidal_current_amperes = 11884578.094260072
mpol = 12
ntor = 12
separation = 0.1
f_B_target = 1e-3
unit_Phi = 1e5
unit_l1_Phi = 1e6
unit_force = 4e6
quadcoil_kwargs_dipole = {
    'nfp': plasma_surface.nfp,
    'stellsym': plasma_surface.stellsym,
    'plasma_coil_distance': separation,
    'mpol': mpol,
    'ntor': ntor,
    'plasma_mpol': plasma_surface.mpol,
    'plasma_ntor': plasma_surface.ntor,
    'net_poloidal_current_amperes': net_poloidal_current_amperes,
    'net_toroidal_current_amperes': 0.,
    'metric_name': ('f_max_Phi', 'f_B'),
    'objective_name': 'f_max_Phi', # IPM works poorly for max_Phi 
    'objective_unit': unit_Phi, # under an f_B constraint
    'constraint_name': ('f_B',),
    'constraint_type': ('<=',),
    'constraint_unit': (f_B_target,),
    'constraint_value': np.array([f_B_target,]),
}
quadcoil_kwargs_force = {
    'nfp': plasma_surface.nfp,
    'stellsym': plasma_surface.stellsym,
    'plasma_coil_distance': separation,
    'mpol': mpol,
    'ntor': ntor,
    'plasma_mpol': plasma_surface.mpol,
    'plasma_ntor': plasma_surface.ntor,
    'net_poloidal_current_amperes': net_poloidal_current_amperes,
    'net_toroidal_current_amperes': 0.,
    'metric_name': ('f_max_force_cyl', 'f_B'),
    'objective_name': 'f_max_force_cyl', # IPM works poorly for max_Phi 
    'objective_unit': unit_force, # under an f_B constraint
    'constraint_name': ('f_B',),
    'constraint_type': ('<=',),
    'constraint_unit': (f_B_target,),
    'constraint_value': np.array([f_B_target,]),
}
# First, test with the NESCOIL problem, auto-generating WS.
print('Running nescoil, with auto-generated '\
      'winding surface.')
# print('unconstrained test')
# for i in [1, 1]:
#     time1 = time.time()
#     nescoil_out_dict, nescoil_qp, nescoil_dofs, _ = quadcoil(
#         nfp=plasma_surface.nfp,
#         stellsym=plasma_surface.stellsym,
#         mpol=4,
#         ntor=4,
#         plasma_dofs=plasma_surface.get_dofs(),
#         plasma_mpol=plasma_surface.mpol,
#         plasma_ntor=plasma_surface.ntor,
#         net_poloidal_current_amperes=net_poloidal_current_amperes,
#         net_toroidal_current_amperes=0.,
#         plasma_coil_distance=plasma_surface.minor_radius(),
#         # Set the objective to 
#         # f_B
#         objective_name='f_B',
#         objective_weight=None,
#         objective_unit=None,
#         # Set the output metrics to f_B and f_K
#         metric_name=('f_B', 'f_K')
#     )
#     time2 = time.time()
#     print('auglag time', time2-time1)
#     time1 = time.time()
#     nescoil2_out_dict, nescoil2_qp, nescoil2_dofs, _ = quadcoil(
#         nfp=plasma_surface.nfp,
#         stellsym=plasma_surface.stellsym,
#         mpol=4,
#         ntor=4,
#         plasma_dofs=plasma_surface.get_dofs(),
#         plasma_mpol=plasma_surface.mpol,
#         plasma_ntor=plasma_surface.ntor,
#         net_poloidal_current_amperes=net_poloidal_current_amperes,
#         net_toroidal_current_amperes=0.,
#         plasma_coil_distance=plasma_surface.minor_radius(),
#         # Set the objective to 
#         # f_B
#         objective_name='f_B',
#         objective_weight=None,
#         objective_unit=None,
#         # Set the output metrics to f_B and f_K
#         metric_name=('f_B', 'f_K'),
#         solver='ipm',
#     )
#     time2 = time.time()
#     print('IPM time', time2-time1)

# print('f_B from auglag', f_B(nescoil_qp, nescoil_dofs))
# print('f_B from ipm', f_B(nescoil2_qp, nescoil2_dofs))
# print('convex constrained test')
# for i in [1, 1]:
#     time1 = time.time()
#     top_out_dict, top_qp, top_dofs, _ = quadcoil(
#         nfp=plasma_surface.nfp,
#         stellsym=plasma_surface.stellsym,
#         mpol=4,
#         ntor=4,
#         plasma_dofs=plasma_surface.get_dofs(),
#         plasma_mpol=plasma_surface.mpol,
#         plasma_ntor=plasma_surface.ntor,
#         net_poloidal_current_amperes=net_poloidal_current_amperes,
#         net_toroidal_current_amperes=0.,
#         plasma_coil_distance=plasma_surface.minor_radius(),
#         # Set the objective to 
#         # f_B
#         objective_name='f_B',
#         objective_weight=None,
#         objective_unit=None,
#         # Set the constraint to K_theta
#         constraint_name=('K_theta',),
#         constraint_type=('>=',),
#         constraint_value=np.array([0.,]),
#         constraint_unit=(None,),
#         smoothing='approx',
#         convex=True,
#         # Set the output metrics to f_B and f_K
#         metric_name=('f_B', 'f_K')
#     )
#     time2 = time.time()
#     print('auglag time', time2-time1)
#     time1 = time.time()
#     top2_out_dict, top2_qp, top2_dofs, _ = quadcoil(
#         nfp=plasma_surface.nfp,
#         stellsym=plasma_surface.stellsym,
#         mpol=4,
#         ntor=4,
#         plasma_dofs=plasma_surface.get_dofs(),
#         plasma_mpol=plasma_surface.mpol,
#         plasma_ntor=plasma_surface.ntor,
#         net_poloidal_current_amperes=net_poloidal_current_amperes,
#         net_toroidal_current_amperes=0.,
#         plasma_coil_distance=plasma_surface.minor_radius(),
#         # Set the objective to 
#         # f_B
#         objective_name='f_B',
#         objective_weight=None,
#         objective_unit=None,
#         # Set the constraint to K_theta
#         constraint_name=('K_theta',),
#         constraint_type=('>=',),
#     time1 = time.time()
#     top_out_dict, top_qp, top_dofs, _ = quadcoil(
#         # nfp=qp_temp.plasma_surface.nfp,
#         # stellsym=qp_temp.plasma_surface.stellsym,
#         plasma_dofs=plasma_surface.get_dofs(),
#         # plasma_mpol=qp_temp.plasma_surface.mpol,
#         # plasma_ntor=qp_temp.plasma_surface.ntor,
#         # net_poloidal_current_amperes=qp_temp.net_poloidal_current_amperes,
#         # Bnormal_plasma=qp_temp.Bnormal_plasma
#         # plasma_coil_distance=-separation, # DESC surface normal points inward
#         **quadcoil_kwargs_dipole|{'solver': 'auglag-lbfgs', 'smoothing': 'approx'},
#     )
#     time2 = time.time()
#     print('auglag time', time2-time1)
#     time1 = time.time()
#     top2_out_dict, top2_qp, top2_dofs, _ = quadcoil(
#         # nfp=qp_temp.plasma_surface.nfp,
#         # stellsym=qp_temp.plasma_surface.stellsym,
#         plasma_dofs=plasma_surface.get_dofs(),
#         # plasma_mpol=qp_temp.plasma_surface.mpol,
#         # plasma_ntor=qp_temp.plasma_surface.ntor,
#         # net_poloidal_current_amperes=qp_temp.net_poloidal_current_amperes,
#         # Bnormal_plasma=qp_temp.Bnormal_plasma
#         # plasma_coil_distance=-separation, # DESC surface normal points inward
#         **quadcoil_kwargs_dipole|{'solver': 'ipm', 'smoothing': 'slack'},
#     )
#     time2 = time.time()
#     print('IPM time', time2-time1)
# print('f_B from auglag:', f_B(top_qp, top_dofs))
# print('f_B from ipm:   ', f_B(top2_qp, top2_dofs))
# print('f_max_Phi from auglag:', f_max_Phi(top_qp, top_dofs))
# print('f_max_Phi from ipm:   ', f_max_Phi(top2_qp, top2_dofs))

# Compare auglag, IPM, and SLSQP
for i in [1, 1]:
    time1 = time.time()
    top_out_dict, top_qp, top_dofs, _ = quadcoil(
        plasma_dofs=plasma_surface.get_dofs(),
        **quadcoil_kwargs_force|{'solver': 'auglag-lbfgs', 'smoothing': 'approx'},
    )
    time2 = time.time()
    print('auglag time', time2-time1)
    # time1 = time.time()
    # top2_out_dict, top2_qp, top2_dofs, _ = quadcoil(
    #     plasma_dofs=plasma_surface.get_dofs(),
    #     **quadcoil_kwargs_force|{'solver': 'ipm', 'smoothing': 'approx'},
    # )
    # time2 = time.time()
    # print('IPM time', time2-time1)
    time1 = time.time()
    top3_out_dict, top3_qp, top3_dofs, _ = quadcoil(
        plasma_dofs=plasma_surface.get_dofs(),
        **quadcoil_kwargs_force|{'solver': 'slsqp', 'smoothing': 'approx'},
    )
    time2 = time.time()
    print('SLSQP time', time2-time1)

print('--- Results comparison ---')
print('f_B from auglag:', f_B(top_qp, top_dofs))
# print('f_B from ipm:   ', f_B(top2_qp, top2_dofs))
print('f_B from slsqp: ', f_B(top3_qp, top3_dofs))
print('f_max_force_cyl from auglag:', f_max_force_cyl(top_qp, top_dofs))
# print('f_max_force_cyl from ipm:   ', f_max_force_cyl(top2_qp, top2_dofs))
print('f_max_force_cyl from slsqp: ', f_max_force_cyl(top3_qp, top3_dofs))
