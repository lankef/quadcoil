"""Quick benchmark / regression test for SurfaceXYZTensorFourierJAX.

Adapted from general_offset.ipynb.  Verifies that the vectorised
``_dof_to_gamma_op`` agrees with the reference ``xyztensor_gammadash``
across several derivative orders, then times JIT compile + first run for
``gamma()`` and ``unitnormal()`` on a moderately large XYZ tensor Fourier
surface (mpol=ntor=10, nfp=3, stellsym, ndof=661 from
``serial1149928.json``).
"""
import os
import time

import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
from simsopt import load

from quadcoil import SurfaceJAX, SurfaceXYZTensorFourierJAX
from quadcoil.surface import xyztensor_gammadash


os.chdir(os.path.dirname(os.path.abspath(__file__)))
plasma_surface_jax2 = SurfaceJAX.from_simsopt(load('serial1149928.json')[0][1])
assert isinstance(plasma_surface_jax2, SurfaceXYZTensorFourierJAX)

s = plasma_surface_jax2
print(
    f'Loaded {type(s).__name__}: mpol={s.mpol}, ntor={s.ntor}, '
    f'nfp={s.nfp}, stellsym={s.stellsym}, '
    f'nphi={s.quadpoints_phi.size}, ntheta={s.quadpoints_theta.size}, '
    f'ndofs={s.dofs.size}'
)

# Correctness: vectorised operator must match the reference implementation.
print('\nCorrectness checks:')
for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]:
    op = SurfaceXYZTensorFourierJAX._dof_to_gamma_op(
        phi_grid=s.phi_mesh, theta_grid=s.theta_mesh,
        nfp=s.nfp, stellsym=s.stellsym,
        dash1_order=a, dash2_order=b,
        mpol=s.mpol, ntor=s.ntor,
    )
    g_op = (op @ s.dofs).block_until_ready()
    g_ref = xyztensor_gammadash(
        dofs=s.dofs,
        quadpoints_phi=s.quadpoints_phi,
        quadpoints_theta=s.quadpoints_theta,
        nfp=s.nfp, stellsym=s.stellsym,
        a=a, b=b, mpol=s.mpol, ntor=s.ntor,
    ).block_until_ready()
    diff = float(jnp.max(jnp.abs(g_op - g_ref)))
    ref_max = float(jnp.max(jnp.abs(g_ref)))
    print(f'  (a={a}, b={b}): max abs diff = {diff:.2e}  (ref max = {ref_max:.3g})')
    assert diff < 1e-8, f'(a={a}, b={b}) max diff {diff} >= 1e-8'

# Timing: compile + first run vs. cached run.
print('\nTimings:')
t0 = time.perf_counter()
plasma_surface_jax2.gamma().block_until_ready()
gamma_t1 = time.perf_counter() - t0

t0 = time.perf_counter()
plasma_surface_jax2.gamma().block_until_ready()
gamma_t2 = time.perf_counter() - t0

t0 = time.perf_counter()
plasma_surface_jax2.unitnormal().block_until_ready()
un_t1 = time.perf_counter() - t0

t0 = time.perf_counter()
plasma_surface_jax2.unitnormal().block_until_ready()
un_t2 = time.perf_counter() - t0

print(f'  gamma()       compile+run = {gamma_t1:.3f}s, run-only = {gamma_t2:.5f}s')
print(f'  unitnormal()  compile+run = {un_t1:.3f}s, run-only = {un_t2:.5f}s')
