r'''
The regularized self-field of a sheet current.

``quantity/force.py`` evaluates the regularized self-force per unit area of a
sheet current using the Robin & Volpe (2022) formula. This module evaluates the
*self-field* :math:`\mathbf B_{self}` that generates that force through
:math:`\mathbf L' = \mathbf K \times \mathbf B_{self}`. See
:func:`_B_self_integrands_xyz` for the derivation.
'''
import jax.numpy as jnp
import numpy as np
from jax import jit
from .current import _K
from .quantity import _Quantity
from quadcoil import project_arr_cylindrical

def _B_self_integrands_xyz(qp, dofs, winding_surface_mode=False):
    r'''
    Calculates the single- and double-layer integrand *vectors* of the
    regularized sheet-current self-field.

    Derivation
    ----------
    Notation follows ``quantity/force.py``. :math:`\mathbf r'` is the
    evaluation point ("y"), :math:`\mathbf r''` is the source point ("x"), both
    on the winding surface,
    :math:`\mathbf R \equiv \mathbf r' - \mathbf r''`, :math:`R=|\mathbf R|`,
    :math:`\mathbf n \equiv \mathbf n(\mathbf r'')`,
    :math:`\pi \equiv \mathbf I - \mathbf n \mathbf n` and
    :math:`\nabla \equiv \nabla_{\mathbf r''}`. The surface gradient is
    :math:`\nabla_S \equiv \pi\nabla = \nabla\phi\,\partial_\phi
    + \nabla\theta\,\partial_\theta`. Note that
    :math:`\nabla(1/R) = \mathbf R/R^3`.

    **Step 1 — the Robin-Volpe force is a cross product.**
    The four terms of the force formula implemented by ``_force_cyl`` are

    .. math::

        \frac{4\pi}{\mu_0}\mathbf L'
        = \underbrace{-\oint \frac{dS''}{R}
          \left\{\nabla\cdot[\pi\mathbf K'] + \pi\mathbf K'\cdot\nabla\right\}
          \mathbf K''}_{T_1}
        + \underbrace{\oint dS''\, [\mathbf K'\cdot\mathbf n]
          \frac{\mathbf R\cdot\mathbf n}{R^3}\mathbf K''}_{T_2} \\
        + \underbrace{\oint \frac{dS''}{R}
          \left\{(\mathbf K'\cdot\mathbf K'')\nabla\cdot\pi
          + \nabla(\mathbf K'\cdot\mathbf K'')\right\}}_{T_3}
        - \underbrace{\oint dS''\,[\mathbf K'\cdot\mathbf K'']
          \frac{\mathbf R\cdot\mathbf n}{R^3}\mathbf n}_{T_4}.

    :math:`T_1` is an integration by parts in disguise. With the *tangential*
    field :math:`\mathbf a \equiv \pi\mathbf K'` (:math:`\mathbf K'` is constant
    in :math:`\mathbf r''`), the surface divergence theorem on a closed surface
    gives :math:`\oint dS''\,\nabla_S\cdot(f\mathbf a) = 0`, hence

    .. math::

        T_1 = \oint dS''\,
        \left[\mathbf a\cdot\nabla_S\frac1R\right]\mathbf K''
        = \oint dS''\,
        \left[\frac{\mathbf K'\cdot\mathbf R}{R^3}
        - \frac{(\mathbf K'\cdot\mathbf n)(\mathbf R\cdot\mathbf n)}{R^3}
        \right]\mathbf K''.

    The second piece cancels :math:`T_2` exactly, leaving
    :math:`T_1+T_2 = \oint dS'' (\mathbf K'\cdot\mathbf R)\mathbf K''/R^3`.

    :math:`T_3` is the same trick for a scalar. Using
    :math:`\nabla\cdot\pi = -(\nabla_S\cdot\mathbf n)\mathbf n`
    (the :math:`(\mathbf n\cdot\nabla)\mathbf n` piece of ``div_pi_x`` in
    ``_force_integrands_xyz`` vanishes identically because
    :math:`\nabla\phi` and :math:`\nabla\theta` are tangential) and the
    closed-surface identity
    :math:`\oint dS''[\nabla_S g - g(\nabla_S\cdot\mathbf n)\mathbf n] = 0`
    with :math:`g = (\mathbf K'\cdot\mathbf K'')/R`,

    .. math::

        T_3 = -\oint dS''\,(\mathbf K'\cdot\mathbf K'')\nabla_S\frac1R
        = -\oint dS''\,(\mathbf K'\cdot\mathbf K'')
        \frac{\mathbf R - \mathbf n(\mathbf n\cdot\mathbf R)}{R^3},

    whose normal piece cancels :math:`T_4`, leaving
    :math:`T_3+T_4 = -\oint dS''(\mathbf K'\cdot\mathbf K'')\mathbf R/R^3`.
    Adding the two halves and using
    :math:`\mathbf K'\times(\mathbf K''\times\mathbf R)
    = \mathbf K''(\mathbf K'\cdot\mathbf R)
    - \mathbf R(\mathbf K'\cdot\mathbf K'')`,

    .. math::

        \mathbf L' = \mathbf K'\times\mathbf B_{self},\quad
        \mathbf B_{self}(\mathbf r') = \frac{\mu_0}{4\pi}
        \oint dS''\,\frac{\mathbf K''\times\mathbf R}{R^3}.

    So the Robin-Volpe self-force is simply :math:`\mathbf K` crossed into the
    principal-value Biot-Savart field of the sheet, i.e. the average of the two
    one-sided limits of :math:`\mathbf B` across the sheet.

    **Step 2 — regularize.** The :math:`1/R^2` kernel above is not absolutely
    integrable, so quadrature of that form converges far too slowly to be
    useful (and would not reproduce ``_force_cyl``). We undo the cancellation
    that produced it. Split the kernel into surface and normal parts,

    .. math::

        \frac{\mathbf R}{R^3} = \nabla\frac1R
        = \nabla_S\frac1R + \mathbf n\frac{\mathbf n\cdot\mathbf R}{R^3},

    so that
    :math:`\mathbf K''\times\mathbf R/R^3
    = \mathbf K''\times\nabla_S(1/R)
    + (\mathbf K''\times\mathbf n)(\mathbf n\cdot\mathbf R)/R^3`.
    The second term already carries the double-layer kernel, which is only
    :math:`O(1/R)` on a smooth surface. For the first term, component
    :math:`i` reads
    :math:`[\mathbf K''\times\nabla_S(1/R)]_i
    = \mathbf a^{(i)}\cdot\nabla_S(1/R)` with the tangential field
    :math:`\mathbf a^{(i)} \equiv \pi(\mathbf e_i\times\mathbf K'')`, so the
    same divergence theorem as in Step 1 gives
    :math:`\oint dS''\,\mathbf a^{(i)}\cdot\nabla_S(1/R)
    = -\oint dS''\,(\nabla_S\cdot\mathbf a^{(i)})/R`, with

    .. math::

        \nabla_S\cdot\mathbf a^{(i)} =
        -\left[\nabla\phi\times\partial_\phi\mathbf K''
        + \nabla\theta\times\partial_\theta\mathbf K''\right]_i
        - (\nabla_S\cdot\mathbf n)\,[\mathbf K''\times\mathbf n]_i.

    This yields the implemented form, in which every kernel is at worst
    :math:`O(1/R)`:

    .. math::

        \mathbf B_{self}(\mathbf r') = \frac{\mu_0}{4\pi}\left[
        \oint \frac{dS''}{R}\,\mathbf S(\mathbf r'')
        + \oint dS''\,\frac{\mathbf R\cdot\mathbf n}{R^3}
        \mathbf D(\mathbf r'')\right], \\
        \mathbf S \equiv
        \nabla\phi\times\partial_\phi\mathbf K''
        + \nabla\theta\times\partial_\theta\mathbf K''
        + (\nabla_S\cdot\mathbf n)(\mathbf K''\times\mathbf n),\quad
        \mathbf D \equiv \mathbf K''\times\mathbf n.

    **Step 3 — why this reproduces ``_force_cyl`` exactly, not just in the
    continuum limit.** ``_force_integrands_xyz`` returns rank-2 tensors
    :math:`T_{ai}` whose first index contracts :math:`\mathbf K'`. Writing out
    its four groups of terms and using
    :math:`\epsilon_{iam}(\mathbf U\times\mathbf V)_m
    = U_iV_a - U_aV_i`, one finds

    .. math::

        T^{single}_{ai} = \epsilon_{iam}S_m,\qquad
        T^{double}_{ai} = \epsilon_{iam}D_m,

    term by term and *before* any quadrature: the pair
    :math:`\mathbf n(\nabla_S\cdot\mathbf n)\mathbf K''
    - \mathbf K''(\nabla_S\cdot\mathbf n)\mathbf n` is the dual of
    :math:`(\nabla_S\cdot\mathbf n)(\mathbf K''\times\mathbf n)`, and each pair
    :math:`\partial_j\mathbf K''\nabla j - \nabla j\,\partial_j\mathbf K''`
    is the dual of :math:`\nabla j\times\partial_j\mathbf K''`. Since
    :math:`K'_a\epsilon_{iam}S_m = (\mathbf K'\times\mathbf S)_i`, sharing the
    same quadrature weights makes
    :math:`\mathbf K'\times\mathbf B_{self}` agree with the Robin-Volpe force
    to machine precision on the grid.

    Parameters
    ----------
    qp : QuadcoilParams
    dofs : dict
        Must contain ``'phi'``.
    winding_surface_mode : bool or ``'divide'``, optional, default=False
        Same convention as ``_force_integrands_xyz``: ``False`` uses the
        evaluation surface, ``True`` the full winding surface, and
        ``'divide'`` one field period of the winding surface.

    Returns
    -------
    integrand_single : ndarray, shape (n_phi_x, n_theta_x, 3(xyz))
        :math:`\mu_0/(4\pi)\,\mathbf S`, to be weighted by :math:`dS''/R`.
    integrand_double : ndarray, shape (n_phi_x, n_theta_x, 3(xyz))
        :math:`\mu_0/(4\pi)\,\mathbf D`, to be weighted by
        :math:`dS''(\mathbf R\cdot\mathbf n)/R^3`.

    Notes
    -----
    The x, y, z components of these integrands are not field-period periodic.
    Their cylindrical components are, but folding the integral onto one field
    period on that basis is still only correct if the evaluation point's
    quantities are rotated into the source point's basis and the result rotated
    back, so :func:`_B_self` integrates over the whole winding surface instead.
    '''
    ''' Surface properties '''
    if winding_surface_mode=='divide':
        n_phi_1fp = len(qp.winding_surface.quadpoints_phi)//qp.winding_surface.nfp
        surface = qp.winding_surface.copy_and_set_quadpoints(
            quadpoints_phi=qp.winding_surface.quadpoints_phi[:n_phi_1fp],
            quadpoints_theta=qp.winding_surface.quadpoints_theta,
        )
    elif winding_surface_mode:
        surface = qp.winding_surface
    else:
        surface = qp.eval_surface
    unitnormal_x = surface.unitnormal()
    unitnormaldash1_x = surface.unitnormaldash(1, 0)
    unitnormaldash2_x = surface.unitnormaldash(0, 1)
    grad1_x, grad2_x = surface.grad_helper()

    ''' K-related quantities '''
    phi_mn = dofs['phi']
    (
        Kdash1_sv_op,
        Kdash2_sv_op,
        Kdash1_const,
        Kdash2_const
    ) = qp.Kdash_helper(winding_surface_mode=winding_surface_mode)
    K_x = _K(qp, dofs, winding_surface_mode=winding_surface_mode)
    Kdash1_x = Kdash1_sv_op @ phi_mn + Kdash1_const
    Kdash2_x = Kdash2_sv_op @ phi_mn + Kdash2_const

    # Divergence of the unit normal, -2H. Same expression as in
    # _force_integrands_xyz.
    # Shape: (n_phi_x, n_theta_x)
    div_n_x = (
        jnp.sum(grad1_x * unitnormaldash1_x, axis=-1)
        + jnp.sum(grad2_x * unitnormaldash2_x, axis=-1)
    )
    # K(x) cross n(x). Shape: (n_phi_x, n_theta_x, 3(xyz))
    K_cross_n_x = jnp.cross(K_x, unitnormal_x, axis=-1)

    # S: the dual of the single-layer integrand tensor of the force.
    # Shape: (n_phi_x, n_theta_x, 3(xyz))
    integrand_single = 1e-7 * (
        jnp.cross(grad1_x, Kdash1_x, axis=-1)
        + jnp.cross(grad2_x, Kdash2_x, axis=-1)
        + div_n_x[:, :, None] * K_cross_n_x
    )
    # D: the dual of the double-layer integrand tensor of the force.
    # Shape: (n_phi_x, n_theta_x, 3(xyz))
    integrand_double = 1e-7 * K_cross_n_x
    return integrand_single, integrand_double

def _singular_layer_kernels(gamma_y, gamma_x, unitnormal_x, da_x, nfp):
    r'''
    Returns the quadrature-weighted single- and double-layer kernels for a
    regularized sheet-current singular integral:

    .. math::

        \text{single\_kernel\_da}_{ij,k,lm}
            = \frac{da(\mathbf r''_{klm})}{|\mathbf r'_{ij} - \mathbf r''_{klm}|},
        \quad
        \text{double\_kernel\_da}_{ij,k,lm}
            = \frac{da(\mathbf r''_{klm})\,
              (\mathbf r'_{ij}-\mathbf r''_{klm})\cdot\mathbf n''_{klm}}
              {|\mathbf r'_{ij}-\mathbf r''_{klm}|^3}.

    Self-interaction pairs (index-diagonal with ``fp == 0``) and coincident
    off-diagonal pairs (``dist_sq == 0``, e.g. when the eval grid shares a
    point with the winding surface) are both masked to zero.  The safe-sqrt
    approach (``sqrt(where(mask, 1, dist_sq))``) keeps the JAX-computed
    derivative finite at the masked entries without any ``1e-10`` fudge.

    Parameters
    ----------
    gamma_y : ndarray, shape ``(n_phiy, n_thetay, 3)``
        Evaluation points in xyz.
    gamma_x : ndarray, shape ``(n_phix_1fp * nfp, n_thetax, 3)``
        Source points in xyz spanning ``nfp`` field periods.
    unitnormal_x : ndarray, shape ``(n_phix_1fp * nfp, n_thetax, 3)``
        Unit normal at every source point.
    da_x : ndarray, shape ``(n_phix_1fp, n_thetax)``
        Surface-area element on the **one-field-period** source grid.
        ``gamma_x.shape[0]`` must equal ``da_x.shape[0] * nfp``.
    nfp : int
        Number of field periods spanned by ``gamma_x``.

    Returns
    -------
    single_kernel_da : ndarray, shape ``(n_phiy, n_thetay, nfp, n_phix_1fp, n_thetax)``
    double_kernel_da : ndarray, shape ``(n_phiy, n_thetay, nfp, n_phix_1fp, n_thetax)``
    '''
    diff = gamma_y[:, :, None, None, :] - gamma_x[None, None, :, :, :]

    shapey = list(diff.shape[:2])
    n_phix_1fp, n_thetax = da_x.shape[0], da_x.shape[1]
    shape_integral = shapey + [nfp, n_phix_1fp, n_thetax]

    n_phiy, n_thetay = shapey
    fp_idx   = np.arange(nfp)[None, None, :, None, None]
    phi_xidx = np.arange(n_phix_1fp)[None, None, None, :, None]
    th_xidx  = np.arange(n_thetax)[None, None, None, None, :]
    phi_yidx = np.arange(n_phiy)[:, None, None, None, None]
    th_yidx  = np.arange(n_thetay)[None, :, None, None, None]
    self_mask = (
        (fp_idx == 0)
        & (phi_xidx == phi_yidx)
        & (th_xidx == th_yidx)
    )

    dist_sq = jnp.sum(diff**2, axis=-1).reshape(shape_integral)
    double_layer_denom = jnp.sum(
        diff * unitnormal_x[None, None, :, :, :], axis=-1
    ).reshape(shape_integral)

    # Also mask coincident off-diagonal pairs (dist_sq == 0) that arise when
    # eval_surface shares a grid point with the winding surface or when the
    # winding surface pinches.  The index-based self_mask alone does not catch
    # these when gamma_y is not a slice of gamma_x.
    singular_mask = self_mask | (dist_sq == 0.0)
    safe_dist = jnp.sqrt(jnp.where(singular_mask, 1.0, dist_sq))

    single_kernel_da = jnp.where(
        singular_mask,
        0.0,
        da_x[None, None, None, :, :] / safe_dist,
    )
    double_kernel_da = jnp.where(
        singular_mask,
        0.0,
        da_x[None, None, None, :, :] * double_layer_denom / (safe_dist**3),
    )
    return single_kernel_da, double_kernel_da


def _integrate_B_self(
    gamma_y,
    gamma_x,
    unitnormal_x,
    da_x,
    single_integrand,
    double_integrand,
    nfp,
):
    r'''
    Performs the singular integration of the self-field using
    :func:`_singular_layer_kernels`.

    The integrands ``single_integrand`` and ``double_integrand`` are vectors of
    shape ``(n_phix_1fp, n_thetax, 3)``; the function returns their weighted
    sums over all source points.

    Note: when called from :func:`_B_self`, ``gamma_y`` is
    ``gamma_x[:n_phi_1fp]`` (a leading slice of the same JAX array), which
    guarantees that the index-diagonal entries of ``diff`` are bitwise zero and
    the index-based self-mask in :func:`_singular_layer_kernels` is exact.
    When called from the force path, ``gamma_y`` may be an independent
    ``eval_surface``; the extra ``dist_sq == 0`` guard in
    :func:`_singular_layer_kernels` covers that case.

    ``nfp`` is the number of field periods that ``gamma_x`` (and hence the
    kernels) span.  Pass ``nfp=1`` when the integrands are in xyz (not
    field-period periodic) and the full winding surface is given as ``gamma_x``
    and ``da_x``.
    '''
    single_kernel_da, double_kernel_da = _singular_layer_kernels(
        gamma_y, gamma_x, unitnormal_x, da_x, nfp,
    )
    single_results = jnp.einsum(
        'ijklm,lmn->ijn',
        single_kernel_da,
        single_integrand,
    )
    double_results = jnp.einsum(
        'ijklm,lmn->ijn',
        double_kernel_da,
        double_integrand,
    )
    return single_results, double_results

@jit 
def _B_self(qp, dofs):
    r'''
    Calculates the regularized sheet-current self-field's x, y, z components
    on the first field period of ``qp.winding_surface``. Linear in
    ``dofs['phi']`` and matrix-free.

    The evaluation grid is ``gamma_x[:n_phi_1fp]``, a leading slice of the
    full winding-surface array ``gamma_x``.  Using a slice of the same array
    (rather than an independent surface evaluated on a 1-fp grid) guarantees
    that the diagonal entries of the difference tensor ``gamma_y - gamma_x``
    are bitwise zero, which makes the index-based self-interaction mask in
    :func:`_integrate_B_self` exact for any winding-surface quadrature choice.

    See :func:`_B_self_integrands_xyz` for the derivation.
    '''
    (
        single_integrand_xyz,
        double_integrand_xyz
    ) = _B_self_integrands_xyz(qp, dofs, winding_surface_mode=True)
    gamma_x = qp.winding_surface.gamma()          # (n_phix*nfp, n_thetax, 3)
    n_phi_1fp = gamma_x.shape[0] // qp.nfp
    single_results, double_results = _integrate_B_self(
        gamma_x[:n_phi_1fp],               # (n_phi_1fp, n_thetax, 3) — slice of gamma_x
        gamma_x,                           # (n_phix*nfp, n_thetax, 3)
        qp.winding_surface.unitnormal(),   # (n_phix*nfp, n_thetax, 3)
        qp.winding_surface.da(),           # (n_phix*nfp, n_thetax)
        single_integrand_xyz,              # (n_phix*nfp, n_thetax, 3)
        double_integrand_xyz,              # (n_phix*nfp, n_thetax, 3)
        1,
    )
    return single_results + double_results

@jit 
def _B_self_cyl(qp, dofs):
    r'''
    Calculates the regularized sheet-current self-field's R, Phi, Z components
    on the first field period of ``qp.winding_surface``, satisfying

    .. math::

        \mathbf K_{cyl} \times \mathbf B_{self, cyl} = \mathbf L'_{cyl},

    where :math:`\mathbf K_{cyl}` is evaluated on the same winding 1fp grid
    and the right-hand side is ``_force_cyl_legacy``. Linear in ``dofs['phi']``
    and matrix-free.

    The integral is taken in x, y, z over the whole winding surface and the
    resulting vector is projected at the evaluation points
    (``gamma_x[:n_phi_1fp]``). Both sides of the identity above therefore use
    the same quadrature weights, and the cross product of two vectors'
    components in an orthonormal, right-handed basis is the components of
    their cross product in that same basis. So the identity holds on the grid
    to machine precision, and not merely in the continuum limit.

    See :func:`_B_self_integrands_xyz` for the derivation.
    '''
    gamma_x = qp.winding_surface.gamma()
    n_phi_1fp = gamma_x.shape[0] // qp.nfp
    return project_arr_cylindrical(
        gamma_x[:n_phi_1fp],
        _B_self(qp, dofs),
    )
_B_self_desc_unit = lambda scales: scales["B"]

@jit 
def _B2_self(qp, dofs):
    return(jnp.sum(_B_self(qp, dofs)**2, axis=-1))
_B2_self_desc_unit = lambda scales: _B_self_desc_unit(scales)**2

@jit 
def _f_B_self(qp, dofs):
    gamma_x = qp.winding_surface.gamma()
    n_phi_1fp = gamma_x.shape[0] // qp.nfp
    da_1fp = qp.winding_surface.da()[:n_phi_1fp]
    B2_val = _B2_self(qp, dofs)
    return jnp.sum(B2_val / 2 * da_1fp) * qp.nfp
_f_B_self_desc_unit = lambda scales: _B_self_desc_unit(scales)**2 * scales["R0"] * scales["a"]

# ----- Wrappers -----
# Linear 3d vector fields. Like winding_surface_B, setting their components
# is close to trivial, but <= and >= are still supported.
B_self = _Quantity.generate_c2(
    func=_B_self,
    compatibility=['<=', '>='],
    desc_unit=_B_self_desc_unit,
)

B_self_cyl = _Quantity.generate_c2(
    func=_B_self_cyl,
    compatibility=['<=', '>='],
    desc_unit=_B_self_desc_unit,
)

B2_self = _Quantity.generate_c2(
    func=_B2_self, 
    compatibility=['<='], 
    desc_unit=_B2_self_desc_unit,
)

# This is a positive definite quadratic scalar. 
f_B_self = _Quantity.generate_c2(
    func=_f_B_self, 
    compatibility=['f', '<='], 
    desc_unit=_f_B_self_desc_unit,
)

f_max_B2_self = _Quantity.generate_linf_norm(
    func=_B2_self, 
    aux_argname='scaled_max_B2_self', 
    desc_unit=_B2_self_desc_unit,
    square=False,
    auto_stellsym=True,
)