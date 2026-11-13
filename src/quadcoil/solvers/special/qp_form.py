"""Canonical QP / conic problem form for special solvers."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import grad, hessian, jacfwd


class ConicProblem(NamedTuple):
    r'''
    Dense conic QP in the form

    .. math::

        \min_x \tfrac12 x^\top P x + q^\top x
        \quad\text{s.t.}\quad
        G x + s = h,\quad s \in K,\quad
        A x = b.

    All fields are arrays (a JAX pytree).  Coordinates are absolute
    (not displacement-from-``x0``); see :func:`canonicalize_qp`.
    '''
    P: jnp.ndarray   # (n, n)
    q: jnp.ndarray   # (n,)
    G: jnp.ndarray   # (m, n)
    h: jnp.ndarray   # (m,)
    A: jnp.ndarray   # (p, n)
    b: jnp.ndarray   # (p,)


# Shared status codes for conic solvers (ipm_conic, clarabel, …).
CONIC_STATUS = (
    'solved',                 # 0
    'solved_inaccurate',      # 1
    'primal_infeasible',      # 2
    'dual_infeasible',        # 3
    'max_iterations',         # 4
    'numerical_error',        # 5
)


def canonicalize_qp(f, g, x0, h_eq=None):
    r'''
    Materialise a quadratic objective and affine inequalities about ``x0``.

    Because ``f`` is quadratic and ``g`` is affine, the expansion

    .. math::

        f(x) = \tfrac12 (x-x_0)^\top P (x-x_0) + q_0^\top (x-x_0) + f_0

    is exact.  We rewrite it in absolute coordinates

    .. math::

        f(x) = \tfrac12 x^\top P x + q^\top x + \text{const},
        \qquad q = q_0 - P x_0,

    and likewise ``G x <= h`` with ``G = Dg(x_0)``, ``h = -g(x_0) + G x_0``.

    Parameters
    ----------
    f : Callable
        Scalar quadratic objective of the flat preconditioned ``x``.
    g : Callable
        Affine inequality residual ``g(x) <= 0``, returning shape ``(m,)``.
    x0 : ndarray
        Expansion point (typically ``x_flat_init``).
    h_eq : Callable or None
        Optional affine equalities ``h(x) = 0``.

    Returns
    -------
    ConicProblem
    '''
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    P = hessian(f)(x0)
    P = 0.5 * (P + P.T)
    # grad f(x0) = P x0 + q  ⇒  q = grad f(x0) - P x0
    q = grad(f)(x0) - P @ x0

    # g(x) = G x - g0_shift, with g(x0) known and G = jac g(x0).
    # We need G x <= h  equivalent to g(x) <= 0.
    # g(x) = G (x - x0) + g(x0) = G x + (g(x0) - G x0),
    # so h = -(g(x0) - G x0) = G x0 - g(x0).
    G = jacfwd(g)(x0)
    g0 = jnp.atleast_1d(g(x0)).ravel()
    if G.ndim == 1:
        G = G.reshape(1, -1)
    h = G @ x0 - g0

    if h_eq is None:
        A = jnp.zeros((0, x0.shape[0]), dtype=x0.dtype)
        b = jnp.zeros(0, dtype=x0.dtype)
    else:
        A = jacfwd(h_eq)(x0)
        h0 = jnp.atleast_1d(h_eq(x0)).ravel()
        if A.ndim == 1:
            A = A.reshape(1, -1)
        # h_eq(x) = A (x - x0) + h_eq(x0) = 0
        # ⇒ A x = A x0 - h_eq(x0)
        b = A @ x0 - h0

    return ConicProblem(P=P, q=q, G=G, h=h, A=A, b=b)


def orthant_cones(m):
    """Static cone spec for a pure nonnegative orthant of dimension ``m``."""
    return (('l', int(m)),)


def canonicalize_socp(f, g, x0, g_blocks, h_eq=None):
    r'''
    Materialise a convex QCQP as a dense SOCP about ``x0``.

    Problem
    -------
    Quadratic / linear objective with a mix of affine inequalities and
    scalar convex quadratic inequalities ``g_i(x) ≤ 0``.  Affine rows stay
    in the nonnegative orthant; each quadratic row is rewritten as one
    second-order cone of size ``n+2``.

    Quadratic → SOC (rotated form)
    ------------------------------
    Expand a scalar quadratic about absolute coordinates

    .. math::

        g_i(x) = \tfrac12 x^\top H x + a^\top x + c \le 0,

    with ``H ≽ 0``.  Factor ``H = R^\top R`` (via ``eigh``, clipping
    negative eigenvalues to zero; ``R`` is kept ``n×n`` so shapes are
    static).  Then

    .. math::

        \|R x\|^2 \le w, \qquad w = -2(a^\top x + c),

    which is equivalent to the Lorentz membership

    .. math::

        \Big(\tfrac{w+1}{2},\; R x,\; \tfrac{w-1}{2}\Big) \in \mathcal{Q}^{n+2}.

    Writing that as ``G_q x + s = h_q`` with ``s ∈ Q`` gives the block

    .. math::

        G_q = \begin{bmatrix} a^\top \\ -R \\ a^\top \end{bmatrix},
        \qquad
        h_q = \begin{bmatrix} \tfrac12 - c \\ 0 \\ -c - \tfrac12 \end{bmatrix}.

    Row order of ``G``
    ------------------
    Orthant rows first (all linear blocks, concatenated in ``g_blocks``
    order), then one SOC block per quadratic constraint.  Cone spec::

        (('l', m_l), ('q', n+2, k))   # omit ('l', …) when m_l = 0

    Dual map (``dual_blocks``)
    -------------------------
    Parallel to the concatenated rows of ``g``.  Each entry is either
    ``('l', start, n)`` — orthant duals copy one-to-one — or
    ``('q', start, dim)`` — the original quadratic multiplier is
    ``λ = z[start] + z[start+dim-1]`` (see the gradient identity
    ``G_q^\top z = λ ∇g_i``).

    Parameters
    ----------
    f, g : Callable
        Scalar objective and concatenated inequality residual ``g(x) ≤ 0``.
    x0 : ndarray
        Expansion point.
    g_blocks : tuple of (convexity_type, n_rows)
        Static layout of ``g``, one entry per constraint quantity.
        Quadratic blocks must have ``n_rows == 1``.
    h_eq : Callable or None
        Optional affine equalities.

    Returns
    -------
    prob : ConicProblem
    cones : tuple
    dual_blocks : tuple
    '''
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    n = x0.shape[0]

    # ── Objective: f(x) = ½ xᵀ P x + qᵀ x + const ───────────────────────
    P = hessian(f)(x0)
    P = 0.5 * (P + P.T)
    q = grad(f)(x0) - P @ x0

    # ── Walk g_blocks: build orthant rows and SOC rows separately ───────
    # Conic G is orthant-first then SOC, regardless of the original
    # interleaving of linear / quadratic constraints in g.
    g0_full = jnp.atleast_1d(g(x0)).ravel()
    G_full = jacfwd(g)(x0)
    if G_full.ndim == 1:
        G_full = G_full.reshape(1, -1)

    m_l = 0
    n_quad = 0
    for ctype, n_rows in g_blocks:
        if ctype == 'linear':
            m_l += int(n_rows)
        elif ctype == 'quadratic':
            if int(n_rows) != 1:
                raise ValueError(
                    'canonicalize_socp only supports scalar quadratic '
                    f'constraints; got n_rows={n_rows}'
                )
            n_quad += 1
        else:
            raise ValueError(f'Unsupported g_blocks type {ctype!r}')

    G_l_rows = []
    h_l_rows = []
    G_q_rows = []
    h_q_rows = []
    dual_blocks = []

    # Orthant rows occupy conic indices [0, m_l); each SOC occupies
    # n+2 consecutive rows starting at m_l + j*(n+2).
    orthant_cursor = 0
    soc_cursor = m_l
    g_offset = 0
    soc_dim = n + 2

    for ctype, n_rows in g_blocks:
        n_rows = int(n_rows)
        lo = g_offset
        hi = g_offset + n_rows
        g_offset = hi

        if ctype == 'linear':
            # Affine: g(x) = G x - h  ≤ 0  with G = Dg, h = G x0 - g(x0).
            G_block = G_full[lo:hi]
            g0_block = g0_full[lo:hi]
            h_block = G_block @ x0 - g0_block
            G_l_rows.append(G_block)
            h_l_rows.append(h_block)
            dual_blocks.append(('l', orthant_cursor, n_rows))
            orthant_cursor += n_rows
            continue

        # ── Quadratic scalar → one rotated SOC of size n+2 ──────────────
        # g_i(x) = ½ xᵀ H x + aᵀ x + c
        # hess(g_i) = H,  grad(g_i) = H x + a.
        def g_i(x, lo=lo):
            return g(x)[lo]

        H = hessian(g_i)(x0)
        H = 0.5 * (H + H.T)
        grad_i = grad(g_i)(x0)
        a = grad_i - H @ x0
        g0_i = g0_full[lo]
        c = g0_i - 0.5 * jnp.dot(x0, H @ x0) - jnp.dot(a, x0)

        # H = Rᵀ R via eigh; clip tiny negative eigenvalues from roundoff.
        evals, evecs = jnp.linalg.eigh(H)
        evals = jnp.maximum(evals, 0.0)
        # R = diag(√λ) @ evecsᵀ  ⇒  Rᵀ R = evecs @ diag(λ) @ evecsᵀ = H.
        R = jnp.sqrt(evals)[:, None] * evecs.T

        # G_q = [aᵀ; -R; aᵀ],  h_q = [½−c; 0; −c−½]
        G_q = jnp.concatenate(
            [a[None, :], -R, a[None, :]],
            axis=0,
        )
        h_q = jnp.concatenate(
            [
                jnp.array([0.5 - c]),
                jnp.zeros(n, dtype=x0.dtype),
                jnp.array([-c - 0.5]),
            ],
        )
        G_q_rows.append(G_q)
        h_q_rows.append(h_q)
        dual_blocks.append(('q', soc_cursor, soc_dim))
        soc_cursor += soc_dim

    # ── Stack into a single ConicProblem ────────────────────────────────
    if m_l > 0:
        G_l = jnp.concatenate(G_l_rows, axis=0)
        h_l = jnp.concatenate(h_l_rows, axis=0)
    else:
        G_l = jnp.zeros((0, n), dtype=x0.dtype)
        h_l = jnp.zeros(0, dtype=x0.dtype)

    if n_quad > 0:
        G_q_all = jnp.concatenate(G_q_rows, axis=0)
        h_q_all = jnp.concatenate(h_q_rows, axis=0)
    else:
        G_q_all = jnp.zeros((0, n), dtype=x0.dtype)
        h_q_all = jnp.zeros(0, dtype=x0.dtype)

    G = jnp.concatenate([G_l, G_q_all], axis=0)
    h = jnp.concatenate([h_l, h_q_all], axis=0)

    if h_eq is None:
        A = jnp.zeros((0, n), dtype=x0.dtype)
        b = jnp.zeros(0, dtype=x0.dtype)
    else:
        A = jacfwd(h_eq)(x0)
        h0 = jnp.atleast_1d(h_eq(x0)).ravel()
        if A.ndim == 1:
            A = A.reshape(1, -1)
        b = A @ x0 - h0

    # Cone spec: orthant (if any), then a batch of equal-sized SOCs.
    cones = []
    if m_l > 0:
        cones.append(('l', m_l))
    if n_quad > 0:
        cones.append(('q', soc_dim, n_quad))
    cones = tuple(cones)

    prob = ConicProblem(P=P, q=q, G=G, h=h, A=A, b=b)
    return prob, cones, tuple(dual_blocks)
