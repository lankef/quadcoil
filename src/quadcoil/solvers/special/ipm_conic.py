"""
Speed-oriented Mehrotra interior-point method for dense QP and SOCP.

Problem
-------
    min   ½ xᵀ P x + qᵀ x
    s.t.  G x + s = h,   s ∈ K,
          A x     = b.

Cone spec (static, hashable)
----------------------------
A tuple of blocks, e.g. ``(('l', m),)`` or
``(('l', m_l), ('q', dim, count), ...)``.
Rows of ``G`` are ordered orthant-first, then SOC groups row-major
(each cone contiguous of length ``dim``).

Status codes (shared with Clarabel backend)
-------------------------------------------
0 solved, 1 solved inaccurately, 2 primal infeasible, 3 dual infeasible,
4 max iterations, 5 numerical error.

The QP and SOCP branches share only Ruiz equilibration and the
termination / Farkas-certificate tests.  Each branch is one long
function that reads like the derivation.
"""
import jax
import jax.numpy as jnp
from jax.lax import while_loop

from .qp_form import ConicProblem, CONIC_STATUS


DEFAULT_CONIC_OPTIONS = {
    'tol': 1e-8,
    'tol_inaccurate': 1e-4,
    'maxiter': 50,
    'tau': 0.99,
    'delta_init': 1e-8,
    'delta_max': 1e-2,
    'ruiz_iters': 15,
    'ruiz_clip': (1e-4, 1e4),
}


# ── Shared helpers ───────────────────────────────────────────────────────

def _cone_totals(cones):
    """Barrier degree ν and total cone dimension m from a static cone spec."""
    m = 0
    nu = 0
    has_soc = False
    for block in cones:
        if block[0] == 'l':
            m_l = int(block[1])
            m += m_l
            nu += m_l
        elif block[0] == 'q':
            dim, count = int(block[1]), int(block[2])
            m += dim * count
            nu += count
            has_soc = True
        else:
            raise ValueError(f'Unknown cone block {block!r}')
    return m, nu, has_soc


def _ruiz_equilibrate(P, q, G, h, A, b, cones, n_iters, clip):
    r"""
    Ruiz equilibration of the KKT saddle matrix.

    Iteratively rescale so that the ∞-norm of each row/column of

        [[P,  Gᵀ, Aᵀ],
         [G,  0,  0 ],
         [A,  0,  0 ]]

    is near one.  SOC blocks share a single scale per cone (cone-
    preserving).  Also applies a cost scale ``c`` so that
    ``‖P‖_∞ ≈ ‖q‖_∞ ≈ 1``.

    Returns scaled ``(P,q,G,h,A,b)`` and the diagonal scalings
    ``(d_x, d_z, d_y, c)`` for unscaling:
    ``x = d_x * x̃``, ``z = d_z * z̃ / c``, ``s = s̃ / d_z``.
    """
    n = P.shape[0]
    m = G.shape[0]
    p = A.shape[0]
    lo, hi = clip

    d_x = jnp.ones(n)
    d_z = jnp.ones(m)
    d_y = jnp.ones(p)
    c = jnp.asarray(1.0, dtype=P.dtype)

    # Precompute SOC cone membership: for each row of G, the cone id
    # (orthant rows are their own "cone"). Used to average scales.
    # Built as a static Python list → jnp arrays of group starts.
    groups = []  # list of (start, length) for SOC averaging
    offset = 0
    for block in cones:
        if block[0] == 'l':
            offset += int(block[1])
        else:
            dim, count = int(block[1]), int(block[2])
            for k in range(count):
                groups.append((offset + k * dim, dim))
            offset += dim * count

    def _equalize_soc(scale):
        # Cone-preserving: every row of an SOC shares one scale.  Use the
        # max within the cone so a structural zero row (common in the
        # rotated-QCQP embedding when a = 0) cannot drag the geometric
        # mean to the clip floor and blow up h.
        out = scale
        for start, dim in groups:
            sl = out[start:start + dim]
            out = out.at[start:start + dim].set(jnp.max(sl))
        return out

    def body(i, carry):
        P, q, G, h, A, b, d_x, d_z, d_y, c = carry
        # Column norms of [P; G; A] → scale x.
        col = (
            jnp.linalg.norm(P, axis=0, ord=jnp.inf)
            + jnp.linalg.norm(G, axis=0, ord=jnp.inf)
            + (jnp.linalg.norm(A, axis=0, ord=jnp.inf) if p > 0 else 0.0)
        )
        col = jnp.clip(jnp.sqrt(jnp.maximum(col, 1e-300)), lo, hi)
        d_x = d_x / col
        P = P / col[None, :] / col[:, None]
        q = q / col
        G = G / col[None, :]
        if p > 0:
            A = A / col[None, :]

        # Row norms of [G h] → scale z (SOC-equalized by max).
        # Including |h| keeps pure-offset cone rows (G_i = 0, h_i ≠ 0)
        # from being treated as empty and then over-scaled.
        row_g = jnp.maximum(
            jnp.linalg.norm(G, axis=1, ord=jnp.inf),
            jnp.abs(h),
        )
        row_g = jnp.clip(jnp.sqrt(jnp.maximum(row_g, 1e-300)), lo, hi)
        row_g = _equalize_soc(row_g)
        d_z = d_z / row_g
        G = G / row_g[:, None]
        h = h / row_g

        # Row norms of [A] → scale y.
        if p > 0:
            row_a = jnp.linalg.norm(A, axis=1, ord=jnp.inf)
            row_a = jnp.clip(jnp.sqrt(jnp.maximum(row_a, 1e-300)), lo, hi)
            d_y = d_y / row_a
            A = A / row_a[:, None]
            b = b / row_a

        # Cost scaling.
        pnorm = jnp.linalg.norm(P, ord=jnp.inf)
        qnorm = jnp.linalg.norm(q, ord=jnp.inf)
        gamma = jnp.clip(
            jnp.sqrt(jnp.maximum(jnp.maximum(pnorm, qnorm), 1e-300)),
            lo, hi,
        )
        c = c / gamma
        P = P / gamma
        q = q / gamma

        return (P, q, G, h, A, b, d_x, d_z, d_y, c)

    carry = (P, q, G, h, A, b, d_x, d_z, d_y, c)
    carry = jax.lax.fori_loop(0, n_iters, body, carry)
    return carry


def _termination(P, q, G, h, A, b, x, s, z, y, tol, tol_inacc):
    r"""
    Relative ∞-norm residuals and Farkas certificates.

    ``pres = ‖G x + s − h‖_∞ / (1 + max(‖h‖, ‖Gx‖, ‖s‖))``
    ``dres = ‖P x + q + Gᵀ z + Aᵀ y‖_∞ / (1 + max(‖q‖, ‖Px‖, ‖Gᵀz‖))``
    ``gap  = |xᵀ P x + qᵀ x + hᵀ z + bᵀ y| / (1 + |pobj| + |dobj|)``

    Primal infeasible certificate: ``hᵀ z + bᵀ y < 0`` and
    ``‖Gᵀ z + Aᵀ y‖`` small on the normalised dual ray.
    Dual infeasible: ``qᵀ x < 0`` and ``‖P x‖``, ``max(G x, 0)`` small.
    """
    p = A.shape[0]
    Gx = G @ x
    Px = P @ x
    GTz = G.T @ z
    ATy = A.T @ y if p > 0 else 0.0
    Ax = A @ x if p > 0 else jnp.zeros(0)

    r_prim = Gx + s - h
    r_dual = Px + q + GTz + ATy
    r_eq = Ax - b if p > 0 else jnp.zeros(0)

    pres = jnp.linalg.norm(r_prim, ord=jnp.inf) / (
        1.0 + jnp.maximum(
            jnp.linalg.norm(h, ord=jnp.inf),
            jnp.maximum(jnp.linalg.norm(Gx, ord=jnp.inf),
                        jnp.linalg.norm(s, ord=jnp.inf)),
        )
    )
    if p > 0:
        peq = jnp.linalg.norm(r_eq, ord=jnp.inf) / (
            1.0 + jnp.maximum(jnp.linalg.norm(b, ord=jnp.inf),
                              jnp.linalg.norm(Ax, ord=jnp.inf))
        )
        pres = jnp.maximum(pres, peq)

    dres = jnp.linalg.norm(r_dual, ord=jnp.inf) / (
        1.0 + jnp.maximum(
            jnp.linalg.norm(q, ord=jnp.inf),
            jnp.maximum(jnp.linalg.norm(Px, ord=jnp.inf),
                        jnp.linalg.norm(GTz, ord=jnp.inf)),
        )
    )

    pobj = 0.5 * jnp.dot(x, Px) + jnp.dot(q, x)
    dobj = -0.5 * jnp.dot(x, Px) - jnp.dot(h, z) - (jnp.dot(b, y) if p > 0 else 0.0)
    gap = jnp.abs(pobj - (-0.5 * jnp.dot(x, Px) - jnp.dot(h, z)
                          - (jnp.dot(b, y) if p > 0 else 0.0))) / (
        1.0 + jnp.abs(pobj) + jnp.abs(dobj)
    )
    # Simpler duality gap: |xᵀz| style for QP — use |pobj + hᵀz + bᵀy + ½ xᵀPx|
    gap = jnp.abs(jnp.dot(q, x) + jnp.dot(h, z) + (jnp.dot(b, y) if p > 0 else 0.0)
                  + jnp.dot(x, Px)) / (1.0 + jnp.abs(pobj) + jnp.abs(dobj))

    solved = (pres < tol) & (dres < tol) & (gap < tol)
    inaccurate = (pres < tol_inacc) & (dres < tol_inacc) & (gap < tol_inacc)

    # Farkas certificates on normalised rays.
    z_norm = jnp.linalg.norm(z) + (jnp.linalg.norm(y) if p > 0 else 0.0)
    z_hat = z / jnp.maximum(z_norm, 1e-300)
    y_hat = y / jnp.maximum(z_norm, 1e-300) if p > 0 else y
    cert_p = (
        (jnp.dot(h, z_hat) + (jnp.dot(b, y_hat) if p > 0 else 0.0) < -tol)
        & (jnp.linalg.norm(G.T @ z_hat + (A.T @ y_hat if p > 0 else 0.0)) < tol_inacc)
    )

    x_norm = jnp.linalg.norm(x)
    x_hat = x / jnp.maximum(x_norm, 1e-300)
    cert_d = (
        (jnp.dot(q, x_hat) < -tol)
        & (jnp.linalg.norm(P @ x_hat) < tol_inacc)
        & (jnp.linalg.norm(jnp.maximum(G @ x_hat, 0.0)) < tol_inacc)
    )

    return pres, dres, gap, pobj, solved, inaccurate, cert_p, cert_d


def _fraction_to_boundary_nn(v, dv, tau):
    """Orthant fraction-to-boundary: largest α with v + α dv > 0."""
    ratios = jnp.where(dv < 0.0, -v / dv, jnp.inf)
    return jnp.minimum(1.0, tau * jnp.min(ratios))


def _chol_solve_reg(M, rhs, delta, delta_max):
    """Cholesky of M+δI with a short geometric δ-retry; returns (sol, δ, ok)."""
    n = M.shape[0]
    eye = jnp.eye(n, dtype=M.dtype)
    scales = jnp.array([1.0, 10.0, 100.0, 1000.0], dtype=M.dtype)
    deltas = jnp.minimum(jnp.maximum(delta * scales, 1e-12), delta_max)

    def _one(d):
        c = jnp.linalg.cholesky(M + d * eye)
        return c

    Cs = jax.vmap(_one)(deltas)
    oks = jnp.isfinite(Cs).all(axis=(1, 2))
    idx = jnp.argmax(oks)
    ok = oks[idx]
    c = Cs[idx]
    d_used = deltas[idx]
    sol = jax.scipy.linalg.cho_solve((c, True), rhs)
    sol = jnp.where(ok, sol, jnp.zeros_like(rhs))
    return sol, c, d_used, ok


# ── Public entry point ───────────────────────────────────────────────────

def solve_conic_ipm(prob, cones, opts=None, verbose=0):
    r'''
    Solve a dense QP or SOCP by Mehrotra PDIP.

    Dispatches at trace time: orthant-only → :func:`_solve_qp`,
    otherwise → :func:`_solve_socp`.
    '''
    options = {**DEFAULT_CONIC_OPTIONS}
    if opts is not None:
        options.update(opts)
    m, nu, has_soc = _cone_totals(cones)
    if prob.G.shape[0] != m:
        raise ValueError(
            f'Cone spec total dimension {m} != G.shape[0]={prob.G.shape[0]}'
        )
    if has_soc:
        return _solve_socp(prob, cones, options, verbose, nu)
    return _solve_qp(prob, cones, options, verbose, nu)


# ── QP branch ────────────────────────────────────────────────────────────

def _solve_qp(prob, cones, opts, verbose, nu):
    r"""
    Mehrotra PDIP specialised to the nonnegative orthant (dense QP).

    Newton system (no NT scaling): with ``D = z/s``,

        M = P + Gᵀ diag(D) G + δ I,
        M Δx = rhs,

    assembled as ``(G * D[:,None]).T @ G``.  One Cholesky serves both
    the predictor and the corrector.  Equality constraints (``p > 0``)
    add a ``p×p`` Schur complement on ``A``.
    """
    P, q, G, h, A, b = prob.P, prob.q, prob.G, prob.h, prob.A, prob.b
    n = P.shape[0]
    m = G.shape[0]
    p = A.shape[0]
    tol = opts['tol']
    tol_inacc = opts['tol_inaccurate']
    maxiter = int(opts['maxiter'])
    tau = opts['tau']
    delta_init = opts['delta_init']
    delta_max = opts['delta_max']

    # ── Ruiz equilibration ───────────────────────────────────────────────
    P, q, G, h, A, b, d_x, d_z, d_y, c_scale = _ruiz_equilibrate(
        P, q, G, h, A, b, cones,
        opts['ruiz_iters'], opts['ruiz_clip'],
    )

    # ── CVXOPT-style initialisation ──────────────────────────────────────
    # Solve (P + Gᵀ G [+ Aᵀ A]) x = -q + Gᵀ h [+ Aᵀ b] with D = I.
    M0 = P + G.T @ G
    if p > 0:
        M0 = M0 + A.T @ A
    rhs0 = -q + G.T @ h
    if p > 0:
        rhs0 = rhs0 + A.T @ b
    x, _, _, _ = _chol_solve_reg(M0, rhs0, delta_init, delta_max)
    s = h - G @ x
    z = -s
    # Shift into the interior: s ← s + (1 - min(s))₊, same for z.
    s = s + jnp.maximum(1.0 - jnp.min(s), 0.0)
    z = z + jnp.maximum(1.0 - jnp.min(z), 0.0)
    y = jnp.zeros(p, dtype=P.dtype)

    if verbose > 0:
        jax.debug.print(
            "QP-IPM: n={n}, m={m}, p={p}, maxiter={mi}",
            n=n, m=m, p=p, mi=maxiter,
        )

    init_state = {
        'x': x, 's': s, 'z': z, 'y': y,
        'delta': jnp.float64(delta_init),
        'n_iter': jnp.int32(0),
        'pres': jnp.float64(jnp.inf),
        'dres': jnp.float64(jnp.inf),
        'gap': jnp.float64(jnp.inf),
        'pobj': jnp.float64(jnp.inf),
        'status_code': jnp.int32(4),
        'done': jnp.bool_(False),
    }

    def cont(state):
        return (state['n_iter'] < maxiter) & (~state['done'])

    def step(state):
        x, s, z, y = state['x'], state['s'], state['z'], state['y']
        delta = state['delta']

        # ── Residuals ────────────────────────────────────────────────────
        # r_dual = P x + q + Gᵀ z + Aᵀ y
        # r_prim = G x + s - h
        # r_eq   = A x - b
        # r_comp = s ∘ z          (affine target 0)
        r_dual = P @ x + q + G.T @ z
        if p > 0:
            r_dual = r_dual + A.T @ y
        r_prim = G @ x + s - h
        r_eq = (A @ x - b) if p > 0 else jnp.zeros(0)
        mu = jnp.dot(s, z) / m
        D = z / s                                         # NT-free scaling

        # ── Condensed Newton matrix ──────────────────────────────────────
        # M = P + Gᵀ diag(D) G + δ I
        M = P + (G * D[:, None]).T @ G

        # ── Predictor ────────────────────────────────────────────────────
        # Eliminate Δs, Δz:
        #   Δs = -r_prim - G Δx
        #   Δz = -D Δs - z          (affine: s∘z + s∘Δz + z∘Δs = 0)
        # ⇒ M Δx + Gᵀ D (-r_prim) wait:
        # r_dual + P Δx + Gᵀ Δz + Aᵀ Δy = 0
        # Δz = -D Δs - z = D (r_prim + G Δx) - z
        # ⇒ (P + Gᵀ D G) Δx + Aᵀ Δy = -r_dual - Gᵀ (D r_prim - z)
        #                            = -r_dual - Gᵀ D r_prim + Gᵀ z
        # but r_dual already contains Gᵀ z, so
        # RHS_x = -q - P x - Aᵀ y - Gᵀ D r_prim
        #       = -(P x + q + Aᵀ y) - Gᵀ D r_prim
        rhs_x = -(P @ x + q) - G.T @ (D * r_prim)
        if p > 0:
            rhs_x = rhs_x - A.T @ y

        if p == 0:
            dx_aff, c_fac, d_used, ok = _chol_solve_reg(M, rhs_x, delta, delta_max)
            dy_aff = jnp.zeros(0, dtype=P.dtype)

            def solve_corr(rhs):
                return jax.scipy.linalg.cho_solve((c_fac, True), rhs), jnp.zeros(0, dtype=P.dtype)
        else:
            # Schur: [M, Aᵀ; A, 0] via M-factor + p×p Schur.
            dx_tmp, c_fac, d_used, ok = _chol_solve_reg(M, rhs_x, delta, delta_max)
            # S = A M⁻¹ Aᵀ,  Δy = S⁻¹ (A M⁻¹ rhs_x - r_eq) with sign:
            # From A Δx = -r_eq  and  M Δx + Aᵀ Δy = rhs_x
            # ⇒ A M⁻¹ Aᵀ Δy = A M⁻¹ rhs_x + r_eq? wait A Δx = -r_eq
            # ⇒ A M⁻¹ (rhs_x - Aᵀ Δy) = -r_eq
            # ⇒ A M⁻¹ Aᵀ Δy = A M⁻¹ rhs_x + r_eq
            AM = jax.scipy.linalg.cho_solve((c_fac, True), A.T).T   # A M⁻¹
            S = AM @ A.T
            # Regularise S lightly.
            S = S + 1e-12 * jnp.eye(p, dtype=P.dtype)
            S_c = jnp.linalg.cholesky(S)
            dy_aff = jax.scipy.linalg.cho_solve(
                (S_c, True), AM @ rhs_x + r_eq,
            )
            dx_aff = jax.scipy.linalg.cho_solve((c_fac, True), rhs_x - A.T @ dy_aff)

            def solve_corr(rhs):
                dy = jax.scipy.linalg.cho_solve(
                    (S_c, True), AM @ rhs + r_eq,
                )
                # For corrector the eq residual target stays the same
                # (linear): we still want A(x+Δx)=b, so A Δx = -r_eq.
                dx = jax.scipy.linalg.cho_solve((c_fac, True), rhs - A.T @ dy)
                return dx, dy

        ds_aff = -(r_prim + G @ dx_aff)
        dz_aff = -z - D * ds_aff

        # ── Centering parameter ──────────────────────────────────────────
        a_aff = jnp.minimum(
            _fraction_to_boundary_nn(s, ds_aff, tau),
            _fraction_to_boundary_nn(z, dz_aff, tau),
        )
        mu_aff = jnp.dot(s + a_aff * ds_aff, z + a_aff * dz_aff) / m
        sigma = jnp.clip((mu_aff / (mu + 1e-30)) ** 3, 0.0, 1.0)

        # ── Corrector (Mehrotra cross term) ──────────────────────────────
        # s∘z + s∘Δz + z∘Δs + Δs_aff∘Δz_aff = σ μ 1
        # ⇒ Δz = (σμ 1 - Δs_aff∘Δz_aff)/s - D Δs - z
        # Complementarity linearisation:
        #   S dz + Z ds = -S Z 1 + σ μ 1 - Δs_aff ∘ Δz_aff
        #   ⇒ dz = -z + cent - D ds,   cent = (σ μ 1 - cross) / s
        # Dual + elimination then give
        #   (P + Gᵀ D G) Δx = -(P x + q + Aᵀ y) - Gᵀ (D r_prim + cent)
        cross = ds_aff * dz_aff
        cent = (sigma * mu - cross) / s
        rhs_c = -(P @ x + q) - G.T @ (D * r_prim + cent)
        if p > 0:
            rhs_c = rhs_c - A.T @ y
        dx, dy = solve_corr(rhs_c)
        ds = -(r_prim + G @ dx)
        dz = cent - D * ds - z

        # ── Common step length ───────────────────────────────────────────
        eta = jnp.maximum(0.99, 1.0 - mu)   # plan: max(0.9, 1-μ); keep ≤ 0.99
        eta = jnp.minimum(eta, 0.99)
        a_p = _fraction_to_boundary_nn(s, ds, tau)
        a_d = _fraction_to_boundary_nn(z, dz, tau)
        alpha = jnp.minimum(1.0, eta * jnp.minimum(a_p, a_d))

        x_new = x + alpha * dx
        s_new = s + alpha * ds
        z_new = z + alpha * dz
        y_new = y + alpha * dy if p > 0 else y

        # Keep strictly positive.
        s_new = jnp.maximum(s_new, 1e-300)
        z_new = jnp.maximum(z_new, 1e-300)

        step_ok = (
            jnp.isfinite(x_new).all()
            & jnp.isfinite(s_new).all()
            & jnp.isfinite(z_new).all()
            & ok
        )
        x_new = jnp.where(step_ok, x_new, x)
        s_new = jnp.where(step_ok, s_new, s)
        z_new = jnp.where(step_ok, z_new, z)
        y_new = jnp.where(step_ok, y_new, y)

        delta_new = jnp.where(
            ~ok,
            jnp.minimum(jnp.maximum(d_used * 10.0, 1e-8), delta_max),
            jnp.maximum(d_used / 3.0, 1e-12),
        )

        # ── Termination / certificates ───────────────────────────────────
        pres, dres, gap, pobj, solved, inaccurate, cert_p, cert_d = _termination(
            P, q, G, h, A, b, x_new, s_new, z_new, y_new, tol, tol_inacc,
        )
        # Stop only on exact solve, infeasibility, or numerical failure.
        # "Solved inaccurately" is assigned after the loop if we exit by
        # maxiter with residuals below tol_inaccurate.
        status = jnp.where(
            solved, jnp.int32(0),
            jnp.where(
                cert_p, jnp.int32(2),
                jnp.where(
                    cert_d, jnp.int32(3),
                    jnp.where(~step_ok, jnp.int32(5), jnp.int32(4)),
                ),
            ),
        )
        done = (status == 0) | (status == 2) | (status == 3) | (status == 5)

        if verbose > 1:
            jax.debug.print(
                "QP-IPM iter {it}: pres={p:.2e} dres={d:.2e} gap={g:.2e} "
                "mu={mu:.2e} a={a:.3f} st={st}",
                it=state['n_iter'] + 1, p=pres, d=dres, g=gap,
                mu=jnp.dot(s_new, z_new) / m, a=alpha, st=status,
            )

        return {
            'x': x_new, 's': s_new, 'z': z_new, 'y': y_new,
            'delta': delta_new,
            'n_iter': state['n_iter'] + 1,
            'pres': pres, 'dres': dres, 'gap': gap, 'pobj': pobj,
            'status_code': status,
            'done': done,
        }

    final = while_loop(cont, step, init_state)

    # ── Unscale ──────────────────────────────────────────────────────────
    x_out = d_x * final['x']
    z_out = d_z * final['z'] / c_scale
    s_out = final['s'] / d_z
    y_out = d_y * final['y'] / c_scale if p > 0 else final['y']
    # pobj in original units: ½ xᵀ P_orig x + q_origᵀ x
    # = (½ x̃ᵀ P̃ x̃ + q̃ᵀ x̃) / c_scale, approximately; recompute from
    # the unscaled data for reporting.
    pobj_out = 0.5 * jnp.dot(x_out, prob.P @ x_out) + jnp.dot(prob.q, x_out)

    status = final['status_code']
    # If we exited by maxiter still at status 4, promote to inaccurate
    # when residuals are below the loose tolerance.
    inaccurate = (
        (final['pres'] < tol_inacc)
        & (final['dres'] < tol_inacc)
        & (final['gap'] < tol_inacc)
    )
    status = jnp.where(
        (status == 4) & inaccurate, jnp.int32(1), status,
    )

    if verbose > 0:
        jax.debug.print(
            "QP-IPM DONE: iters={it} status={st} pres={p:.2e} dres={d:.2e}",
            it=final['n_iter'], st=status, p=final['pres'], d=final['dres'],
        )

    return {
        'fin_x': x_out,
        'fin_s': s_out,
        'fin_z': z_out,
        'fin_y': y_out,
        'fin_f': pobj_out,
        'fin_g': prob.G @ x_out - prob.h,   # g = Gx - h  (<= 0 when feasible)
        'fin_h': jnp.zeros(0, dtype=x_out.dtype) if p == 0 else (prob.A @ x_out - prob.b),
        'status_code': status,
        'niter': final['n_iter'],
        'pres': final['pres'],
        'dres': final['dres'],
        'gap': final['gap'],
        'pobj': pobj_out,
    }


# ── SOCP branch ──────────────────────────────────────────────────────────

def _unpack_cones(cones, s, z):
    """Split orthant / SOC slices. Returns (s_l, z_l, list of (s_q, z_q) batches)."""
    offset = 0
    s_l = z_l = None
    soc_batches = []  # (s_q shaped (count, dim), z_q, start)
    for block in cones:
        if block[0] == 'l':
            m_l = int(block[1])
            s_l = s[offset:offset + m_l]
            z_l = z[offset:offset + m_l]
            offset += m_l
        else:
            dim, count = int(block[1]), int(block[2])
            s_q = s[offset:offset + dim * count].reshape(count, dim)
            z_q = z[offset:offset + dim * count].reshape(count, dim)
            soc_batches.append((s_q, z_q, offset, dim, count))
            offset += dim * count
    return s_l, z_l, soc_batches


def _soc_jordan(u, v):
    """Jordan product u ∘ v for a batch of second-order cones, shape (N, d)."""
    u0 = u[:, :1]
    v0 = v[:, :1]
    ub = u[:, 1:]
    vb = v[:, 1:]
    out0 = jnp.sum(u * v, axis=1, keepdims=True)
    outb = u0 * vb + v0 * ub
    return jnp.concatenate([out0, outb], axis=1)


def _soc_J(u):
    """Apply the quadratic-form matrix J = diag(+1, −1, …) to a batch."""
    return jnp.concatenate([u[:, :1], -u[:, 1:]], axis=1)


def _soc_nt_scaling(s, z):
    r"""
    Nesterov–Todd scaling for a batch of second-order cones.

    Following Vandenberghe (CVXOPT notes §4):

        s̄ = s / √(sᵀ J s),   z̄ = z / √(zᵀ J z),
        γ  = √((1 + s̄ᵀ z̄)/2),
        w̄ = (s̄ + J z̄) / (2 γ),
        η  = (sᵀ J s / zᵀ J z)^{1/4}.

    Returns ``(eta, w_bar, lam)`` where ``lam = W z`` (scaled dual).
    Identity check: ``W z = W⁻¹ s`` (tested in unit tests).
    """
    sJs = jnp.sum(s * _soc_J(s), axis=1, keepdims=True)
    zJz = jnp.sum(z * _soc_J(z), axis=1, keepdims=True)
    sJs = jnp.maximum(sJs, 1e-300)
    zJz = jnp.maximum(zJz, 1e-300)
    s_bar = s / jnp.sqrt(sJs)
    z_bar = z / jnp.sqrt(zJz)
    gamma = jnp.sqrt(jnp.maximum((1.0 + jnp.sum(s_bar * z_bar, axis=1, keepdims=True)) / 2.0, 1e-300))
    w_bar = (s_bar + _soc_J(z_bar)) / (2.0 * gamma)
    eta = (sJs / zJz) ** 0.25
    # λ = W z.  W = η (2 w̄ w̄ᵀ − J) applied carefully:
    # W z = η (2 (w̄ᵀ z) w̄ − J z) is WRONG; the correct action is
    # W = η [[w0, w̄ᵀ], [w̄, I + w̄ w̄ᵀ/(w0+1)]] in the spectral form,
    # or equivalently λ = η * jordan(w_bar_ext, z) style.  We use the
    # closed form from CVXOPT:
    #   λ₀ = η (w̄ᵀ J? no) — use λ = W^{-T} s = W z with
    #   λ = η * ( (z0 + z̄·w̄)*e_scaled ... )
    # Practical formula (Vandenberghe eq. 19–22):
    #   λ = η * Q(w_bar) z where Q(w) = 2 w wᵀ − J(wᵀ J? )
    # Q(w) v = 2 (wᵀ v) w − J v   is for the Lorentz Jordan product
    # when wᵀ J w = 1.
    # Wait: for second-order cone, Q(w) = 2 w wᵀ − (wᵀ J w) J, and
    # when wᵀ J w = 1, Q(w) v = 2 (wᵀ v) w − J v.
    # NT: W = η Q(w_bar)^{1/2}, but W z = η Q(w_bar)^{1/2} z.
    # Simpler: λ₀ = √(η² (zᵀ J?)/…) — use the identity λ = W^{-1} s
    # computed as jordan-sqrt style.  Direct:
    #   λ = (√(sᵀz / d) stuff).  We'll compute λ via
    #   λ = η * Q(w_bar)^{1/2} z using the spectral form of Q^{1/2}.
    #
    # Spectral: Q(w)^{1/2} v = (wᵀ v)/(w0+1) * (w + e) + J? ...
    # CVXOPT implements:
    #   a = √(2/(1+w0)) ;  λ = η * ( (z0 + w·z)/√2 * e_scaled + …)
    # Use the well-known:
    #   tmp = z + _soc_J(z) * 0 + ...
    # Practical stable formula from CVXOPT src (misc.c):
    #   λ = W^{-T} s with W^{-T} = (1/η) Q(w_bar)^{-1/2}
    # and Q(w)^{-1/2} = Q(w_bar_inv)^{1/2} where w_inv = J w / (wᵀJw) = J w.
    #
    # Since w_barᵀ J w_bar = 1, Q(w_bar)^{-1} = Q(J w_bar).
    # We compute λ = W z as η Q(w_bar)^{1/2} z:
    w0 = w_bar[:, :1]
    wb = w_bar[:, 1:]
    # Q^{1/2}(w) v = α (β 1̂ + v) style:
    # Let e = (1,0,…),  u = w + e.
    # Q^{1/2}(w) = [[w0, wbᵀ],[wb, I]] / something? 
    # From CVXOPT documentation eq (17):
    #   W = η [ w0, w̄ᵀ ; w̄, I + w̄ w̄ᵀ/(w0+1) ]
    # (this IS W, not W^2 — for the second-order cone NT scaling).
    # Verify W z = W^{-1} s below in tests.
    Wz0 = w0 * z[:, :1] + jnp.sum(wb * z[:, 1:], axis=1, keepdims=True)
    Wzb = wb * z[:, :1] + z[:, 1:] + wb * (
        jnp.sum(wb * z[:, 1:], axis=1, keepdims=True) / (w0 + 1.0)
    )
    lam = eta * jnp.concatenate([Wz0, Wzb], axis=1)
    return eta, w_bar, lam


def _soc_apply_Winv(eta, w_bar, v):
    """Apply W⁻¹ to a batch: W⁻¹ = (1/η) [w0, −w̄ᵀ; −w̄, I + w̄ w̄ᵀ/(w0+1)]."""
    w0 = w_bar[:, :1]
    wb = w_bar[:, 1:]
    v0 = v[:, :1]
    vb = v[:, 1:]
    out0 = w0 * v0 - jnp.sum(wb * vb, axis=1, keepdims=True)
    outb = -wb * v0 + vb + wb * (
        jnp.sum(wb * vb, axis=1, keepdims=True) / (w0 + 1.0)
    )
    return jnp.concatenate([out0, outb], axis=1) / eta


def _soc_apply_W(eta, w_bar, v):
    """Apply W to a batch (inverse of :func:`_soc_apply_Winv`)."""
    w0 = w_bar[:, :1]
    wb = w_bar[:, 1:]
    v0 = v[:, :1]
    vb = v[:, 1:]
    out0 = w0 * v0 + jnp.sum(wb * vb, axis=1, keepdims=True)
    outb = wb * v0 + vb + wb * (
        jnp.sum(wb * vb, axis=1, keepdims=True) / (w0 + 1.0)
    )
    return eta * jnp.concatenate([out0, outb], axis=1)


def _soc_apply_Winv2_G(eta, w_bar, G_k):
    r"""
    Contribution of one SOC batch to ``Gᵀ W⁻² G``.

    ``W⁻² = η⁻² (2 J w̄ w̄ᵀ J − J)``, so

        Gᵀ W⁻² G = η⁻² [ 2 (J w̄)ᵀ-scaled rows  −  Gᵀ J G ]

    Implemented as a signed diagonal part plus a rank-``count`` update
    ``Uᵀ U`` with ``U_i = √2 / η_i * (J w̄_i)ᵀ G_k[i]``.
    """
    # G_k: (count, dim, n)
    count, dim, n = G_k.shape
    inv_eta2 = 1.0 / (eta[:, 0] ** 2 + 1e-300)          # (count,)
    Jw = _soc_J(w_bar)                                   # (count, dim)

    # Signed diagonal: −J scaled by η⁻², i.e. contrib −η⁻² Gᵀ J G
    # J = diag(+1,−1,…), so Gᵀ J G = G0ᵀ G0 − Grestᵀ Grest
    # −Gᵀ J G = −G0ᵀG0 + GrestᵀGrest
    dsgn = jnp.concatenate(
        [-jnp.ones((count, 1)), jnp.ones((count, dim - 1))], axis=1,
    ) * inv_eta2[:, None]                                # (count, dim)
    # Σ_i G_k[i].T @ diag(dsgn[i]) @ G_k[i]
    Gd = G_k * dsgn[:, :, None]                          # (count, dim, n)
    # tensordot over (count, dim)
    diag_part = jnp.einsum('cdi,cdj->ij', G_k, Gd)

    # Rank-count update: Uᵀ U with U_i = √2/η (J w)ᵀ G_k[i]
    U = jnp.sqrt(2.0) * jnp.einsum('cd,cdn->cn', Jw, G_k)  # (count, n)
    U = U * jnp.sqrt(inv_eta2)[:, None]
    rank_part = U.T @ U
    return diag_part + rank_part


def _soc_step_to_boundary(lam, dlam, tau):
    r"""
    Max step keeping ``λ + α dλ`` in the SOC.

    Solve the Lorentz-boundary quadratic in ``α`` for each cone

        (λ₀ + α d₀)² − ‖λ̄ + α d̄‖² = 0

    i.e.

        γ + 2 α β + α² δ = 0,
        γ = λ₀² − ‖λ̄‖²,   β = λ₀ d₀ − λ̄·d̄,   δ = d₀² − ‖d̄‖²,

    take the smallest positive root (and the root of λ₀ + α d₀ = 0 when
    d₀ < 0), then scale by ``tau`` and minimise over the batch.
    """
    lam0 = lam[:, 0]
    d0 = dlam[:, 0]
    lamb = lam[:, 1:]
    db = dlam[:, 1:]
    gamma = lam0 * lam0 - jnp.sum(lamb * lamb, axis=1)   # > 0 in the interior
    beta = lam0 * d0 - jnp.sum(lamb * db, axis=1)
    delta = d0 * d0 - jnp.sum(db * db, axis=1)

    # Roots of δ α² + 2 β α + γ = 0.  When |δ| is tiny, fall back to the
    # linear root −γ / (2 β) (only limiting when β < 0).
    disc = jnp.maximum(beta * beta - delta * gamma, 0.0)
    sqrt_disc = jnp.sqrt(disc)

    def _root(sign):
        # (−β ± √disc) / δ, with a linear fallback.
        num = -beta + sign * sqrt_disc
        return jnp.where(
            jnp.abs(delta) > 1e-14,
            num / delta,
            jnp.where(beta < 0.0, -gamma / (2.0 * beta), jnp.inf),
        )

    r_plus = _root(1.0)
    r_minus = _root(-1.0)
    # Keep only positive roots; ignore non-positive.
    r_plus = jnp.where(r_plus > 0.0, r_plus, jnp.inf)
    r_minus = jnp.where(r_minus > 0.0, r_minus, jnp.inf)
    a_quad = jnp.minimum(r_plus, r_minus)

    # Also hit λ₀ + α d₀ = 0 when stepping toward λ₀ ≤ 0.
    a_comp = jnp.where(d0 < 0.0, -lam0 / d0, jnp.inf)
    a_cone = jnp.minimum(a_quad, a_comp)
    return jnp.minimum(1.0, tau * jnp.min(a_cone))


def _soc_min_eig(u):
    """Minimum Lorentz eigenvalue u0 − ‖ū‖ for a batch."""
    return u[:, 0] - jnp.linalg.norm(u[:, 1:], axis=1)


def _solve_socp(prob, cones, opts, verbose, nu):
    r"""
    Mehrotra PDIP for a product of nonnegative orthants and second-order
    cones.  Same condensed-Cholesky skeleton as the QP branch; the
    scaling ``W`` is Nesterov–Todd and ``Gᵀ W⁻² G`` is assembled as a
    signed diagonal plus a rank-``n_cones`` update.
    """
    P, q, G, h, A, b = prob.P, prob.q, prob.G, prob.h, prob.A, prob.b
    n = P.shape[0]
    m = G.shape[0]
    p = A.shape[0]
    tol = opts['tol']
    tol_inacc = opts['tol_inaccurate']
    maxiter = int(opts['maxiter'])
    tau = opts['tau']
    delta_init = opts['delta_init']
    delta_max = opts['delta_max']

    # ── Ruiz equilibration (cone-preserving) ─────────────────────────────
    P, q, G, h, A, b, d_x, d_z, d_y, c_scale = _ruiz_equilibrate(
        P, q, G, h, A, b, cones,
        opts['ruiz_iters'], opts['ruiz_clip'],
    )

    # ── Initialisation (coneqp-style with cone shifts) ───────────────────
    M0 = P + G.T @ G
    if p > 0:
        M0 = M0 + A.T @ A
    rhs0 = -q + G.T @ h
    if p > 0:
        rhs0 = rhs0 + A.T @ b
    x, _, _, _ = _chol_solve_reg(M0, rhs0, delta_init, delta_max)
    s = h - G @ x
    z = -s

    # Shift orthant and SOC into the interior.
    s_l, z_l, soc_batches = _unpack_cones(cones, s, z)
    if s_l is not None:
        shift_s = jnp.maximum(1.0 - jnp.min(s_l), 0.0)
        shift_z = jnp.maximum(1.0 - jnp.min(z_l), 0.0)
        s = s.at[:s_l.shape[0]].add(shift_s)
        z = z.at[:s_l.shape[0]].add(shift_z)
    for s_q, z_q, start, dim, count in soc_batches:
        # Re-read after possible orthant shift.
        s_q = s[start:start + dim * count].reshape(count, dim)
        z_q = z[start:start + dim * count].reshape(count, dim)
        sh_s = jnp.maximum(1.0 - jnp.min(_soc_min_eig(s_q)), 0.0)
        sh_z = jnp.maximum(1.0 - jnp.min(_soc_min_eig(z_q)), 0.0)
        e = jnp.zeros((count, dim)).at[:, 0].set(1.0)
        s = s.at[start:start + dim * count].set((s_q + sh_s * e).ravel())
        z = z.at[start:start + dim * count].set((z_q + sh_z * e).ravel())
    y = jnp.zeros(p, dtype=P.dtype)

    if verbose > 0:
        jax.debug.print(
            "SOCP-IPM: n={n}, m={m}, p={p}, nu={nu}, maxiter={mi}",
            n=n, m=m, p=p, nu=nu, mi=maxiter,
        )

    init_state = {
        'x': x, 's': s, 'z': z, 'y': y,
        'delta': jnp.float64(delta_init),
        'n_iter': jnp.int32(0),
        'pres': jnp.float64(jnp.inf),
        'dres': jnp.float64(jnp.inf),
        'gap': jnp.float64(jnp.inf),
        'pobj': jnp.float64(jnp.inf),
        'status_code': jnp.int32(4),
        'done': jnp.bool_(False),
    }

    def cont(state):
        return (state['n_iter'] < maxiter) & (~state['done'])

    def step(state):
        x, s, z, y = state['x'], state['s'], state['z'], state['y']
        delta = state['delta']

        # ── Residuals ────────────────────────────────────────────────────
        r_dual = P @ x + q + G.T @ z
        if p > 0:
            r_dual = r_dual + A.T @ y
        r_prim = G @ x + s - h
        r_eq = (A @ x - b) if p > 0 else jnp.zeros(0)
        mu = jnp.dot(s, z) / nu

        # ── NT scaling + assemble Gᵀ W⁻² G ───────────────────────────────
        s_l, z_l, soc_batches = _unpack_cones(cones, s, z)
        GTW2G = jnp.zeros((n, n), dtype=P.dtype)
        # Orthant: W⁻² = diag(z/s), contribution G_lᵀ diag(z/s) G_l.
        offset = 0
        if s_l is not None:
            m_l = s_l.shape[0]
            D_l = z_l / s_l
            G_l = G[:m_l]
            GTW2G = GTW2G + (G_l * D_l[:, None]).T @ G_l
            offset = m_l

        # Per-SOC-batch NT data for later corrector / step length.
        nt_data = []
        for s_q, z_q, start, dim, count in soc_batches:
            eta, w_bar, lam = _soc_nt_scaling(s_q, z_q)
            G_k = G[start:start + dim * count].reshape(count, dim, n)
            GTW2G = GTW2G + _soc_apply_Winv2_G(eta, w_bar, G_k)
            nt_data.append((eta, w_bar, lam, start, dim, count))

        M = P + GTW2G

        # Helper: apply W⁻ᵀ to a vector in the product cone (for Δs recovery
        # paths that need scaled residuals).  For the condensed system we
        # only need W⁻² and the orthant D.
        def apply_Winv2(v):
            out = jnp.zeros_like(v)
            if s_l is not None:
                m_l = s_l.shape[0]
                out = out.at[:m_l].set((z_l / s_l) * v[:m_l])
            for eta, w_bar, _lam, start, dim, count in nt_data:
                vv = v[start:start + dim * count].reshape(count, dim)
                # W⁻² v = η⁻² (2 Jw (Jw)ᵀ − J) v
                inv_eta2 = 1.0 / (eta[:, 0] ** 2 + 1e-300)
                Jw = _soc_J(w_bar)
                Jv = _soc_J(vv)
                term = 2.0 * Jw * jnp.sum(Jw * vv, axis=1, keepdims=True) - Jv
                out = out.at[start:start + dim * count].set(
                    (inv_eta2[:, None] * term).ravel()
                )
            return out

        # ── Predictor ────────────────────────────────────────────────────
        # Same elimination as QP with D ← W⁻²:
        # RHS_x = -(P x + q + Aᵀ y) - Gᵀ W⁻² r_prim
        # (using that affine Δz = -W⁻² Δs - W⁻²(s∘z)/… ≈ -W⁻² Δs - z
        #  for the orthant; for SOC the affine target is λ∘λ = 0 in
        #  scaled space, giving Δz = -W⁻² Δs - z when λ∘λ maps to s∘z).
        # For product cones the affine complementarity is s∘z = 0 in the
        # Jordan sense; the condensed RHS uses W⁻² r_prim as in QP.
        rhs_x = -(P @ x + q) - G.T @ apply_Winv2(r_prim)
        if p > 0:
            rhs_x = rhs_x - A.T @ y

        if p == 0:
            dx_aff, c_fac, d_used, ok = _chol_solve_reg(M, rhs_x, delta, delta_max)
            dy_aff = jnp.zeros(0, dtype=P.dtype)

            def solve_corr(rhs):
                return jax.scipy.linalg.cho_solve((c_fac, True), rhs), jnp.zeros(0, dtype=P.dtype)
        else:
            dx_tmp, c_fac, d_used, ok = _chol_solve_reg(M, rhs_x, delta, delta_max)
            AM = jax.scipy.linalg.cho_solve((c_fac, True), A.T).T
            S = AM @ A.T + 1e-12 * jnp.eye(p, dtype=P.dtype)
            S_c = jnp.linalg.cholesky(S)
            dy_aff = jax.scipy.linalg.cho_solve((S_c, True), AM @ rhs_x + r_eq)
            dx_aff = jax.scipy.linalg.cho_solve((c_fac, True), rhs_x - A.T @ dy_aff)

            def solve_corr(rhs):
                dy = jax.scipy.linalg.cho_solve((S_c, True), AM @ rhs + r_eq)
                dx = jax.scipy.linalg.cho_solve((c_fac, True), rhs - A.T @ dy)
                return dx, dy

        ds_aff = -(r_prim + G @ dx_aff)
        # Affine dual: in scaled coords λ ∘ dλ_aff = -λ ∘ λ, so
        # dλ_aff = -λ, hence dz_aff = W⁻¹ dλ_aff = -z (since λ = W z).
        # More carefully with linearisation: dz_aff = -apply_Winv2(ds_aff) - z
        # for the orthant-style product; for SOC Jordan it is analogous.
        dz_aff = -apply_Winv2(ds_aff) - z

        # ── Centering ────────────────────────────────────────────────────
        # Step to boundary in the product cone.
        a_list = []
        if s_l is not None:
            m_l = s_l.shape[0]
            a_list.append(_fraction_to_boundary_nn(s[:m_l], ds_aff[:m_l], tau))
            a_list.append(_fraction_to_boundary_nn(z[:m_l], dz_aff[:m_l], tau))
        for eta, w_bar, lam, start, dim, count in nt_data:
            dlam_s = _soc_apply_Winv(eta, w_bar, ds_aff[start:start + dim * count].reshape(count, dim))
            dlam_z = _soc_apply_W(eta, w_bar, dz_aff[start:start + dim * count].reshape(count, dim))
            a_list.append(_soc_step_to_boundary(lam, dlam_s, tau))
            a_list.append(_soc_step_to_boundary(lam, dlam_z, tau))
        a_aff = jnp.min(jnp.stack(a_list)) if a_list else 1.0

        mu_aff = jnp.dot(s + a_aff * ds_aff, z + a_aff * dz_aff) / nu
        sigma = jnp.clip((mu_aff / (mu + 1e-30)) ** 3, 0.0, 1.0)

        # ── Corrector with Jordan cross term ─────────────────────────────
        # Orthant: cent = (σμ − ds_aff ∘ dz_aff) / s
        # SOC:     target_sc = σμ e − (W⁻ᵀ ds_aff) ∘ (W dz_aff)
        #          cent = W⁻¹ (λ \ target_sc)
        # Condensed RHS matches the QP elimination with D ← W⁻²:
        #   RHS_x = -(Px+q+Aᵀy) - Gᵀ (W⁻² r_prim − cent)
        # Dual recovery: Δz = cent − W⁻² Δs − z
        def jordan_div(lam_b, v):
            det_b = jnp.sum(lam_b * _soc_J(lam_b), axis=1, keepdims=True)
            det_b = jnp.where(jnp.abs(det_b) < 1e-300, 1e-300, det_b)
            lam0_b = lam_b[:, :1]
            lamb_b = lam_b[:, 1:]
            v0 = v[:, :1]
            vb = v[:, 1:]
            rho0_b = (lam0_b * v0 - jnp.sum(lamb_b * vb, axis=1, keepdims=True)) / det_b
            rhob_b = (lam0_b * vb - v0 * lamb_b) / det_b
            return jnp.concatenate([rho0_b, rhob_b], axis=1)

        cent = jnp.zeros(m, dtype=P.dtype)
        if s_l is not None:
            m_l = s_l.shape[0]
            cent = cent.at[:m_l].set(
                (sigma * mu - ds_aff[:m_l] * dz_aff[:m_l]) / s[:m_l]
            )
        for eta, w_bar, lam, start, dim, count in nt_data:
            ds_sc = _soc_apply_Winv(
                eta, w_bar,
                ds_aff[start:start + dim * count].reshape(count, dim),
            )
            dz_sc = _soc_apply_W(
                eta, w_bar,
                dz_aff[start:start + dim * count].reshape(count, dim),
            )
            cross_sc = _soc_jordan(ds_sc, dz_sc)
            e = jnp.zeros((count, dim)).at[:, 0].set(1.0)
            target_sc = sigma * mu * e - cross_sc
            cent_sc = jordan_div(lam, target_sc)
            cent = cent.at[start:start + dim * count].set(
                _soc_apply_Winv(eta, w_bar, cent_sc).ravel()
            )

        # Elimination with Δz = cent − W⁻² Δs − z, Δs = −r_prim − G Δx:
        #   (P + Gᵀ W⁻² G) Δx = -(Px+q+Aᵀy) - Gᵀ (W⁻² r_prim + cent)
        rhs_c = -(P @ x + q) - G.T @ (apply_Winv2(r_prim) + cent)
        if p > 0:
            rhs_c = rhs_c - A.T @ y

        dx, dy = solve_corr(rhs_c)
        ds = -(r_prim + G @ dx)
        dz = cent - apply_Winv2(ds) - z

        # ── Step length in the product cone ──────────────────────────────
        a_list = []
        if s_l is not None:
            m_l = s_l.shape[0]
            a_list.append(_fraction_to_boundary_nn(s[:m_l], ds[:m_l], tau))
            a_list.append(_fraction_to_boundary_nn(z[:m_l], dz[:m_l], tau))
        for eta, w_bar, lam, start, dim, count in nt_data:
            dlam_s = _soc_apply_Winv(eta, w_bar, ds[start:start + dim * count].reshape(count, dim))
            dlam_z = _soc_apply_W(eta, w_bar, dz[start:start + dim * count].reshape(count, dim))
            a_list.append(_soc_step_to_boundary(lam, dlam_s, tau))
            a_list.append(_soc_step_to_boundary(lam, dlam_z, tau))
        eta_step = jnp.minimum(jnp.maximum(0.99, 1.0 - mu), 0.99)
        alpha = jnp.minimum(1.0, eta_step * jnp.min(jnp.stack(a_list)))

        x_new = x + alpha * dx
        s_new = s + alpha * ds
        z_new = z + alpha * dz
        y_new = y + alpha * dy if p > 0 else y

        # Project lightly back into the cone if floating-point drifted.
        if s_l is not None:
            m_l = s_l.shape[0]
            s_new = s_new.at[:m_l].set(jnp.maximum(s_new[:m_l], 1e-300))
            z_new = z_new.at[:m_l].set(jnp.maximum(z_new[:m_l], 1e-300))
        for _eta, _wb, _lam, start, dim, count in nt_data:
            e = jnp.zeros((count, dim)).at[:, 0].set(1.0)
            sb = s_new[start:start + dim * count].reshape(count, dim)
            zb = z_new[start:start + dim * count].reshape(count, dim)
            s_new = s_new.at[start:start + dim * count].set(
                (sb + jnp.maximum(1e-8 - _soc_min_eig(sb), 0.0)[:, None] * e).ravel()
            )
            z_new = z_new.at[start:start + dim * count].set(
                (zb + jnp.maximum(1e-8 - _soc_min_eig(zb), 0.0)[:, None] * e).ravel()
            )

        step_ok = (
            jnp.isfinite(x_new).all()
            & jnp.isfinite(s_new).all()
            & jnp.isfinite(z_new).all()
            & ok
        )
        x_new = jnp.where(step_ok, x_new, x)
        s_new = jnp.where(step_ok, s_new, s)
        z_new = jnp.where(step_ok, z_new, z)
        y_new = jnp.where(step_ok, y_new, y)

        delta_new = jnp.where(
            ~ok,
            jnp.minimum(jnp.maximum(d_used * 10.0, 1e-8), delta_max),
            jnp.maximum(d_used / 3.0, 1e-12),
        )

        pres, dres, gap, pobj, solved, inaccurate, cert_p, cert_d = _termination(
            P, q, G, h, A, b, x_new, s_new, z_new, y_new, tol, tol_inacc,
        )
        status = jnp.where(
            solved, jnp.int32(0),
            jnp.where(
                cert_p, jnp.int32(2),
                jnp.where(
                    cert_d, jnp.int32(3),
                    jnp.where(~step_ok, jnp.int32(5), jnp.int32(4)),
                ),
            ),
        )
        done = (status == 0) | (status == 2) | (status == 3) | (status == 5)

        if verbose > 1:
            jax.debug.print(
                "SOCP-IPM iter {it}: pres={p:.2e} dres={d:.2e} gap={g:.2e} a={a:.3f} st={st}",
                it=state['n_iter'] + 1, p=pres, d=dres, g=gap, a=alpha, st=status,
            )

        return {
            'x': x_new, 's': s_new, 'z': z_new, 'y': y_new,
            'delta': delta_new,
            'n_iter': state['n_iter'] + 1,
            'pres': pres, 'dres': dres, 'gap': gap, 'pobj': pobj,
            'status_code': status,
            'done': done,
        }

    final = while_loop(cont, step, init_state)

    x_out = d_x * final['x']
    z_out = d_z * final['z'] / c_scale
    s_out = final['s'] / d_z
    y_out = d_y * final['y'] / c_scale if p > 0 else final['y']
    pobj_out = 0.5 * jnp.dot(x_out, prob.P @ x_out) + jnp.dot(prob.q, x_out)
    status = final['status_code']
    inaccurate = (
        (final['pres'] < tol_inacc)
        & (final['dres'] < tol_inacc)
        & (final['gap'] < tol_inacc)
    )
    status = jnp.where((status == 4) & inaccurate, jnp.int32(1), status)

    if verbose > 0:
        jax.debug.print(
            "SOCP-IPM DONE: iters={it} status={st} pres={p:.2e} dres={d:.2e}",
            it=final['n_iter'], st=status, p=final['pres'], d=final['dres'],
        )

    return {
        'fin_x': x_out,
        'fin_s': s_out,
        'fin_z': z_out,
        'fin_y': y_out,
        'fin_f': pobj_out,
        'fin_g': prob.G @ x_out - prob.h,
        'fin_h': jnp.zeros(0, dtype=x_out.dtype) if p == 0 else (prob.A @ x_out - prob.b),
        'status_code': status,
        'niter': final['n_iter'],
        'pres': final['pres'],
        'dres': final['dres'],
        'gap': final['gap'],
        'pobj': pobj_out,
    }
