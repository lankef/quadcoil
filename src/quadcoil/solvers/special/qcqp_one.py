"""Jittable one-constraint convex QCQP solver via simultaneous diagonalization."""
import jax
import jax.numpy as jnp
from jax import grad, hessian, value_and_grad
from jax.lax import while_loop, cond


_DEFAULT_OPTIONS = {
    'lam_init': 1.0,
    'lam_grow': 2.0,
    'lam_max': 1e12,
    'root_maxiter': 100,
    # None → 10 * finfo(eps) (relative).
    'root_tol': None,
    'ctol': 0.0,
}

QCQP_ONE_STATUS = ('unconstrained', 'active', 'infeasible')


def _materialize_quadratic(fun, x0):
    r'''Exact quadratic coefficients of ``fun`` about ``x0``.

    Returns ``(H, b, c)`` such that
    ``fun(x) = 0.5 (x-x0)^T H (x-x0) + b · (x-x0) + c``.
    '''
    H = hessian(fun)(x0)
    H = 0.5 * (H + H.T)
    b = grad(fun)(x0)
    c = fun(x0)
    return H, b, c


def _simultaneous_diagonalize(H_f, H_g_scaled, b_f, b_g_scaled, n, eps):
    r'''Diagonalize the pencil ``(H_f, H_g_scaled)`` via Cholesky of their sum.

    Returns ``(a, b, p, q, L, Q)`` for coordinates
    ``y = Q^T L^T (x - x0)``:

    * ``f = 0.5 sum a_i y_i^2 + p · y + c_f``
    * ``g~ = 0.5 sum b_i y_i^2 + q · y + c_g~``
    '''
    M = H_f + H_g_scaled
    # Tiny ridge for numerical Cholesky when both Hessians are nearly singular.
    ridge = eps * (jnp.trace(M) + jnp.asarray(1.0, dtype=M.dtype))
    M = M + ridge * jnp.eye(n, dtype=M.dtype)
    L = jnp.linalg.cholesky(M)

    # A = L^{-1} H_f L^{-T}, B = L^{-1} H_g_scaled L^{-T}.
    Lf = jax.scipy.linalg.solve_triangular(L, H_f, lower=True)
    A = jax.scipy.linalg.solve_triangular(L, Lf.T, lower=True).T
    Lg = jax.scipy.linalg.solve_triangular(L, H_g_scaled, lower=True)
    B = jax.scipy.linalg.solve_triangular(L, Lg.T, lower=True).T

    A = 0.5 * (A + A.T)
    evals, Q = jnp.linalg.eigh(A)
    a = evals
    b = jnp.diag(Q.T @ B @ Q)

    # Linear terms in y: p = Q^T L^{-1} b_f, q = Q^T L^{-1} b_g_scaled.
    p = Q.T @ jax.scipy.linalg.solve_triangular(L, b_f, lower=True)
    q = Q.T @ jax.scipy.linalg.solve_triangular(L, b_g_scaled, lower=True)
    return a, b, p, q, L, Q


def _y_of_lam(a, b, p, q, lam, tau):
    r'''Closed-form minimizer of ``f + lam * g~`` in diagonal coordinates.

    Singular denominators fall back to the argmin of the other quadratic.
    '''
    denom = a + lam * b
    numer = -(p + lam * q)
    y_reg = numer / jnp.where(jnp.abs(denom) > tau, denom, 1.0)

    # a ~ 0: prefer argmin of g~ along that mode.
    y_a0 = jnp.where(jnp.abs(b) > tau, -q / b, 0.0)
    # b ~ 0: prefer argmin of f along that mode.
    y_b0 = jnp.where(jnp.abs(a) > tau, -p / a, 0.0)

    y = jnp.where(jnp.abs(denom) > tau, y_reg, jnp.where(jnp.abs(a) <= tau, y_a0, y_b0))
    return y


def _y_lam0(a, b, p, q, tau):
    r'''``lam → 0+``: argmin f, breaking ties by minimizing g~.'''
    y_f = jnp.where(jnp.abs(a) > tau, -p / a, 0.0)
    y_g = jnp.where(jnp.abs(b) > tau, -q / b, 0.0)
    return jnp.where(jnp.abs(a) > tau, y_f, y_g)


def _y_lam_inf(a, b, p, q, tau):
    r'''``lam → ∞``: argmin g~, breaking ties by minimizing f.'''
    y_g = jnp.where(jnp.abs(b) > tau, -q / b, 0.0)
    y_f = jnp.where(jnp.abs(a) > tau, -p / a, 0.0)
    return jnp.where(jnp.abs(b) > tau, y_g, y_f)


def _phi_from_y(y, b, q, c_g_scaled):
    return 0.5 * jnp.sum(b * y * y) + jnp.dot(q, y) + c_g_scaled


def _f_from_y(y, a, p, c_f):
    return 0.5 * jnp.sum(a * y * y) + jnp.dot(p, y) + c_f


def _x_from_y(y, x0, L, Q):
    # y = Q^T L^T (x - x0)  ⇒  x = x0 + L^{-T} Q y
    return x0 + jax.scipy.linalg.solve_triangular(L, Q @ y, lower=True, trans=1)


def _safeguarded_newton(a, b, p, q, c_g_scaled, tau, opts, verbose):
    r'''Find ``lam > 0`` with ``phi(lam) = 0`` by safeguarded Newton + bisection.'''
    lam_init = jnp.asarray(opts['lam_init'], dtype=jnp.float64)
    lam_grow = jnp.asarray(opts['lam_grow'], dtype=jnp.float64)
    lam_max = jnp.asarray(opts['lam_max'], dtype=jnp.float64)
    root_maxiter = int(opts['root_maxiter'])
    root_tol = opts['root_tol']
    ctol = opts['ctol']

    def phi_raw(lam):
        y = _y_of_lam(a, b, p, q, lam, tau)
        return _phi_from_y(y, b, q, c_g_scaled) - ctol

    phi0 = phi_raw(jnp.asarray(0.0, dtype=jnp.float64))
    y_inf = _y_lam_inf(a, b, p, q, tau)
    phi_inf = _phi_from_y(y_inf, b, q, c_g_scaled) - ctol
    gtol = root_tol * jnp.maximum(jnp.abs(phi0 - phi_inf), 1.0)

    # ----- Bracket -----
    def grow_cond(carry):
        lam_hi, phi_hi, k = carry
        return (phi_hi > 0.0) & (lam_hi < lam_max) & (k < root_maxiter)

    def grow_body(carry):
        lam_hi, _phi_hi, k = carry
        lam_new = jnp.minimum(lam_hi * lam_grow, lam_max)
        return lam_new, phi_raw(lam_new), k + 1

    lam_hi0 = lam_init
    phi_hi0 = phi_raw(lam_hi0)
    lam_hi, phi_hi, grow_iters = while_loop(
        grow_cond, grow_body, (lam_hi0, phi_hi0, jnp.int32(0)),
    )

    lam_lo = jnp.asarray(0.0, dtype=jnp.float64)
    phi_lo = phi0

    # ----- Safeguarded Newton -----
    def root_cond(carry):
        lam, lam_lo, lam_hi, _phi, k = carry
        width_ok = (lam_hi - lam_lo) > root_tol * jnp.maximum(lam_hi, 1.0)
        return width_ok & (k < root_maxiter)

    def root_body(carry):
        lam, lam_lo, lam_hi, phi, k = carry
        phi_val, dphi = value_and_grad(phi_raw)(lam)
        # Newton candidate; fall back to midpoint if derivative is tiny.
        newton = lam - phi_val / jnp.where(
            jnp.abs(dphi) > tau, dphi, jnp.asarray(1.0, dtype=lam.dtype),
        )
        mid = 0.5 * (lam_lo + lam_hi)
        use_newton = (
            (jnp.abs(dphi) > tau)
            & (newton > lam_lo)
            & (newton < lam_hi)
        )
        lam_new = jnp.where(use_newton, newton, mid)
        phi_new = phi_raw(lam_new)

        take_hi = phi_new > 0.0
        lam_lo_new = jnp.where(take_hi, lam_new, lam_lo)
        phi_lo_new = jnp.where(take_hi, phi_new, phi)  # unused; keep carry small
        del phi_lo_new
        lam_hi_new = jnp.where(take_hi, lam_hi, lam_new)
        return lam_new, lam_lo_new, lam_hi_new, phi_new, k + 1

    lam0 = lam_hi
    phi_start = phi_hi
    lam, lam_lo, lam_hi, phi, n_root = while_loop(
        root_cond,
        root_body,
        (lam0, lam_lo, lam_hi, phi_start, jnp.int32(0)),
    )

    # Prefer the Newton iterate when feasible; otherwise the feasible hi.
    lam_out = jnp.where(phi <= gtol, lam, lam_hi)
    niter = grow_iters + n_root

    if verbose > 0:
        jax.debug.print(
            'qcqp_one: active lam={lam}, phi={phi}, niter={n}',
            lam=lam_out, phi=phi_raw(lam_out), n=niter,
        )
    return lam_out, niter


def solve_qcqp_one(f, g, x_init, solver_options=None, verbose=0):
    r'''
    Solve a one-constraint convex QCQP.

    Cases (``status_code``)
    -----------------------
    0. ``min f`` is feasible → unconstrained
    1. Find ``lam>0`` with ``g(x(lam))=0`` → active
    2. ``min g`` still violates ``g<=0`` → infeasible

    Return keys
    -----------
    ``fin_x``, ``fin_f``, ``fin_g``, ``fin_h``, ``fin_z``, ``niter``,
    ``status_code``.  ``fin_z`` always has shape ``(1,)`` (zero when inactive).
    '''
    opts = {**_DEFAULT_OPTIONS}
    if solver_options is not None:
        opts.update(solver_options)

    x0 = jnp.asarray(x_init, dtype=jnp.float64)
    n = x0.shape[0]
    eps = jnp.finfo(x0.dtype).eps
    if opts['root_tol'] is None:
        opts['root_tol'] = float(10 * eps)
    ctol = opts['ctol']
    tau = float(n) * float(eps)

    def g_scalar(x, g=g):
        return jnp.reshape(g(x), ())

    # Trace-time shape check (raises during tracing if g is not scalar).
    g_shape = jax.eval_shape(g, x0).shape
    g_size = 1
    for s in g_shape:
        g_size *= int(s)
    if g_shape != () and g_size != 1:
        raise ValueError(
            "qcqp_one requires a scalar inequality residual; "
            f"got shape {g_shape}."
        )

    H_f, b_f, c_f = _materialize_quadratic(f, x0)
    H_g, b_g, c_g = _materialize_quadratic(g_scalar, x0)

    # Rescale g so its curvature matches f.
    tr_f = jnp.trace(H_f)
    tr_g = jnp.trace(H_g)
    s = jnp.where(
        jnp.abs(tr_g) > tau,
        tr_f / tr_g,
        jnp.asarray(1.0, dtype=x0.dtype),
    )
    # Keep s positive for a positive combination H_f + s H_g.
    s = jnp.where(s > 0, s, jnp.asarray(1.0, dtype=x0.dtype))
    H_g_s = s * H_g
    b_g_s = s * b_g
    c_g_s = s * c_g

    a, b, p, q, L, Q = _simultaneous_diagonalize(
        H_f, H_g_s, b_f, b_g_s, n, float(eps),
    )

    y0 = _y_lam0(a, b, p, q, tau)
    y_inf = _y_lam_inf(a, b, p, q, tau)
    # phi uses scaled g; compare against scaled ctol.
    phi0 = _phi_from_y(y0, b, q, c_g_s) - s * ctol
    phi_inf = _phi_from_y(y_inf, b, q, c_g_s) - s * ctol

    unconstrained = phi0 <= 0.0
    infeasible = phi_inf > 0.0
    status_code = jnp.where(
        unconstrained, jnp.int32(0),
        jnp.where(infeasible, jnp.int32(2), jnp.int32(1)),
    )

    # Active branch: safeguarded Newton. Other branches ignore the result.
    def do_active(_):
        # Pass scaled ctol into the root finder via a local opts copy.
        opts_s = {**opts, 'ctol': s * ctol}
        return _safeguarded_newton(a, b, p, q, c_g_s, tau, opts_s, verbose)

    def skip_active(_):
        return jnp.asarray(0.0, dtype=jnp.float64), jnp.int32(0)

    lam_tilde, n_active = cond(status_code == 1, do_active, skip_active, None)

    y_active = _y_of_lam(a, b, p, q, lam_tilde, tau)
    y_opt = jnp.where(
        status_code == 0, y0,
        jnp.where(status_code == 2, y_inf, y_active),
    )
    x_opt = _x_from_y(y_opt, x0, L, Q)
    # Original (unscaled) objective / constraint for reporting.
    fin_f = f(x_opt)
    fin_g = g_scalar(x_opt)
    # Dual for the original (unscaled) constraint: lam = s * lam~.
    lam = s * lam_tilde
    fin_z = jnp.where(status_code == 1, lam, jnp.asarray(0.0, dtype=x0.dtype))
    niter = jnp.where(status_code == 1, n_active + 2, jnp.int32(2))

    if verbose > 0:
        jax.debug.print(
            'qcqp_one: status_code={sc}, f={f}, g={g}, z={z}',
            sc=status_code, f=fin_f, g=fin_g, z=fin_z,
        )

    return {
        'fin_x': x_opt,
        'fin_f': fin_f,
        'fin_g': jnp.atleast_1d(fin_g),
        'fin_h': jnp.zeros(0, dtype=x_opt.dtype),
        'fin_z': jnp.atleast_1d(fin_z),
        'niter': niter,
        'status_code': status_code,
    }
