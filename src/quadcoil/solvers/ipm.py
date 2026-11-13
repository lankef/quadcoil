"""
Mehrotra predictor-corrector primal-dual interior-point method (PDIP)
for small-scale NLP in JAX.

Problem
-------
    min  f_obj(x)
    s.t. g_ineq(x) <= 0   (m inequality constraints)

Convention: independent slacks ``s > 0`` satisfy ``g(x) + s = 0`` at
feasibility. Dual variables ``z > 0`` are the Lagrange multipliers.

Provides the QUADCOIL solver interface:
    solve_unconstrained_ipm
    solve_constrained_ipm

Equality constraints (``h_eq``) are not supported; a nonempty return
raises ``NotImplementedError`` at trace time.

Status codes
------------
0 solved, 4 max iterations, 5 numerical error.
"""

import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, hessian
from jax.lax import while_loop
import lineax as lx
from jax import config as config_jax
config_jax.update('jax_enable_x64', True)


# ── Shared helpers (fraction-to-boundary + scale-aware residuals) ────────

def fraction_to_boundary(x, dx, tau=0.995):
    """Largest alpha in (0,1] keeping x + alpha*dx > 0."""
    ratios = jnp.where(dx < 0.0, -x / dx, jnp.inf)
    return jnp.minimum(1.0, tau * jnp.min(ratios))


def kkt_residual(grad_f, A, g, s, z):
    r"""
    Scale-aware KKT residual (stationarity + primal feasibility).

    Returns ``(max(sd, pf), r_stat, r_prim)`` where
    ``sd = \|grad_f + A^T z\| / max(1, \|grad_f\|)``,
    ``pf = \|g + s\|          / max(1, \|g\|)``.
    """
    r_stat = grad_f + A.T @ z
    r_prim = g + s
    sd = jnp.linalg.norm(r_stat) / jnp.maximum(1.0, jnp.linalg.norm(grad_f))
    pf = jnp.linalg.norm(r_prim) / jnp.maximum(1.0, jnp.linalg.norm(g))
    return jnp.maximum(sd, pf), r_stat, r_prim


def _cho_factor_with_retry(M, delta, delta_max, max_tries=4):
    r"""
    Cholesky of ``M + delta I`` with inertia correction.

    Tries a geometric sequence of regularisations (``delta``,
    ``10 delta``, …) and returns the first finite factor.  The factor
    itself is not carried through a ``while_loop`` (``cho_factor``'s
    boolean ``lower`` flag is not a valid tracer).

    Returns ``(c, delta_used, ok)`` where ``c`` is the Cholesky factor
    suitable for ``cho_solve((c, True), ·)``.
    """
    n = M.shape[0]
    eye = jnp.eye(n, dtype=M.dtype)
    d0 = jnp.asarray(delta, dtype=M.dtype)
    # Geometric sequence: d0, 10 d0, … clipped to delta_max.
    scales = jnp.array([10.0 ** k for k in range(max_tries)], dtype=M.dtype)
    deltas = jnp.minimum(jnp.maximum(d0 * scales, 1e-8), delta_max)

    # Compute all candidate factors; pick the first finite one.
    def _one(d):
        c = jnp.linalg.cholesky(M + d * eye)  # lower-triangular; NaN if indefinite
        return c

    Cs = jax.vmap(_one)(deltas)                 # (max_tries, n, n)
    oks = jnp.isfinite(Cs).all(axis=(1, 2))     # (max_tries,)
    # Index of first success (or last if none).
    first = jnp.argmax(oks)                     # 0 if none succeed and all False
    ok = oks[first]
    c = Cs[first]
    d_used = deltas[first]
    # If nothing succeeded, still return last attempt (caller checks ok).
    return c, d_used, ok


# ── Solvers ──────────────────────────────────────────────────────────────

DEFAULT_IPM_OPTIONS = {
    'tol_kkt': 1e-6,
    'tau': 0.995,
    'delta_init': 1e-6,
    'delta_min': 1e-10,
    'delta_max': 1e-2,
    'merit_nu': 10.0,
    'armijo_c': 1e-4,
    'bt_max': 10,
}


def solve_unconstrained_ipm(
    init_params,
    fun,
    convex=False,
    maxiter: int = 100,
    solver_options=None,
    verbose=0,
):
    r'''
    Unconstrained optimization via a single Newton step (direct solve).

    Thin wrapper around :func:`solve_constrained_ipm` with no constraints.
    '''
    result = solve_constrained_ipm(
        x_init=init_params,
        f_obj=fun,
        convex=convex,
        maxiter=maxiter,
        solver_options=solver_options,
        verbose=verbose,
    )
    return {
        'fin_f': result['fin_f'],
        'fin_x': result['fin_x'],
        'niter': result['niter'],
        'fin_kkt_res': result['fin_kkt_res'],
        'converged': result['converged'],
        'status_code': result['status_code'],
    }


def solve_constrained_ipm(
    x_init,
    f_obj,
    h_eq=lambda x: jnp.zeros(0),
    g_ineq=lambda x: jnp.zeros(0),
    convex=False,
    maxiter: int = 100,
    solver_options=None,
    verbose=0,
):
    r'''
    Primal-dual interior-point method (Mehrotra predictor-corrector) for:

    .. math::

        \min_x f(x) \quad \text{s.t.}\; g(x) \le 0

    Parameters
    ----------
    x_init : ndarray, shape (n,)
    f_obj : Callable, x -> scalar
    h_eq : Callable, optional
        Equality constraints — **not supported**. Must return ``jnp.zeros(0)``.
    g_ineq : Callable, optional
        Inequality constraints, x -> (m,), g(x) <= 0.
    convex : bool
        When ``True``, starts with a smaller Hessian regularisation.
    solver_options : dict, optional
        Recognised keys: ``tol_kkt``, ``tau``, ``delta_init``,
        ``delta_min``, ``delta_max``, ``merit_nu``, ``armijo_c``, ``bt_max``.
    verbose : int, optional, default=0

    Returns
    -------
    status : dict
        ``fin_x``, ``fin_f``, ``fin_g``, ``fin_z``, ``fin_s``, ``niter``,
        ``fin_kkt_res``, ``converged``, ``status_code``.
    '''
    opts = {**DEFAULT_IPM_OPTIONS}
    if solver_options is not None:
        opts.update(solver_options)

    tol_kkt = opts['tol_kkt']
    tau = opts['tau']
    delta_init = opts['delta_init'] if not convex else min(opts['delta_init'], 1e-8)
    delta_min = opts['delta_min']
    delta_max = opts['delta_max']
    merit_nu = opts['merit_nu']
    armijo_c = opts['armijo_c']
    bt_max = int(opts['bt_max'])

    x_init = jnp.asarray(x_init, dtype=jnp.float64)
    n = x_init.shape[0]

    # F6: equality constraints are not supported.
    h0 = h_eq(x_init)
    if h0.shape[0] != 0:
        raise NotImplementedError(
            'solve_constrained_ipm does not support equality constraints; '
            f'h_eq returned shape {h0.shape}.'
        )

    g0 = g_ineq(x_init)
    m = g0.shape[0]

    # F5: choose jacfwd when m > n (tall Jacobian), else jacrev.
    g_jac = jacfwd(g_ineq) if m > n else jacrev(g_ineq)

    # ── m = 0: unconstrained — single Newton step ────────────────────────
    if m == 0:
        grad_f0 = grad(f_obj)(x_init)
        H = hessian(f_obj)(x_init) + jnp.float64(delta_init) * jnp.eye(n)
        tags = (lx.symmetric_tag, lx.positive_semidefinite_tag) if convex \
            else (lx.symmetric_tag,)
        H_op = lx.MatrixLinearOperator(H, tags=tags)
        sol = lx.linear_solve(
            H_op, -grad_f0,
            solver=lx.AutoLinearSolver(well_posed=convex),
            throw=False,
        )
        x_fin = x_init + sol.value
        kkt0 = jnp.linalg.norm(grad(f_obj)(x_fin))
        empty = jnp.zeros(0, dtype=jnp.float64)
        converged = kkt0 < tol_kkt
        return {
            'fin_x': x_fin,
            'fin_f': f_obj(x_fin),
            'fin_g': empty,
            'fin_z': empty,
            'fin_s': empty,
            'niter': jnp.int32(1),
            'fin_kkt_res': kkt0,
            'converged': converged,
            'status_code': jnp.where(converged, jnp.int32(0), jnp.int32(4)),
        }

    # ── m > 0: full PDIP with barrier ────────────────────────────────────
    # Independent slack: s >= max(-g, eps). Does NOT force feasibility.
    s_init = jnp.maximum(-g0, jnp.float64(1e-4))

    # F2: dual init via lstsq on A.T z ≈ -grad_f (no m×m solve).
    A0 = g_jac(x_init)
    grad_f0 = grad(f_obj)(x_init)
    z_ls, _, _, _ = jnp.linalg.lstsq(A0.T, -grad_f0, rcond=None)
    z_init = jnp.maximum(z_ls, 1.0)

    mu_b0 = jnp.dot(s_init, z_init) / m

    if verbose > 0:
        jax.debug.print(
            "IPM SOLVER: n={n}, m={m}, tol_kkt={tol}, maxiter={mi}",
            n=n, m=m, tol=tol_kkt, mi=maxiter,
        )

    init_state = {
        "x":          x_init,
        "s":          s_init,
        "z":          z_init,
        "mu_barrier": mu_b0,
        "delta":      jnp.float64(delta_init),
        "n_iter":     jnp.int32(0),
        "kkt_res":    jnp.float64(jnp.inf),
        "converged":  jnp.bool_(False),
        "grad_f":     grad_f0,
        "A":          A0,
        "g_val":      g0,
        "status_code": jnp.int32(4),
    }

    def not_converged(state):
        return (
            (state["n_iter"] == 0)
            | ((state["n_iter"] < maxiter) & ~state["converged"])
        )

    def ipm_step(state):
        x      = state["x"]
        s      = state["s"]
        z      = state["z"]
        mu_b   = state["mu_barrier"]
        delta  = state["delta"]
        g_val  = state["g_val"]
        grad_f = state["grad_f"]
        A      = state["A"]

        # ── Residuals ────────────────────────────────────────────────────
        # r_stat = ∇f + Aᵀ z
        # r_prim = g(x) + s
        r_stat = grad_f + A.T @ z
        r_prim = g_val + s
        zs     = z / s

        # ── Condensed n×n Newton matrix ──────────────────────────────────
        # W_c = ∇²L + δ I + Aᵀ diag(z/s) A
        #      where L = f + zᵀ g
        def lagrangian(xx):
            return f_obj(xx) + jnp.dot(z, g_ineq(xx))

        H_lag = hessian(lagrangian)(x)
        W_c_mat = H_lag + A.T @ (zs[:, None] * A)

        # F4: always Cholesky with inertia correction (bump δ until finite).
        c_fac, delta_used, factor_ok = _cho_factor_with_retry(
            W_c_mat, delta, delta_max,
        )

        def _solve(rhs_v):
            return jax.scipy.linalg.cho_solve((c_fac, True), rhs_v)

        # ── Predictor (affine-scaling direction) ─────────────────────────
        # Solve W_c Δx_aff = -(∇f + Aᵀ (z/s) r_prim)
        # Then  Δs_aff = -(r_prim + A Δx_aff)
        #       Δz_aff = -z - (z/s) Δs_aff
        rhs_x_aff = -(grad_f + A.T @ (zs * r_prim))
        dx_aff = _solve(rhs_x_aff)
        ds_aff = -(r_prim + A @ dx_aff)
        dz_aff = -z - zs * ds_aff

        # ── Mehrotra centering parameter ─────────────────────────────────
        # σ = (μ_aff / μ)³,  μ_aff from the largest affine step that
        # keeps (s, z) positive.
        alpha_p_aff = fraction_to_boundary(s, ds_aff, tau)
        alpha_d_aff = fraction_to_boundary(z, dz_aff, tau)
        alpha_aff   = jnp.minimum(alpha_p_aff, alpha_d_aff)
        mu_aff = jnp.dot(s + alpha_aff * ds_aff, z + alpha_aff * dz_aff) / m
        sigma  = jnp.clip((mu_aff / (mu_b + 1e-30)) ** 3, 0.0, 1.0)

        # ── Corrector (F1: full Mehrotra with cross term) ────────────────
        # Complementarity RHS:  s∘z + Δs_aff∘Δz_aff − σ μ 1 = 0
        # ⇒  Δz contribution involves (σμ − Δs_aff∘Δz_aff)/s
        cross = ds_aff * dz_aff
        cent_term = (sigma * mu_b - cross) / s
        rhs_x_cor = -(grad_f + A.T @ (zs * r_prim) + A.T @ cent_term)
        dx = _solve(rhs_x_cor)
        ds = -(r_prim + A @ dx)
        dz = cent_term - zs * ds - z

        factor_failed = (~factor_ok) | (~jnp.isfinite(dx).all())

        # ── Fraction-to-boundary step lengths ────────────────────────────
        # Common (x, s) step; separate z step.
        alpha_p_max = fraction_to_boundary(s, ds, tau)
        alpha_d_max = fraction_to_boundary(z, dz, tau)

        # ── Merit function for Armijo globalization (F3) ─────────────────
        # φ(x,s) = f(x) + ν ‖g(x)+s‖₁ − μ Σ log s_i
        # Allows infeasible starts; s stays independent (no slack reset).
        def merit(xx, ss):
            gg = g_ineq(xx)
            return (
                f_obj(xx)
                + merit_nu * jnp.sum(jnp.abs(gg + ss))
                - mu_b * jnp.sum(jnp.log(jnp.maximum(ss, 1e-300)))
            )

        phi0 = merit(x, s)
        # Directional derivative estimate along (dx, ds):
        # Dφ ≈ ∇f·dx + ν sign(r_prim)·(A dx + ds) − μ 1ᵀ (ds / s)
        sign_rp = jnp.sign(r_prim)
        sign_rp = jnp.where(sign_rp == 0, 1.0, sign_rp)
        dphi = (
            jnp.dot(grad_f, dx)
            + merit_nu * jnp.dot(sign_rp, A @ dx + ds)
            - mu_b * jnp.sum(ds / s)
        )
        # If the Newton direction is not a descent direction for the
        # merit, fall back to pure fraction-to-boundary (still safe).
        dphi = jnp.minimum(dphi, -1e-8 * jnp.abs(phi0))

        # Armijo backtracking; optional second-order correction via W_c
        # when the merit test fails (F2: no m×m solve).
        def _bt_continue(bt_state):
            _alpha, _x_b, _s_b, accepted, bt_iter = bt_state
            return (~accepted) & (bt_iter < bt_max)

        def _bt_body(bt_state):
            alpha, x_best, s_best, accepted, bt_iter = bt_state
            x_t = x + alpha * dx
            s_t = s + alpha * ds
            # Keep s positive even if fraction-to-boundary was optimistic.
            s_t = jnp.maximum(s_t, 1e-300)
            phi_t = merit(x_t, s_t)
            ok = (phi_t <= phi0 + armijo_c * alpha * dphi) & jnp.isfinite(phi_t)

            # Second-order correction on the condensed system when the
            # trial point is rejected: Δx_soc = -W_c⁻¹ Aᵀ (z/s) g⁺.
            g_t = g_ineq(x_t)
            g_corr = jnp.maximum(g_t + 1e-6, 0.0)
            dx_soc = _solve(-A.T @ (zs * g_corr))
            x_soc = x_t + dx_soc
            s_soc = jnp.maximum(-(g_ineq(x_soc)), 1e-8)  # only for SOC trial
            # Prefer the SOC point if it improves the merit; otherwise
            # keep the plain trial when it passes Armijo.
            phi_soc = merit(x_soc, s_soc)
            use_soc = (~ok) & (phi_soc < phi_t) & jnp.isfinite(phi_soc)
            x_cand = jnp.where(use_soc, x_soc, x_t)
            s_cand = jnp.where(use_soc, jnp.maximum(s_t, 1e-8), s_t)
            # Accept if either plain Armijo or SOC improved over phi0.
            accept_now = ok | (use_soc & (phi_soc <= phi0))
            x_best = jnp.where(accept_now, x_cand, x_best)
            s_best = jnp.where(accept_now, s_cand, s_best)
            accepted = accepted | accept_now
            return (alpha * 0.5, x_best, s_best, accepted, bt_iter + 1)

        _, x_new, s_new, found, _ = while_loop(
            _bt_continue, _bt_body,
            (alpha_p_max, x, s, jnp.bool_(False), jnp.int32(0)),
        )
        # If backtracking never accepted, take a tiny fraction-to-boundary
        # step so the iterate still moves.
        fallback_a = jnp.minimum(alpha_p_max, 1e-2)
        x_new = jnp.where(found, x_new, x + fallback_a * dx)
        s_new = jnp.where(found, s_new, jnp.maximum(s + fallback_a * ds, 1e-300))

        # Dual step (independent fraction-to-boundary).
        z_new = jnp.maximum(z + alpha_d_max * dz, 1e-300)

        mu_b_new = jnp.dot(s_new, z_new) / m

        # ── Regularisation schedule ──────────────────────────────────────
        delta_new = jnp.where(
            factor_failed,
            jnp.minimum(jnp.maximum(delta_used * 10.0, 1e-8), jnp.float64(delta_max)),
            jnp.maximum(delta_used / 3.0, jnp.float64(delta_min)),
        )

        # ── NaN guard ────────────────────────────────────────────────────
        step_ok = (
            jnp.isfinite(x_new).all()
            & jnp.isfinite(s_new).all()
            & jnp.isfinite(z_new).all()
        )
        x_new    = jnp.where(step_ok, x_new,    x)
        s_new    = jnp.where(step_ok, s_new,    s)
        z_new    = jnp.where(step_ok, z_new,    z)
        mu_b_new = jnp.where(step_ok, mu_b_new, mu_b)

        # ── KKT check + cache derivatives for next iteration ─────────────
        # F5: single g evaluation (no duplicated g_ineq calls).
        grad_f_new = grad(f_obj)(x_new)
        A_new      = g_jac(x_new)
        g_new      = g_ineq(x_new)
        res_sf, _, _ = kkt_residual(grad_f_new, A_new, g_new, s_new, z_new)
        # F7: scale-aware convergence — max of relative residual and μ.
        kkt_res_new   = jnp.maximum(res_sf, mu_b_new)
        converged_new = kkt_res_new < tol_kkt
        # status: 0 solved, 5 numerical if step failed hard, else 4 (still iterating)
        status_new = jnp.where(
            converged_new, jnp.int32(0),
            jnp.where(~step_ok, jnp.int32(5), jnp.int32(4)),
        )

        if verbose > 1:
            jax.debug.print(
                "IPM iter {it}: kkt={kkt:.3e}  mu_b={mu:.3e}  "
                "ap={ap:.3f}  ad={ad:.3f}  delta={d:.2e}  sigma={sg:.3f}",
                it=state["n_iter"] + 1,
                kkt=kkt_res_new, mu=mu_b_new,
                ap=alpha_p_max, ad=alpha_d_max, d=delta_new, sg=sigma,
            )
        elif verbose > 0:
            jax.debug.print(
                "IPM iter {it}: kkt={kkt:.3e}  mu_b={mu:.3e}",
                it=state["n_iter"] + 1, kkt=kkt_res_new, mu=mu_b_new,
            )

        return {
            "x":          x_new,
            "s":          s_new,
            "z":          z_new,
            "mu_barrier": mu_b_new,
            "delta":      delta_new,
            "n_iter":     state["n_iter"] + 1,
            "kkt_res":    kkt_res_new,
            "converged":  converged_new,
            "grad_f":     grad_f_new,
            "A":          A_new,
            "g_val":      g_new,
            "status_code": status_new,
        }

    final = while_loop(not_converged, ipm_step, init_state)

    if verbose > 0:
        jax.debug.print(
            "IPM DONE: iters={it}  kkt_res={kkt:.3e}  converged={cv}  status={st}",
            it=final["n_iter"], kkt=final["kkt_res"],
            cv=final["converged"], st=final["status_code"],
        )

    x_opt = final["x"]
    status_code = final["status_code"]
    # If we exited by maxiter without converging, keep status 4.
    status_code = jnp.where(
        final["converged"], jnp.int32(0),
        jnp.where(status_code == 5, jnp.int32(5), jnp.int32(4)),
    )

    return {
        'fin_x':       x_opt,
        'fin_f':       f_obj(x_opt),
        'fin_g':       final["g_val"],
        'fin_z':       final["z"],
        'fin_s':       final["s"],
        'niter':       final["n_iter"],
        'fin_kkt_res': final["kkt_res"],
        'converged':   final["converged"],
        'status_code': status_code,
    }
