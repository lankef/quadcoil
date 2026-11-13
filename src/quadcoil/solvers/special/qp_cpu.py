"""
Clarabel CPU backend for dense QP / SOCP.

Primary path
------------
``jax.pure_callback`` wrapping the ``clarabel`` Python wheel.  This is
jittable, keeps shapes fixed (no recompile on data changes), and works
on CPU.  Prefer ``JAX_PLATFORMS=cpu`` when calling this backend.

Optional FFI path
-----------------
If the ``quadcoil_clarabel_ffi`` extension (built from ``ffi/clarabel``)
is importable, it is preferred via ``jax.ffi.ffi_call``.  The pure-
callback path is the fallback and the reference for tests.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .qp_form import ConicProblem, CONIC_STATUS


def clarabel_available():
    """True if either the FFI extension or the clarabel wheel is importable."""
    try:
        import quadcoil_clarabel_ffi  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import clarabel  # noqa: F401
        return True
    except ImportError:
        return False


def _cones_to_clarabel(cones):
    import clarabel
    out = []
    for block in cones:
        if block[0] == 'l':
            m_l = int(block[1])
            if m_l > 0:
                out.append(clarabel.NonnegativeConeT(m_l))
        elif block[0] == 'q':
            dim, count = int(block[1]), int(block[2])
            for _ in range(count):
                out.append(clarabel.SecondOrderConeT(dim))
        else:
            raise ValueError(f'Unknown cone block {block!r}')
    return out


def _clarabel_status_code(status):
    """Map a Clarabel status object / string to the shared integer codes."""
    name = str(status)
    # Clarabel Python statuses look like "Solved", "PrimalInfeasible", …
    if 'Solved' in name and 'Inaccurate' not in name and 'Almost' not in name:
        return 0
    if 'AlmostSolved' in name or 'SolvedInaccurate' in name or 'Inaccurate' in name:
        return 1
    if 'PrimalInfeasible' in name:
        return 2
    if 'DualInfeasible' in name:
        return 3
    if 'MaxIterations' in name or 'MaxTime' in name:
        return 4
    return 5


def _solve_clarabel_numpy(P, q, G, h, A, b, cone_spec, settings_vec):
    """Host-side Clarabel solve.  All inputs are NumPy arrays."""
    import clarabel
    from scipy import sparse

    n = q.shape[0]
    m = h.shape[0]
    p = b.shape[0]

    # Clarabel expects P as upper-triangular CSC, and a single constraint
    # matrix [A; G] with cones [ZeroCone(p), Nonneg / SOC …].
    P_csc = sparse.csc_matrix(np.triu(P))
    if p > 0:
        A_full = sparse.csc_matrix(np.vstack([A, G]))
        b_full = np.concatenate([b, h])
        cones = [clarabel.ZeroConeT(p)] + _cones_to_clarabel(cone_spec)
    else:
        A_full = sparse.csc_matrix(G)
        b_full = np.asarray(h, dtype=np.float64)
        cones = _cones_to_clarabel(cone_spec)

    settings = clarabel.DefaultSettings()
    # settings_vec: [tol, max_iter, verbose, equilibrate, unused…]
    if settings_vec is not None and len(settings_vec) >= 2:
        settings.tol_gap_abs = float(settings_vec[0])
        settings.tol_gap_rel = float(settings_vec[0])
        settings.tol_feas = float(settings_vec[0])
        settings.max_iter = int(settings_vec[1])
    if settings_vec is not None and len(settings_vec) >= 3:
        settings.verbose = bool(settings_vec[2] > 0.5)

    solver = clarabel.DefaultSolver(P_csc, np.asarray(q), A_full, b_full, cones, settings)
    sol = solver.solve()

    status = _clarabel_status_code(sol.status)
    x = np.asarray(sol.x, dtype=np.float64)
    # Clarabel z is the full dual for [A; G]; split into y (eq) and z (ineq).
    z_full = np.asarray(sol.z, dtype=np.float64)
    s_full = np.asarray(sol.s, dtype=np.float64)
    if p > 0:
        y = z_full[:p]
        z = z_full[p:]
        s = s_full[p:]
    else:
        y = np.zeros(0, dtype=np.float64)
        z = z_full
        s = s_full

    if not np.isfinite(x).all():
        x = np.full(n, np.nan)
        z = np.full(m, np.nan)
        s = np.full(m, np.nan)
        y = np.full(p, np.nan)
        status = 5

    info = np.array([
        float(getattr(sol, 'obj_val', np.nan)),
        float(getattr(sol, 'obj_val_dual', np.nan)),
        float(status),
        float(getattr(sol, 'iterations', -1)),
        float(getattr(sol, 'solve_time', np.nan)),
    ], dtype=np.float64)
    return x, y, z, s, info


def _try_ffi_solve(prob, cones, opts, verbose):
    """Attempt the compiled XLA FFI extension; return None if unavailable."""
    try:
        import quadcoil_clarabel_ffi as ext
    except ImportError:
        return None

    # Register once.
    if not getattr(_try_ffi_solve, '_registered', False):
        jax.ffi.register_ffi_target(
            'quadcoil_clarabel',
            ext.clarabel_solve(),
            platform='cpu',
        )
        _try_ffi_solve._registered = True

    P, q, G, h, A, b = prob.P, prob.q, prob.G, prob.h, prob.A, prob.b
    n, m, p = q.shape[0], h.shape[0], b.shape[0]
    tol = float((opts or {}).get('tol', 1e-8))
    maxiter = float((opts or {}).get('maxiter', 50))
    settings = jnp.array([tol, maxiter, float(verbose), 1.0, 0.0])

    # Pack cone spec as a flat int32 attribute vector:
    # [n_blocks, tag0, a0, b0, tag1, a1, b1, …] with tag 0=l, 1=q.
    flat = [len(cones)]
    for block in cones:
        if block[0] == 'l':
            flat.extend([0, int(block[1]), 0])
        else:
            flat.extend([1, int(block[1]), int(block[2])])
    cone_attr = np.asarray(flat, dtype=np.int32)

    out_types = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((p,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((5,), jnp.float64),
    )
    call = jax.ffi.ffi_call(
        'quadcoil_clarabel',
        out_types,
        vmap_method='sequential',
    )
    # Attributes must be compile-time constants — pass cone_attr via the
    # handler's Attr binding; for a first cut we pass it as an int32 buffer.
    x, y, z, s, info = call(P, q, A, b, G, h, settings,
                            cone_attr=cone_attr)
    return x, y, z, s, info


def solve_conic_clarabel(prob, cones, opts=None, verbose=0):
    r'''
    Solve a dense QP/SOCP with Clarabel on the CPU.

    Returns the same dictionary layout as :func:`solve_conic_ipm`.
    '''
    options = dict(opts or {})
    P, q, G, h, A, b = prob.P, prob.q, prob.G, prob.h, prob.A, prob.b
    n = q.shape[0]
    m = h.shape[0]
    p = b.shape[0]

    # Prefer FFI when the extension is present.
    ffi_result = _try_ffi_solve(prob, cones, options, verbose)
    if ffi_result is not None:
        x, y, z, s, info = ffi_result
        status = jnp.asarray(info[2], dtype=jnp.int32)
        niter = jnp.asarray(info[3], dtype=jnp.int32)
        pobj = 0.5 * jnp.dot(x, P @ x) + jnp.dot(q, x)
        return {
            'fin_x': x,
            'fin_s': s,
            'fin_z': z,
            'fin_y': y,
            'fin_f': pobj,
            'fin_g': G @ x - h,
            'fin_h': (A @ x - b) if p > 0 else jnp.zeros(0, dtype=x.dtype),
            'status_code': status,
            'niter': niter,
            'pres': jnp.asarray(jnp.nan),
            'dres': jnp.asarray(jnp.nan),
            'gap': jnp.asarray(jnp.nan),
            'pobj': pobj,
        }

    # Fallback: pure_callback around the clarabel Python wheel.
    try:
        import clarabel  # noqa: F401
        from scipy import sparse  # noqa: F401
    except ImportError as e:
        raise ImportError(
            'Clarabel backend requires the clarabel package '
            '(and scipy for the CSC conversion), or the '
            'quadcoil_clarabel_ffi extension.'
        ) from e

    tol = float(options.get('tol', 1e-8))
    maxiter = float(options.get('maxiter', 50))
    settings_vec = np.array([tol, maxiter, float(verbose), 1.0, 0.0])

    # Cone spec must be a concrete Python tuple for the callback closure.
    cone_spec = tuple(cones)

    def _host(P_, q_, G_, h_, A_, b_):
        return _solve_clarabel_numpy(
            np.asarray(P_), np.asarray(q_),
            np.asarray(G_), np.asarray(h_),
            np.asarray(A_), np.asarray(b_),
            cone_spec, settings_vec,
        )

    result_shape = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((p,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((5,), jnp.float64),
    )
    x, y, z, s, info = jax.pure_callback(
        _host, result_shape, P, q, G, h, A, b,
        vmap_method='sequential',
    )
    status = jnp.asarray(info[2], dtype=jnp.int32)
    niter = jnp.asarray(info[3], dtype=jnp.int32)
    pobj = 0.5 * jnp.dot(x, P @ x) + jnp.dot(q, x)

    if verbose > 0:
        jax.debug.print(
            "Clarabel DONE: status={st} iters={it} pobj={f}",
            st=status, it=niter, f=pobj,
        )

    return {
        'fin_x': x,
        'fin_s': s,
        'fin_z': z,
        'fin_y': y,
        'fin_f': pobj,
        'fin_g': G @ x - h,
        'fin_h': (A @ x - b) if p > 0 else jnp.zeros(0, dtype=x.dtype),
        'status_code': status,
        'niter': niter,
        'pres': jnp.asarray(jnp.nan),
        'dres': jnp.asarray(jnp.nan),
        'gap': jnp.asarray(jnp.nan),
        'pobj': pobj,
    }
