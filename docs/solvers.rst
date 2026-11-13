More on solvers
===============

QUADCOIL uses specialized solvers when it recognizes a convex problem class,
and general-purpose solvers for non-convex problems (and for convex problems
when special detection is disabled or the problem is unsupported). Routing
is automatic under ``enable_special=True`` (the default); set
``enable_special=False`` to force a generic backend chosen by ``solver``.

Convex solvers
--------------

These solvers live under ``quadcoil.solvers.special`` and are selected by
problem classification, not by the ``solver`` string.

``qcqp_one``
~~~~~~~~~~~

Handles a single quadratic objective with exactly one quadratic
``'<='`` constraint. The solver diagonalizes the two quadratics and finds
the active dual multiplier by a safeguarded 1-D root search.

``socp``
~~~~~~~~

Quadratic or linear objectives with a mix of linear ``'<='`` / ``'>='``
constraints and at least one convex quadratic ``'<='`` constraint
(no ``*_slack``). A typical case is constrained REGCOIL: minimize
``f_K`` subject to pointwise ``K_theta`` and ``f_B <= target``.
Default backend ``special_backend='ipm_conic'``.

Canonicalization
^^^^^^^^^^^^^^^^

``canonicalize_socp`` expands the objective and constraints about the
flat preconditioned point ``x0`` (usually ``x_flat_init``) into the
dense conic form

.. math::

    \min_x \tfrac12 x^\top P x + q^\top x
    \quad\text{s.t.}\quad
    G x + s = h,\quad s \in K.

Here ``P`` and ``q`` come from the Hessian and gradient of the
objective at ``x0`` (rewritten in absolute coordinates). Linear
constraint blocks become nonnegative-orthant rows
``G_\ell x \le h_\ell`` (a ``'>='`` quantity is already sign-flipped
inside the residual ``g``). Each *scalar* quadratic inequality

.. math::

    g_i(x) = \tfrac12 x^\top H x + a^\top x + c \le 0
    \qquad (H \succeq 0)

is factored as ``H = R^\top R`` (symmetric eigendecomposition, negative
eigenvalues clipped to zero; ``R`` is kept ``n \times n`` so shapes stay
static). With ``w = -2(a^\top x + c)``, the constraint
``\|R x\|^2 \le w`` is equivalent to Lorentz membership

.. math::

    \Big(\tfrac{w+1}{2},\; R x,\; \tfrac{w-1}{2}\Big)
    \in \mathcal{Q}^{n+2},

which as ``G_q x + s = h_q`` with ``s \in \mathcal{Q}`` is the block

.. math::

    G_q =
    \begin{bmatrix} a^\top \\ -R \\ a^\top \end{bmatrix},
    \qquad
    h_q =
    \begin{bmatrix} \tfrac12 - c \\ 0 \\ -c - \tfrac12 \end{bmatrix}.

Rows of ``G`` are ordered orthant-first, then one SOC of size ``n+2``
per quadratic constraint, i.e. cone spec
``(('l', m_\ell), ('q', n+2, k))`` (the orthant block is omitted when
``m_\ell = 0``). For the KKT adjoint, orthant duals map one-to-one; the
multiplier of an original quadratic row is recovered as
``\lambda = z_0 + z_{\mathrm{last}}`` of its cone
(``G_q^\top z = \lambda \nabla g_i``).

Initial condition
^^^^^^^^^^^^^^^^^

After Ruiz equilibration of ``(P, q, G, h)``, the Mehrotra IPM builds
its own interior start (NESCOIL / ``phi_init`` is *not* used as the
``(x, s, z)`` guess; ``x0`` only enters canonicalization). Solve the
regularized least-squares system

.. math::

    \bigl(P + G^\top G\bigr)\, x
    = -q + G^\top h

(with ``A^\top A`` / ``A^\top b`` added if equalities are present), set
``s = h - G x`` and ``z = -s``, then shift into the cone interior:
orthant components by ``(1 - \min)_+``, and each second-order cone
along ``e = (1, 0, \ldots)`` using the minimum Lorentz eigenvalue.
If ``G`` were empty this would be the unconstrained minimizer of the
quadratic objective; with inequalities it is a compromise between that
Newton point and ``G x \approx h``.

Algorithm
^^^^^^^^^

The SOCP branch of ``solve_conic_ipm`` is a dense Mehrotra primal-dual
interior-point method on the product cone
``K = \mathbb{R}_+^{m_\ell} \times \mathcal{Q}^{n+2} \times \cdots``
(barrier degree ``\nu``). Each iteration:

1. Form primal / dual residuals and complementarity measure ``\mu``.
2. Build a Nesterov–Todd scaling ``W`` (orthant: ``D = z/s``; SOC: NT
   scaling of each cone).
3. Assemble the condensed Newton matrix
   ``M = P + G^\top W^{-2} G`` — orthant contribution
   ``G_\ell^\top \mathrm{diag}(D) G_\ell``, SOC contribution a signed
   diagonal plus a rank-``n_{\mathrm{cones}}`` update — and solve for
   the affine (predictor) direction by Cholesky with ``\delta``-retries.
4. Choose a Mehrotra centering parameter ``\sigma`` from the affine step
   length; form the corrector RHS with the Jordan cross term; recover
   ``(\Delta x, \Delta s, \Delta z)``.
5. Take a common step that stays in the product cone (orthant
   fraction-to-boundary; SOC via the Lorentz-boundary quadratic in the
   step length).

Termination uses relative primal / dual residuals and duality gap,
plus Farkas certificates for infeasibility. Status codes match the
Clarabel backend (``solved``, ``solved_inaccurate``, …).

``qp``
~~~~~~

Quadratic or linear objectives with only linear ``'<='`` / ``'>='``
constraints (no quadratic inequalities). Detected before ``socp`` in
``classify_problem``.

Canonicalization (``canonicalize_qp``) uses the same absolute
objective ``P``, ``q`` as above. Affine inequalities become
``G = Dg(x_0)``, ``h = G x_0 - g(x_0)``, with cone spec
``(('l', m),)`` only. The problem is then solved by the *orthant*
branch of the same ``ipm_conic`` Mehrotra method described under
``socp`` (no second-order cones or NT scaling).

Clarabel
~~~~~~~~

Optional CPU backend for the same QP / SOCP cone form, selected with
``special_backend='clarabel'``. Useful as a reference or host-side
fallback; not available on GPU.

Non-convex solvers
------------------

These are the generic backends selected by the ``solver`` argument when
the special path is not used.

Augmented Lagrangian L-BFGS (``auglag-lbfgs``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Default generic solver. Outer augmented-Lagrangian updates with an
L-BFGS inner loop. Robust on a wide range of smooth constrained
problems.

Interior-point method (``ipm``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Mehrotra-style primal-dual interior-point method for general smooth
NLPs with inequality constraints. Suitable when a barrier formulation
is preferred over AugLag.

SLSQP (``slsqp``)
~~~~~~~~~~~~~~~~

Sequential quadratic programming via ``slsqp-jax``. An alternative
generic constrained optimizer for smooth problems.
