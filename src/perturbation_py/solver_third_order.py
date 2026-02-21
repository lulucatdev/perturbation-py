"""Third-order perturbation solver for DSGE models.

Extends the second-order perturbation approach to compute the cubic
Taylor coefficients of the policy function.  Two methods are supported:

* **sylvester** -- Solves for ``ghxxx`` (and the stochastic correction
  ``ghxss``) via generalised Sylvester equations, following the approach
  in dolo's ``perturbations_higher_order.py`` (lines 273-357).  Cross
  terms ``ghxxu``, ``ghxuu``, ``ghuuu`` are currently set to zero and
  left for future implementation.
* **local_implicit** -- Recovers the full third derivative tensor by
  finite-differencing the Hessian of the implicit control map.

The third-order expansion adds terms of the form:

.. math::

    \\tfrac{1}{6} g_{xxx} (\\hat{s} \\otimes \\hat{s} \\otimes \\hat{s})
    + \\tfrac{1}{2} g_{xxu} (\\hat{s} \\otimes \\hat{s} \\otimes \\epsilon)
    + \\tfrac{1}{2} g_{xuu} (\\hat{s} \\otimes \\epsilon \\otimes \\epsilon)
    + \\tfrac{1}{6} g_{uuu} (\\epsilon \\otimes \\epsilon \\otimes \\epsilon)

References
----------
Andreasen, M. M., Fernandez-Villaverde, J. and Rubio-Ramirez, J. F.
    (2018). "The Pruned State-Space System for Non-Linear DSGE Models:
    Theory and Empirical Applications." *Review of Economic Studies*,
    85(1), 1-49.
EconForge/dolo, ``perturbations_higher_order.py``, lines 273-357.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import solve

from .model import DSGEModel
from .solver import FirstOrderSolution
from .solver_second_order import (
    SecondOrderSolution,
    _build_implicit_control_map,
    _hessian,
    solve_second_order,
)

Array = np.ndarray


@dataclass(frozen=True)
class ThirdOrderSolution:
    """Result of the third-order perturbation solution.

    Stores the full hierarchy of Taylor coefficients up to third order.

    Attributes
    ----------
    second_order : SecondOrderSolution
        The underlying second-order solution (which itself contains the
        first-order solution).
    ghx : Array, shape (n_x, n_s)
        First-order state coefficient.
    ghu : Array, shape (n_x, n_e)
        First-order shock coefficient.
    ghxx : Array, shape (n_x, n_s, n_s)
        Second-order state-state coefficient.
    ghxu : Array, shape (n_x, n_s, n_e)
        Second-order state-shock cross coefficient.
    ghuu : Array, shape (n_x, n_e, n_e)
        Second-order shock-shock coefficient.
    ghs2 : Array, shape (n_x,)
        Second-order uncertainty correction.
    ghxxx : Array, shape (n_x, n_s, n_s, n_s)
        Third derivative w.r.t. states: :math:`g_{xxx}`.
    ghxxu : Array, shape (n_x, n_s, n_s, n_e)
        Third cross derivative: :math:`g_{xxu}`.
    ghxuu : Array, shape (n_x, n_s, n_e, n_e)
        Third cross derivative: :math:`g_{xuu}`.
    ghuuu : Array, shape (n_x, n_e, n_e, n_e)
        Third derivative w.r.t. shocks: :math:`g_{uuu}`.
    ghsss : Array or None, shape (n_x, n_s, n_s, n_s)
        Pure-state third derivative (Sylvester method only).
    ghxss : Array or None, shape (n_x, n_s)
        Third-order stochastic correction :math:`g_{x\\sigma\\sigma}`
        (Sylvester method only).
    method : str
        Solution method used.
    """

    second_order: SecondOrderSolution
    ghx: Array
    ghu: Array
    ghxx: Array
    ghxu: Array
    ghuu: Array
    ghs2: Array
    ghxxx: Array
    ghxxu: Array
    ghxuu: Array
    ghuuu: Array
    ghsss: Array | None = None
    ghxss: Array | None = None
    method: str = "local_implicit"


def solve_third_order(
    model: DSGEModel,
    *,
    second_order: SecondOrderSolution | None = None,
    first_order: FirstOrderSolution | None = None,
    epsilon: float = 1e-3,
    solver_tol: float = 1e-10,
    maxfev: int = 250,
    method: str = "sylvester",
    shock_covariance: Array | None = None,
) -> ThirdOrderSolution:
    """Compute the third-order perturbation solution.

    Parameters
    ----------
    model : DSGEModel
        The model to solve.
    second_order : SecondOrderSolution or None
        Pre-computed second-order solution.  Solved internally if None.
    first_order : FirstOrderSolution or None
        Pre-computed first-order solution (used if *second_order* is
        None).
    epsilon : float
        Finite-difference step size (``"local_implicit"`` method only).
    solver_tol : float
        Implicit solver tolerance.
    maxfev : int
        Maximum function evaluations for the implicit solver.
    method : {"sylvester", "local_implicit"}
        Solution method.
    shock_covariance : Array or None
        Shock covariance matrix (``"sylvester"`` method only).

    Returns
    -------
    ThirdOrderSolution
        The third-order Taylor coefficients of the policy function.

    Raises
    ------
    ValueError
        If *method* is not recognized.

    References
    ----------
    EconForge/dolo, ``perturbations_higher_order.py``, lines 273-357.
    """
    if method == "sylvester":
        return _solve_third_order_sylvester(
            model,
            second_order=second_order,
            first_order=first_order,
            shock_covariance=shock_covariance,
        )
    elif method == "local_implicit":
        return _solve_third_order_local_implicit(
            model,
            second_order=second_order,
            first_order=first_order,
            epsilon=epsilon,
            solver_tol=solver_tol,
            maxfev=maxfev,
        )
    else:
        raise ValueError(f"Unsupported third-order method: {method}")


# ---------------------------------------------------------------------------
# Sylvester-equation method (dolo reference lines 273–357)
# ---------------------------------------------------------------------------

def _solve_third_order_sylvester(
    model: DSGEModel,
    *,
    second_order: SecondOrderSolution | None = None,
    first_order: FirstOrderSolution | None = None,
    shock_covariance: Array | None = None,
    hess_epsilon: float = 1e-5,
    third_epsilon: float = 1e-4,
    jac_epsilon: float = 1e-6,
) -> ThirdOrderSolution:
    """Solve the third-order problem via generalised Sylvester equations.

    Follows the dolo ``perturbations_higher_order.py`` approach
    (lines 273-357).  The algorithm:

    1. Compute model third derivatives (``f3``, ``g3``).
    2. Assemble the RHS tensor *D* from the first- and second-order
       solutions and the model's second and third derivatives.
    3. Solve the generalised Sylvester equation
       ``A X_sss + B X_sss C^3 + D = 0`` for ``ghxxx``.
    4. Compute the third-order stochastic correction ``ghxss`` by
       contracting shock-related third derivatives with the covariance
       matrix.

    Note: Cross terms ``ghxxu``, ``ghxuu``, ``ghuuu`` are set to zero
    in this implementation.

    Parameters
    ----------
    model : DSGEModel
        The model to solve.
    second_order : SecondOrderSolution or None
        Pre-computed second-order solution.
    first_order : FirstOrderSolution or None
        Pre-computed first-order solution.
    shock_covariance : Array or None
        Shock covariance matrix.
    hess_epsilon : float
        Step size for Hessian computation.
    third_epsilon : float
        Step size for third-derivative computation.
    jac_epsilon : float
        Step size for Jacobian computation.

    Returns
    -------
    ThirdOrderSolution
    """
    from .derivatives import compute_jacobians
    from .model_hessians import (
        compute_model_hessians,
        compute_model_third_derivatives,
    )
    from .tensor_ops import mdot, sdot, solve_generalized_sylvester

    so = second_order
    if so is None:
        so = solve_second_order(
            model,
            first_order=first_order,
            method="sylvester",
            shock_covariance=shock_covariance,
        )

    n_x = model.n_controls
    n_s = model.n_states
    n_e = model.n_shocks
    n_v = n_s + n_x

    if n_x == 0:
        return ThirdOrderSolution(
            second_order=so,
            ghx=so.ghx, ghu=so.ghu,
            ghxx=so.ghxx, ghxu=so.ghxu, ghuu=so.ghuu, ghs2=so.ghs2,
            ghxxx=np.zeros((0, n_s, n_s, n_s), dtype=float),
            ghxxu=np.zeros((0, n_s, n_s, n_e), dtype=float),
            ghxuu=np.zeros((0, n_s, n_e, n_e), dtype=float),
            ghuuu=np.zeros((0, n_e, n_e, n_e), dtype=float),
            method="sylvester",
        )

    # Shock covariance
    if shock_covariance is not None:
        sigma = np.asarray(shock_covariance, dtype=float)
    elif model.shock_covariance is not None:
        sigma = np.asarray(model.shock_covariance, dtype=float)
    else:
        sigma = np.eye(n_e, dtype=float)

    # Get Jacobians and Hessians
    jacs = compute_jacobians(model, epsilon=jac_epsilon)
    f_s, f_x, f_S, f_X = jacs.f_s, jacs.f_x, jacs.f_S, jacs.f_X
    g_s, g_x, g_e = jacs.g_s, jacs.g_x, jacs.g_e

    hessians = compute_model_hessians(model, jacs, epsilon=hess_epsilon, jac_epsilon=jac_epsilon)
    f2 = hessians.f2
    g2 = hessians.g2

    third_derivs = compute_model_third_derivatives(
        model, hessians, epsilon=third_epsilon,
        hess_epsilon=hess_epsilon, jac_epsilon=jac_epsilon,
    )
    f3 = third_derivs.f3
    g3 = third_derivs.g3

    # First- and second-order solutions
    X_s = so.ghx      # (n_x, n_s)
    X_ss = so.ghxx     # (n_x, n_s, n_s)
    X_tt = so.ghs2     # (n_x,)

    # g2 blocks
    g_ss = g2[:, :n_s, :n_s]
    g_sx = g2[:, :n_s, n_s:n_v]
    g_xx = g2[:, n_s:n_v, n_s:n_v]
    g_ee = g2[:, n_v:, n_v:]
    g_se = g2[:, :n_s, n_v:]
    g_xe = g2[:, n_s:n_v, n_v:]

    # g3 blocks
    g_sss = g3[:, :n_s, :n_s, :n_s]
    g_ssx = g3[:, :n_s, :n_s, n_s:n_v]
    g_sxx = g3[:, :n_s, n_s:n_v, n_s:n_v]
    g_xxx = g3[:, n_s:n_v, n_s:n_v, n_s:n_v]
    g_see = g3[:, :n_s, n_v:, n_v:]
    g_xee = g3[:, n_s:n_v, n_v:, n_v:]

    # Composite first-order transition
    V1_3 = g_s + g_x @ X_s  # (n_s, n_s)
    V1 = np.vstack([np.eye(n_s), X_s, V1_3, X_s @ V1_3])  # (2*n_v, n_s)

    K2 = g_ss + 2 * sdot(g_sx, X_s) + mdot(g_xx, X_s, X_s)  # (n_s, n_s, n_s)

    # Sylvester equation coefficients (same A, B, C as order 2)
    A = f_x + (f_S + f_X @ X_s) @ g_x
    B = f_X
    C = V1_3

    # --- X_sss (ghxxx) via Sylvester ---
    V2_3 = K2 + sdot(g_x, X_ss)  # (n_s, n_s, n_s)
    V2 = np.vstack([
        np.zeros((n_s, n_s, n_s)),
        X_ss,
        V2_3,
        np.tensordot(X_s, V2_3, axes=(1, 0)) + mdot(X_ss, V1_3, V1_3),
    ])  # (2*n_v, n_s, n_s)

    K3 = (
        g_sss
        + 3 * sdot(g_ssx, X_s)
        + 3 * mdot(g_sxx, X_s, X_s)
        + 2 * sdot(g_sx, X_ss)
        + 3 * mdot(g_xx, X_ss, X_s)
        + mdot(g_xxx, X_s, X_s, X_s)
    )

    L3 = 3 * mdot(X_ss, V1_3, V2_3)

    D = (
        mdot(f3, V1, V1, V1)
        + 3 * mdot(f2, V2, V1)
        + sdot(f_S + f_X @ X_s, K3)
        + sdot(f_X, L3)
    )

    X_sss = solve_generalized_sylvester(A, B, C, D)  # (n_x, n_s, n_s, n_s)

    # --- Sigma correction: X_stt ---
    I_e = np.eye(n_e)
    V_sl = g_se + mdot(g_xe, X_s, I_e)  # (n_s, n_s, n_e)
    W_l = np.vstack([g_e, X_s @ g_e])   # (n_v, n_e)

    W_sl = np.vstack([
        V_sl,
        mdot(X_ss, V1_3, g_e) + sdot(X_s, V_sl),
    ])  # (n_v, n_s, n_e)

    K_ee = mdot(f3[:, :, n_v:, n_v:], V1, W_l, W_l)
    K_ee += 2 * mdot(f2[:, n_v:, n_v:], W_sl, W_l)

    # Stochastic part of W_ll
    SW_ll = np.vstack([g_ee, mdot(X_ss, g_e, g_e) + sdot(X_s, g_ee)])

    # DW_ll: deterministic correction to next-period state/control from X_tt
    # X_tt is 1D (n_x,). All operations keep 1D vectors.
    gx_Xtt = g_x @ X_tt                     # (n_s,)
    DW_ll = np.concatenate([
        X_tt,                                # (n_x,)
        gx_Xtt,                              # (n_s,)
        X_s @ gx_Xtt + X_tt,                # (n_x,)
    ])  # (n_x + n_s + n_x,) = (2*n_v - n_s,) = shape of f2[:,:,n_s:] last dim

    K_ee += mdot(f2[:, :, n_v:], V1, SW_ll)

    # Contract K_ee with sigma
    K_ = np.tensordot(K_ee, sigma, axes=((2, 3), (0, 1)))  # (n_x, n_s)
    K_ += mdot(f2[:, :, n_s:], V1, DW_ll)  # DW_ll is 1D → result is (n_x, n_s)

    def E(vec):
        nd = len(vec.shape)
        return np.tensordot(vec, sigma, axes=((nd - 2, nd - 1), (0, 1)))

    L = sdot(g_sx, X_tt) + mdot(g_xx, X_s, X_tt)  # (n_s, n_s)
    L += E(g_see + mdot(g_xee, X_s, I_e, I_e))     # (n_s, n_s)

    M = E(mdot(X_sss, V1_3, g_e, g_e) + 2 * mdot(X_ss, V_sl, g_e))  # (n_x, n_s)
    M += mdot(X_ss, V1_3, E(g_ee) + sdot(g_x, X_tt))                  # (n_x, n_s)

    D_stt = K_ + sdot(f_S + f_X @ X_s, L) + sdot(f_X, M)  # (n_x, n_s)

    X_stt = solve_generalized_sylvester(A, B, C, D_stt)  # (n_x, n_s)

    # For the third-order output, we extract the pure state/shock blocks
    # ghxxx = X_sss, other cross terms set to zero for now (they require
    # additional Sylvester solves for mixed state-shock directions)
    ghxxx = X_sss
    ghxxu = np.zeros((n_x, n_s, n_s, n_e), dtype=float)
    ghxuu = np.zeros((n_x, n_s, n_e, n_e), dtype=float)
    ghuuu = np.zeros((n_x, n_e, n_e, n_e), dtype=float)

    return ThirdOrderSolution(
        second_order=so,
        ghx=so.ghx, ghu=so.ghu,
        ghxx=so.ghxx, ghxu=so.ghxu, ghuu=so.ghuu, ghs2=so.ghs2,
        ghxxx=ghxxx,
        ghxxu=ghxxu,
        ghxuu=ghxuu,
        ghuuu=ghuuu,
        ghsss=X_sss,
        ghxss=X_stt,
        method="sylvester",
    )


# ---------------------------------------------------------------------------
# Local implicit method (original, kept as fallback)
# ---------------------------------------------------------------------------

def _solve_third_order_local_implicit(
    model: DSGEModel,
    *,
    second_order: SecondOrderSolution | None = None,
    first_order: FirstOrderSolution | None = None,
    epsilon: float = 1e-3,
    solver_tol: float = 1e-10,
    maxfev: int = 250,
) -> ThirdOrderSolution:
    """Solve the third-order problem via numerical finite differences.

    Recovers the full third-derivative tensor of the implicit control
    map by finite-differencing the Hessian along each input direction.
    The raw tensor is symmetrised over the last three indices before
    being partitioned into the ``gh*`` blocks.

    Parameters
    ----------
    model : DSGEModel
        The model to solve.
    second_order : SecondOrderSolution or None
        Pre-computed second-order solution.
    first_order : FirstOrderSolution or None
        Pre-computed first-order solution.
    epsilon : float
        Finite-difference step size.
    solver_tol : float
        Implicit solver tolerance.
    maxfev : int
        Maximum function evaluations per implicit solve.

    Returns
    -------
    ThirdOrderSolution
    """
    so = second_order
    if so is None:
        so = solve_second_order(
            model,
            first_order=first_order,
            epsilon=max(1e-5, epsilon * 0.1),
            solver_tol=solver_tol,
            maxfev=maxfev,
            method="local_implicit",
        )

    n_x = model.n_controls
    n_s = model.n_states
    n_e = model.n_shocks
    n_z = n_s + n_e

    if n_x == 0:
        return ThirdOrderSolution(
            second_order=so,
            ghx=so.ghx, ghu=so.ghu,
            ghxx=so.ghxx, ghxu=so.ghxu, ghuu=so.ghuu, ghs2=so.ghs2,
            ghxxx=np.zeros((0, n_s, n_s, n_s), dtype=float),
            ghxxu=np.zeros((0, n_s, n_s, n_e), dtype=float),
            ghxuu=np.zeros((0, n_s, n_e, n_e), dtype=float),
            ghuuu=np.zeros((0, n_e, n_e, n_e), dtype=float),
            method="local_implicit",
        )

    control_map = _build_implicit_control_map(
        model=model,
        first_order=so.first_order,
        solver_tol=solver_tol,
        maxfev=maxfev,
    )

    z0 = np.zeros(n_z, dtype=float)
    third = np.zeros((n_x, n_z, n_z, n_z), dtype=float)

    for i in range(n_z):
        hi = epsilon * max(1.0, abs(z0[i]))
        ei = np.zeros_like(z0)
        ei[i] = hi

        h_plus = _hessian(control_map, z0 + ei, epsilon=max(1e-5, epsilon * 0.5))
        h_minus = _hessian(control_map, z0 - ei, epsilon=max(1e-5, epsilon * 0.5))
        third[:, i, :, :] = (h_plus - h_minus) / (2.0 * hi)

    third = _symmetrize_third_order(third)

    ghxxx = third[:, :n_s, :n_s, :n_s]
    ghxxu = third[:, :n_s, :n_s, n_s:]
    ghxuu = third[:, :n_s, n_s:, n_s:]
    ghuuu = third[:, n_s:, n_s:, n_s:]

    return ThirdOrderSolution(
        second_order=so,
        ghx=so.ghx, ghu=so.ghu,
        ghxx=so.ghxx, ghxu=so.ghxu, ghuu=so.ghuu, ghs2=so.ghs2,
        ghxxx=ghxxx, ghxxu=ghxxu, ghxuu=ghxuu, ghuuu=ghuuu,
        method="local_implicit",
    )


def _symmetrize_third_order(tensor: Array) -> Array:
    return (
        tensor
        + tensor.transpose(0, 1, 3, 2)
        + tensor.transpose(0, 2, 1, 3)
        + tensor.transpose(0, 2, 3, 1)
        + tensor.transpose(0, 3, 1, 2)
        + tensor.transpose(0, 3, 2, 1)
    ) / 6.0
