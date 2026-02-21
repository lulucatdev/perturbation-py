"""First-order perturbation solver using generalized Schur (QZ) decomposition.

The first-order perturbation approach linearises the equilibrium conditions
around the deterministic steady state and solves the resulting matrix
quadratic equation via the generalised Schur (QZ) decomposition.  The
Blanchard-Kahn (1980) conditions are checked to ensure a unique, stable
rational-expectations equilibrium: the number of unstable generalised
eigenvalues must equal the number of forward-looking (control) variables.

Algorithm outline
-----------------
1. Stack the linearised transition and arbitrage Jacobians into a
   generalised eigenvalue problem ``A z = B z``.
2. Apply ``scipy.linalg.ordqz`` to obtain the ordered QZ decomposition,
   partitioning eigenvalues into stable and unstable blocks.
3. Extract the policy matrix ``X_s`` from the stable block of the
   Z matrix: ``X_s = Z_{21} Z_{11}^{-1}``.
4. Recover the state transition ``T = g_s + g_x X_s`` and the
   shock-impact matrices.

References
----------
Blanchard, O. J. and Kahn, C. M. (1980). "The Solution of Linear
    Difference Models under Rational Expectations." *Econometrica*,
    48(5), 1305-1311.
Klein, P. (2000). "Using the generalized Schur form to solve a
    multivariate linear rational expectations model." *JEDC*, 24(10),
    1405-1423.
Villemot, S. (2011). "Solving rational expectations models at first
    order: what Dynare does." *Dynare Working Papers*, 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import ordqz

from .derivatives import Jacobians, compute_jacobians
from .model import DSGEModel
from .qz import BKDiagnostics, classify_bk_failure, compute_generalized_eigenvalues

Array = np.ndarray


class BlanchardKahnError(RuntimeError):
    """Raised when the Blanchard-Kahn (1980) conditions are not satisfied.

    The Blanchard-Kahn conditions require that the number of unstable
    generalised eigenvalues equals the number of forward-looking
    (control) variables.  Failure indicates either indeterminacy (too
    few unstable roots) or non-existence of a stable equilibrium (too
    many unstable roots).

    Attributes
    ----------
    diagnostics : BKDiagnostics
        Full diagnostic information including eigenvalues.
    reason : str
        One of ``"indeterminacy"``, ``"no_stable_equilibrium"``, or
        ``"rank_failure"``.
    unstable_count : int
        Number of unstable eigenvalues found.
    expected_unstable : int
        Number of unstable eigenvalues required (equal to ``n_controls``).
    eigenvalues : Array
        The moduli of all generalised eigenvalues.
    """

    def __init__(self, diagnostics: BKDiagnostics):
        self.diagnostics = diagnostics
        self.reason = diagnostics.reason
        self.unstable_count = diagnostics.unstable_count
        self.expected_unstable = diagnostics.expected_unstable
        self.eigenvalues = diagnostics.eigenvalues
        super().__init__(
            "Blanchard-Kahn condition failed: "
            f"reason={self.reason}, unstable={self.unstable_count}, expected={self.expected_unstable}."
        )


@dataclass(frozen=True)
class FirstOrderSolution:
    """Result of the first-order perturbation solution.

    Attributes
    ----------
    policy : Array, shape (n_x, n_s)
        Policy matrix ``X_s`` mapping state deviations to control
        deviations: :math:`\\hat{x}_t = X_s \\hat{s}_t`.
    transition : Array, shape (n_s, n_s)
        State transition matrix ``T = g_s + g_x X_s``.  The linearised
        law of motion is :math:`\\hat{s}_{t+1} = T \\hat{s}_t + R \\epsilon_t`.
    shock_impact : Array, shape (n_s, n_e)
        Shock-to-state impact matrix ``R = g_e + g_x X_e``.
    control_shock_impact : Array, shape (n_x, n_e)
        Shock-to-control impact matrix ``X_e``.
    eigenvalues : Array
        Moduli of the generalised eigenvalues of the system.
    unstable_eigenvalues : int
        Count of eigenvalues with modulus above the cutoff.
    blanchard_kahn_satisfied : bool
        Whether the Blanchard-Kahn order condition holds.
    """

    policy: Array
    transition: Array
    shock_impact: Array
    control_shock_impact: Array
    eigenvalues: Array
    unstable_eigenvalues: int
    blanchard_kahn_satisfied: bool

    @property
    def stable(self) -> bool:
        """Check whether the state transition is asymptotically stable.

        Returns
        -------
        bool
            True if all eigenvalues of the transition matrix have
            modulus strictly less than one.
        """
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(self.transition))))
        return spectral_radius < 1.0


def solve_first_order(
    model: DSGEModel,
    *,
    jacobians: Jacobians | None = None,
    eig_cutoff: float = 1.0 - 1e-8,
    singular_tol: float = 1e-12,
) -> FirstOrderSolution:
    """Compute the first-order perturbation solution via QZ decomposition.

    Linearises the model around its deterministic steady state and solves
    for the unique stable rational-expectations equilibrium using the
    ordered generalised Schur (QZ) decomposition.

    The linearised system is cast as a generalised eigenvalue problem:

    .. math::

        A z_{t+1} = B z_t

    where :math:`z = [s, x]^\\top`.  The QZ decomposition reorders
    eigenvalues so that unstable roots appear first.  The policy matrix
    is then extracted from the Z-matrix partition:
    :math:`X_s = Z_{21} Z_{11}^{-1}`.

    Parameters
    ----------
    model : DSGEModel
        The model to solve.
    jacobians : Jacobians or None
        Pre-computed Jacobians.  Computed on the fly if ``None``.
    eig_cutoff : float
        Eigenvalue modulus threshold for the stable/unstable partition.
        Eigenvalues with ``|lambda| > eig_cutoff`` are classified as
        unstable.  Default is ``1 - 1e-8``.
    singular_tol : float
        Tolerance for near-singular beta values in the generalised
        eigenvalue computation.

    Returns
    -------
    FirstOrderSolution
        The solved policy, transition, and shock-impact matrices.

    Raises
    ------
    BlanchardKahnError
        If the Blanchard-Kahn conditions are violated or the Z-matrix
        partition is rank-deficient.

    References
    ----------
    Blanchard and Kahn (1980), Klein (2000), Villemot (2011).
    """
    jac = jacobians if jacobians is not None else compute_jacobians(model)

    n_s = model.n_states
    n_x = model.n_controls

    A = np.block(
        [
            [np.eye(n_s), np.zeros((n_s, n_x))],
            [-jac.f_S, -jac.f_X],
        ]
    )
    B = np.block(
        [
            [jac.g_s, jac.g_x],
            [jac.f_s, jac.f_x],
        ]
    )

    def unstable(alpha: Array, beta: Array) -> Array:
        with np.errstate(divide="ignore", invalid="ignore"):
            ev = np.abs(alpha / beta)
        ev = np.where(np.abs(beta) < singular_tol, np.inf, ev)
        return ev > eig_cutoff

    unstable_selector: Any = unstable
    _, _, alpha, beta, _, z = ordqz(A, B, sort=unstable_selector, output="real")

    eigenvalues = compute_generalized_eigenvalues(alpha, beta, singular_tol)

    unstable_eigenvalues = int(np.sum(eigenvalues > eig_cutoff))
    bk_ok = unstable_eigenvalues == n_s
    if not bk_ok:
        raise BlanchardKahnError(
            BKDiagnostics(
                unstable_count=unstable_eigenvalues,
                expected_unstable=n_s,
                reason=classify_bk_failure(unstable_eigenvalues, n_s),
                eigenvalues=eigenvalues,
            )
        )

    z11 = z[:n_s, :n_s]
    z21 = z[n_s:, :n_s]
    cond = np.linalg.cond(z11)
    if not np.isfinite(cond) or cond > 1.0 / singular_tol:
        raise BlanchardKahnError(
            BKDiagnostics(
                unstable_count=unstable_eigenvalues,
                expected_unstable=n_s,
                reason="rank_failure",
                eigenvalues=eigenvalues,
            )
        )

    policy = np.linalg.solve(z11.T, z21.T).T
    transition = jac.g_s + jac.g_x @ policy

    f_future = jac.f_S + jac.f_X @ policy
    lhs = jac.f_x + f_future @ jac.g_x
    rhs = -(f_future @ jac.g_e)
    if lhs.size == 0:
        control_shock_impact = np.zeros((model.n_controls, model.n_shocks), dtype=float)
    else:
        control_shock_impact = np.linalg.solve(lhs, rhs)

    shock_impact = jac.g_e + jac.g_x @ control_shock_impact

    return FirstOrderSolution(
        policy=policy,
        transition=transition,
        shock_impact=shock_impact,
        control_shock_impact=control_shock_impact,
        eigenvalues=eigenvalues,
        unstable_eigenvalues=unstable_eigenvalues,
        blanchard_kahn_satisfied=True,
    )
