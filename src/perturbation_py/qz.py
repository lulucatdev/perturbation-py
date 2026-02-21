"""Blanchard-Kahn condition diagnostics and generalised eigenvalue utilities.

The Blanchard-Kahn (1980) conditions require that the number of unstable
(explosive) generalised eigenvalues of the linearised system equals the
number of forward-looking (non-predetermined, control) variables.  When
this condition fails, the rational-expectations equilibrium either does
not exist (too many unstable roots) or is indeterminate (too few).

This module provides helpers for computing generalised eigenvalues from
the QZ decomposition and classifying failures.

References
----------
Blanchard, O. J. and Kahn, C. M. (1980). "The Solution of Linear
    Difference Models under Rational Expectations." *Econometrica*,
    48(5), 1305-1311.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class BKDiagnostics:
    """Diagnostic information for a Blanchard-Kahn condition check.

    Attributes
    ----------
    unstable_count : int
        Number of generalised eigenvalues with modulus above the cutoff
        (i.e., unstable / explosive roots).
    expected_unstable : int
        Number of unstable eigenvalues required for a unique equilibrium,
        equal to the number of forward-looking (control) variables.
    reason : str
        Classification of the result: ``"ok"``, ``"indeterminacy"``
        (too few unstable), ``"no_stable_equilibrium"`` (too many
        unstable), or ``"rank_failure"`` (singular Z partition).
    eigenvalues : Array
        Moduli of all generalised eigenvalues.
    """

    unstable_count: int
    expected_unstable: int
    reason: str
    eigenvalues: Array


def compute_generalized_eigenvalues(
    alpha: Array, beta: Array, singular_tol: float
) -> Array:
    """Compute moduli of generalised eigenvalues from QZ output.

    Given the diagonal entries ``alpha`` and ``beta`` from the
    generalised Schur decomposition ``A = Q S Z*``, ``B = Q T Z*``,
    computes ``|alpha_i / beta_i|``.  Entries where ``|beta_i|`` is
    below *singular_tol* are mapped to infinity (representing infinite
    eigenvalues).

    Parameters
    ----------
    alpha : Array
        Diagonal of the S factor from the QZ decomposition.
    beta : Array
        Diagonal of the T factor from the QZ decomposition.
    singular_tol : float
        Threshold below which ``|beta_i|`` is considered zero.

    Returns
    -------
    Array
        Moduli of the generalised eigenvalues.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        eigenvalues = np.abs(alpha / beta)
    return np.where(np.abs(beta) < singular_tol, np.inf, eigenvalues)


def classify_bk_failure(unstable_count: int, expected_unstable: int) -> str:
    """Classify the type of Blanchard-Kahn condition failure.

    Parameters
    ----------
    unstable_count : int
        Number of unstable eigenvalues found.
    expected_unstable : int
        Number required for a unique equilibrium.

    Returns
    -------
    str
        ``"indeterminacy"`` if too few unstable roots (multiple
        equilibria), ``"no_stable_equilibrium"`` if too many, or
        ``"ok"`` if the condition is satisfied.
    """
    if unstable_count < expected_unstable:
        return "indeterminacy"
    if unstable_count > expected_unstable:
        return "no_stable_equilibrium"
    return "ok"
